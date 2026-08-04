"""Synchronous whole-game GRPO for the Catan policy.

Each update freezes the current policy, rolls out groups of complete games on
fresh training seeds, and normalizes terminal win-plus-VP rewards within each
seed group. Every valid model completion receives its game's advantage. The
policy is updated only after the entire rollout batch has finished.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.prompts import SYSTEM_PROMPT
from catan_llm.whole_game import (
    DEFAULT_INVALID_RETRIES,
    DEFAULT_MAX_TURNS,
    DEFAULT_VP_COEF,
    OPPONENT_POLICY,
    DecisionSample,
    Rollout,
    SampledCompletion,
    assign_group_advantages,
    hero_vps,
    make_rollouts,
    rollout_games,
    trainable_samples,
    training_seeds,
    vp_margin,
)
from run_dpo import (
    assert_fast_linear_attention,
    load_text_only_qwen35,
    package_versions,
    sha256,
)
from run_grpo import completion_token_logprobs, render_prompt, stop_token_ids

DEFAULT_GROUP_SIZE = 8
DEFAULT_GROUPS_PER_UPDATE = 4
DEFAULT_UPDATES = 25
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LR = 1e-6
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_POLICY_EPOCHS = 1
DEFAULT_SEED = 42
DEFAULT_SEED_START = EVAL_SEED_LIMIT
MAX_NEW_TOKENS = 32
MAX_PROMPT_TOKENS = 8192
GRAD_CLIP_NORM = 1.0
INITIAL_ADAPTER_SHA256 = (
    "faee1bea00eb2a0b51a8cefcad1af9bd2a4c0afaf12668308af0f2f1ff408ab0"
)


def clipped_surrogate_loss(new_logprob, old_logprob, advantage, clip_epsilon):
    """Paper-style token-level PPO surrogate for one model decision."""
    import torch

    ratio = torch.exp((new_logprob - old_logprob).clamp(-20.0, 20.0))
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    raw = ratio * advantage
    clipped = clipped_ratio * advantage
    return -torch.minimum(raw, clipped).mean(), ratio


class ModelSampler:
    """Batched, exact-token sampler for stateless Catan decision prompts."""

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_prompt_tokens = max_prompt_tokens
        self.stop_ids = stop_token_ids(model, tokenizer)

    def __call__(self, user_prompts: list[str]) -> list[SampledCompletion]:
        import torch

        prompts = [
            render_prompt(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                self.tokenizer,
            )
            for user_prompt in user_prompts
        ]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        too_long = [length for length in lengths if length > self.max_prompt_tokens]
        if too_long:
            raise RuntimeError(
                f"prompt has {max(too_long)} tokens; limit is "
                f"{self.max_prompt_tokens} and truncation is forbidden"
            )
        encoded = encoded.to(self.model.device)
        prompt_width = encoded["input_ids"].shape[1]
        self.model.eval()
        with torch.inference_mode():
            sequences = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.temperature,
                top_k=0,
                top_p=1.0,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        sampled = []
        for row, length in enumerate(lengths):
            prompt_ids = encoded["input_ids"][row, -int(length) :].tolist()
            completion_ids = sequences[row, prompt_width:].tolist()
            stops = [
                position
                for position, token in enumerate(completion_ids)
                if token in self.stop_ids
            ]
            if stops:
                completion_ids = completion_ids[: stops[0] + 1]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            sampled.append(SampledCompletion(text, prompt_ids, completion_ids))
        return sampled


def _sample_token_logprobs(model, sample: DecisionSample):
    import torch

    ids = sample.prompt_ids + sample.completion_ids
    sequence = torch.tensor([ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(sequence)
    values, mask = completion_token_logprobs(
        model, sequence, attention_mask, len(sample.prompt_ids)
    )
    return values[0][mask[0].bool()]


def record_old_logprobs(model, trainable) -> None:
    """Record the behavior-policy probability before any optimizer step."""
    import torch

    model.eval()
    with torch.no_grad():
        for _, sample in trainable:
            values = _sample_token_logprobs(model, sample)
            if not torch.isfinite(values).all():
                raise RuntimeError("non-finite behavior-policy log-probability")
            sample.old_logprobs = values.tolist()


def update_policy(
    model,
    optimizer,
    trainable,
    games_in_batch: int,
    *,
    policy_epochs: int,
    clip_epsilon: float,
) -> list[dict]:
    """Apply trajectory-factorized PPO updates, averaging over games."""
    import torch

    if games_in_batch <= 0:
        raise ValueError("games_in_batch must be positive")
    if not trainable:
        return []
    if any(sample.old_logprobs is None for _, sample in trainable):
        raise ValueError("all samples need behavior-policy log-probabilities")

    epoch_metrics = []
    model.train()
    for epoch in range(policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        ratios = []
        clipped = 0
        for rollout, sample in trainable:
            new_logprob = _sample_token_logprobs(model, sample)
            old_logprob = torch.tensor(
                sample.old_logprobs,
                dtype=new_logprob.dtype,
                device=new_logprob.device,
            )
            advantage = torch.tensor(
                rollout.advantage,
                dtype=new_logprob.dtype,
                device=new_logprob.device,
            )
            loss, ratio = clipped_surrogate_loss(
                new_logprob, old_logprob, advantage, clip_epsilon
            )
            if not torch.isfinite(loss) or not torch.isfinite(ratio).all():
                raise RuntimeError(
                    f"non-finite policy objective in policy epoch {epoch}"
                )
            (loss / games_in_batch).backward()
            ratio_values = ratio.detach().tolist()
            ratios.extend(ratio_values)
            losses.append(float(loss.detach().item()))
            clipped += sum(abs(value - 1.0) > clip_epsilon for value in ratio_values)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            GRAD_CLIP_NORM,
        )
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"non-finite gradient norm in policy epoch {epoch}")
        optimizer.step()
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "loss_mean": sum(losses) / len(losses),
                "ratio_mean": sum(ratios) / len(ratios),
                "clip_fraction": clipped / len(ratios),
                "grad_norm": float(grad_norm),
            }
        )
    return epoch_metrics


def load_policy_model(model_path: str, adapter_path: Path):
    """Load one trainable policy adapter; there is no fixed KL reference."""
    from peft import PeftModel

    model = load_text_only_qwen35(model_path)
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        adapter_name="policy",
        is_trainable=True,
    )
    model.set_adapter("policy")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".policy." not in name
    ]
    if unexpected:
        raise RuntimeError(f"non-policy parameters are trainable: {unexpected[:4]}")
    return model


def _adapter_file(adapter_path: Path) -> Path:
    direct = adapter_path / "adapter_model.safetensors"
    nested = adapter_path / "policy" / "adapter_model.safetensors"
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"no adapter_model.safetensors under {adapter_path}")


def require_initial_adapter(adapter_path: Path) -> str:
    """Refuse silently starting from anything except the evaluated DPO policy."""
    actual = sha256(_adapter_file(adapter_path))
    if actual != INITIAL_ADAPTER_SHA256:
        raise RuntimeError(
            f"initial adapter hash mismatch: {actual} != {INITIAL_ADAPTER_SHA256}"
        )
    return actual


def run_config(args) -> dict:
    """Return every setting that changes generation or optimization."""
    return {
        "updates": args.updates,
        "groups_per_update": args.groups_per_update,
        "group_size": args.group_size,
        "games_total": args.updates * args.groups_per_update * args.group_size,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": 0,
        "learning_rate": args.lr,
        "lr_scheduler": "constant",
        "warmup_steps": 0,
        "optimizer": "adamw_torch_fused",
        "weight_decay": 0.0,
        "clip_epsilon": args.clip_epsilon,
        "policy_epochs": args.policy_epochs,
        "importance_sampling_level": "token",
        "advantage_normalization": "group_population_std",
        "loss_normalization": "per_decision_token_mean_then_sum_per_game",
        "seed": args.seed,
        "seed_start": args.seed_start,
        "invalid_retries": args.invalid_retries,
        "invalid_fallback": "uniform_legal_action",
        "max_turns": args.max_turns,
        "vp_coef": args.vp_coef,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": args.max_prompt_tokens,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "opponent_policy": OPPONENT_POLICY,
        "fixed_reference_kl": 0.0,
    }


def validate_resume(
    manifest: dict,
    state: dict,
    args,
    initial_adapter_sha256: str,
) -> None:
    """Refuse a resume whose checkpoint or experimental config changed."""
    if manifest.get("status") == "complete":
        raise RuntimeError("refusing to resume a completed run")
    if manifest.get("model") != args.model:
        raise RuntimeError("resume model does not match run manifest")
    if manifest.get("initial_adapter_sha256") != initial_adapter_sha256:
        raise RuntimeError("resume initial adapter does not match run manifest")

    expected_config = run_config(args)
    recorded_config = manifest.get("config")
    if recorded_config != expected_config:
        keys = sorted(
            key
            for key in set(recorded_config or {}) | set(expected_config)
            if (recorded_config or {}).get(key) != expected_config.get(key)
        )
        raise RuntimeError(f"resume config mismatch: {', '.join(keys)}")

    completed = int(state.get("completed_update", -1))
    if state.get("next_update_index") != completed:
        raise RuntimeError("resume checkpoint has inconsistent update indices")
    if manifest.get("last_completed_update") != completed:
        raise RuntimeError("resume checkpoint is not the manifest's latest update")
    updates = manifest.get("updates") or []
    if not updates or updates[-1].get("update") != completed:
        raise RuntimeError("resume manifest has inconsistent update history")
    recorded_checkpoint = Path(updates[-1].get("checkpoint", ""))
    if recorded_checkpoint.resolve() != args.resume_from.resolve():
        raise RuntimeError("resume checkpoint path does not match run manifest")
    if sha256(_adapter_file(args.resume_from)) != state.get("adapter_sha256"):
        raise RuntimeError("resume checkpoint adapter hash mismatch")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint_state(completed_update: int) -> dict:
    """Map a one-based completed update to the next zero-based loop index."""
    return {
        "completed_update": completed_update,
        "next_update_index": completed_update,
        "optimizer_retained": True,
    }


def save_update(model, optimizer, output: Path, update: int) -> Path:
    """Save one complete, resumable update without overwriting older state."""
    import torch

    path = output / "checkpoints" / f"update_{update:03d}"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(temporary), selected_adapters=["policy"])
    adapter_file = _adapter_file(temporary)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    state = checkpoint_state(update)
    state["adapter_sha256"] = sha256(adapter_file)
    _write_json(
        temporary / "state.json",
        state,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(path)
    return path


def retire_optimizer(checkpoint: Path) -> None:
    """Keep the adapter for analysis but retain optimizer state only if current."""
    optimizer = checkpoint / "optimizer.pt"
    if not optimizer.exists():
        return
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text())
    state["optimizer_retained"] = False
    _write_json(state_path, state)
    optimizer.unlink()


def _mean(values) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_update(
    update: int,
    seed_start: int,
    rollouts: list[Rollout],
    group_stats: dict,
    trainable,
    policy_metrics: list[dict],
    phase_seconds: dict[str, float],
) -> dict:
    completed = [rollout for rollout in rollouts if rollout.outcome is not None]
    rewards = [float(rollout.reward) for rollout in completed]
    invalid = sum(rollout.invalid_replies for rollout in rollouts)
    responses = invalid + sum(len(rollout.samples) for rollout in rollouts)
    return {
        "update": update,
        "seed_start": seed_start,
        "seed_end": max(rollout.seed for rollout in rollouts),
        "games_requested": len(rollouts),
        "games_completed": len(completed),
        "groups": group_stats["groups"],
        "degenerate_groups": group_stats["degenerate_groups"],
        "failed_games": group_stats["failed_games"],
        "win_rate": _mean(
            [float(rollout.outcome.winner == rollout.hero) for rollout in completed]
        ),
        "vp_mean": _mean([float(hero_vps(rollout)) for rollout in completed]),
        "vp_margin_mean": _mean([float(vp_margin(rollout)) for rollout in completed]),
        "reward_mean": _mean(rewards),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "truncation_rate": _mean(
            [float(rollout.outcome.truncated) for rollout in completed]
        ),
        "decision_states": sum(rollout.decision_states for rollout in rollouts),
        "valid_decisions": sum(len(rollout.samples) for rollout in rollouts),
        "trainable_decisions": len(trainable),
        "prompt_tokens": sum(
            len(sample.prompt_ids) for rollout in rollouts for sample in rollout.samples
        ),
        "completion_tokens": sum(
            len(sample.completion_ids)
            for rollout in rollouts
            for sample in rollout.samples
        ),
        "max_prompt_tokens": max(
            (
                len(sample.prompt_ids)
                for rollout in rollouts
                for sample in rollout.samples
            ),
            default=0,
        ),
        "invalid_replies": invalid,
        "invalid_rate": invalid / responses if responses else 0.0,
        "policy_epochs": policy_metrics,
        "phase_seconds": phase_seconds,
        "wall_seconds": sum(phase_seconds.values()),
    }


def run(args) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.time()

    import torch
    from transformers import AutoTokenizer, set_seed

    check_fixed_hashseed()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    kernels = assert_fast_linear_attention()
    initial_adapter_sha256 = require_initial_adapter(args.adapter)

    if args.output.exists() and any(args.output.iterdir()) and args.resume_from is None:
        raise RuntimeError(
            f"output directory is not empty: {args.output}; use --resume-from"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    adapter_path = args.adapter
    start_update = 0
    if args.resume_from is not None:
        state = json.loads((args.resume_from / "state.json").read_text())
        if not state.get("optimizer_retained", False):
            raise RuntimeError(
                f"checkpoint has no retained optimizer state: {args.resume_from}"
            )
        adapter_path = args.resume_from / "policy"
        start_update = int(state["next_update_index"])
        orphan = args.output / "checkpoints" / f"update_{start_update + 1:03d}"
        if orphan.exists():
            raise RuntimeError(
                f"checkpoint newer than the manifest exists: {orphan}; "
                "delete it to resume"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_policy_model(args.model, adapter_path).to("cuda")
    base = model.get_base_model()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=0.0, fused=True
    )
    if args.resume_from is not None:
        optimizer.load_state_dict(
            torch.load(
                args.resume_from / "optimizer.pt",
                map_location="cuda",
                weights_only=True,
            )
        )

    sampler = ModelSampler(model, tokenizer, args.temperature, args.max_prompt_tokens)
    manifest_path = args.output / "run_manifest.json"
    if args.resume_from is not None:
        if not manifest_path.exists():
            raise RuntimeError("resume requires the original run_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        validate_resume(manifest, state, args, initial_adapter_sha256)
        manifest["status"] = "running"
        manifest.pop("finished_unix", None)
        manifest.pop("wall_seconds", None)
    else:
        manifest = {
            "status": "running",
            "started_unix": started,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "hostname": platform.node(),
            "model": args.model,
            "initial_adapter": str(args.adapter),
            "initial_adapter_sha256": initial_adapter_sha256,
            "kernels": kernels,
            "packages": package_versions(),
            "config": run_config(args),
            "updates": [],
        }
    _write_json(manifest_path, manifest)

    try:
        for update in range(start_update, args.updates):
            update_started = time.time()
            set_seed(args.seed + update)
            batch_seeds = training_seeds(
                args.seed_start,
                update * args.groups_per_update,
                args.groups_per_update,
            )
            rollouts = make_rollouts(batch_seeds, args.group_size, args.seed)
            rollout_started = time.time()
            rollout_games(
                rollouts,
                sampler,
                invalid_retries=args.invalid_retries,
                max_turns=args.max_turns,
                vp_coef=args.vp_coef,
            )
            rollout_seconds = time.time() - rollout_started
            group_stats = assign_group_advantages(rollouts)
            trainable = trainable_samples(rollouts)
            old_logprob_started = time.time()
            if trainable:
                record_old_logprobs(model, trainable)
                old_logprob_seconds = time.time() - old_logprob_started
                optimize_started = time.time()
                policy_metrics = update_policy(
                    model,
                    optimizer,
                    trainable,
                    games_in_batch=len(rollouts) - group_stats["failed_games"],
                    policy_epochs=args.policy_epochs,
                    clip_epsilon=args.clip_epsilon,
                )
                optimize_seconds = time.time() - optimize_started
            else:
                policy_metrics = []
                old_logprob_seconds = 0.0
                optimize_seconds = 0.0

            checkpoint_started = time.time()
            checkpoint = save_update(model, optimizer, args.output, update + 1)
            checkpoint_seconds = time.time() - checkpoint_started
            phase_seconds = {
                "rollout": rollout_seconds,
                "old_logprobs": old_logprob_seconds,
                "optimize": optimize_seconds,
                "checkpoint": checkpoint_seconds,
            }
            summary = summarize_update(
                update + 1,
                batch_seeds[0],
                rollouts,
                group_stats,
                trainable,
                policy_metrics,
                phase_seconds,
            )
            summary["wall_seconds"] = time.time() - update_started
            summary["checkpoint"] = str(checkpoint)
            summary["optimizer_retained"] = True
            previous_checkpoint = None
            if manifest["updates"]:
                previous = manifest["updates"][-1]
                previous["optimizer_retained"] = False
                previous_checkpoint = Path(previous["checkpoint"])
            manifest["updates"].append(summary)
            manifest["last_completed_update"] = update + 1
            manifest["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
            _write_json(manifest_path, manifest)
            if previous_checkpoint is not None:
                retire_optimizer(previous_checkpoint)
            print(json.dumps(summary, sort_keys=True), flush=True)

        final = args.output / "final"
        model.save_pretrained(str(final), selected_adapters=["policy"])
        tokenizer.save_pretrained(str(final))
        manifest["status"] = "complete"
        manifest["finished_unix"] = time.time()
        manifest["wall_seconds"] = time.time() - started
        manifest["final_adapter_sha256"] = sha256(_adapter_file(final))
        _write_json(manifest_path, manifest)
        return manifest
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_unix"] = time.time()
        manifest["wall_seconds"] = time.time() - started
        _write_json(manifest_path, manifest)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged dagger2 checkpoint")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("data/checkpoints/dpo_fair/dpo_full"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument(
        "--groups-per-update", type=int, default=DEFAULT_GROUPS_PER_UPDATE
    )
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--clip-epsilon", type=float, default=DEFAULT_CLIP_EPSILON)
    parser.add_argument("--policy-epochs", type=int, default=DEFAULT_POLICY_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--invalid-retries", type=int, default=DEFAULT_INVALID_RETRIES)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--vp-coef", type=float, default=DEFAULT_VP_COEF)
    parser.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    args = parser.parse_args()

    positive = {
        "--updates": args.updates,
        "--groups-per-update": args.groups_per_update,
        "--policy-epochs": args.policy_epochs,
        "--max-turns": args.max_turns,
        "--max-prompt-tokens": args.max_prompt_tokens,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.group_size <= 1:
        parser.error("--group-size must be at least 2")
    if args.seed_start < EVAL_SEED_LIMIT:
        parser.error(f"--seed-start must be at least {EVAL_SEED_LIMIT}")
    if args.invalid_retries < 0:
        parser.error("--invalid-retries must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a positive finite number")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be a positive finite number")
    if not 0 < args.clip_epsilon < 1:
        parser.error("--clip-epsilon must be between 0 and 1")
    if not math.isfinite(args.vp_coef) or args.vp_coef < 0:
        parser.error("--vp-coef must be non-negative and finite")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
