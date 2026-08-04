"""Per-decision GRPO on logged Catan decisions with engine-rollout rewards.

For each replayed non-forced decision the policy samples K completions,
each parsed as one stable move id. Every unique legal move is scored once
(mean reward over M paired determinized scenarios, served from the reward
cache with misses computed on the fly) and duplicates share the score.
The update is single-action REINFORCE with a group-mean baseline — no std
normalization, no length normalization — plus a sequence-level KL penalty
to the frozen initial adapter state via peft multi-adapter switching.

Prompts render exactly as in the DPO data: SYSTEM_PROMPT plus the decision
serialization, through the non-thinking Qwen chat template. Sampling runs
in-process by default; --api-base samples from a vLLM endpoint instead, in
which case completions are re-tokenized locally for the loss.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.parse import parse_move
from catan_llm.prompts import SYSTEM_PROMPT
from catan_llm.replay import replay_model_decisions
from catan_llm.schema import GameRecord
from catan_llm.serialize import decision_to_prompt, move_id
from build_reward_cache import (
    HERO_POLICY,
    OPPONENT_POLICY,
    append_rows,
    checked_cache,
    is_val_seed,
    row_key,
    scenario_seeds,
    score_decision_moves,
    scorer_fingerprint,
    write_index,
)
from run_dpo import assert_fast_linear_attention, load_text_only_qwen35, sha256

DEFAULT_K = 8
DEFAULT_SCENARIOS = 8
DEFAULT_TEMPERATURE = 1.0
DEFAULT_BETA_KL = 5e-3
DEFAULT_LR = 2e-6
DEFAULT_SEED = 42
STATES_PER_STEP = 32
MAX_NEW_TOKENS = 32
GRAD_CLIP_NORM = 1.0
DEFAULT_CHECKPOINT_EVERY = 50
THINK_SUFFIX = "<think>\n\n</think>\n\n"
REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "config",
        "packages",
        "states_seen",
        "degenerate_rate",
        "invalid_rate",
        "cache_hit_rate",
        "optimizer_steps",
        "epoch_stats",
        "adapter_sha256",
    }
)


def prompt_messages(game_record, decision) -> list[dict]:
    """The exact chat messages used to build the DPO dataset."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": decision_to_prompt(game_record, decision)},
    ]


def render_prompt(messages: list[dict], tokenizer) -> str:
    """Render one non-thinking Qwen generation prompt."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not prompt.endswith(THINK_SUFFIX):
        raise ValueError("unexpected Qwen non-thinking generation prompt suffix")
    return prompt


def iter_states(paths, seed: int, max_states: int | None = None) -> list[tuple]:
    """Return (path, game_record, decision) tuples in one global shuffle.

    Decisions from every game are pooled before shuffling, so an optimizer
    step is not dominated by a single game's decisions. Trajectories with
    eval-range seeds are refused and DPO-validation games are skipped,
    exactly as in run_grpo_trl's build_rows.
    """
    states = []
    for path in sorted(paths):
        with path.open() as handle:
            header = GameRecord.model_validate_json(handle.readline())
        if header.seed is None or header.seed < EVAL_SEED_LIMIT:
            raise ValueError(
                f"{path.name}: training state seed in eval range: {header.seed}"
            )
        if is_val_seed(header.seed):
            continue
        for replayed in replay_model_decisions(path):
            states.append((path, replayed.game_record, replayed.decision))
    random.Random(seed).shuffle(states)
    return states[:max_states] if max_states is not None else states


def degenerate(parses: list) -> bool:
    """All K samples parse identically (one action, or all invalid)."""
    return len(set(parses)) == 1


def group_rewards(parses: list, action_scores: dict) -> list[float]:
    """Per-sample rewards: duplicates share scores, invalid parses get 0."""
    return [action_scores[parsed] if parsed is not None else 0.0 for parsed in parses]


def advantages(rewards: list[float]) -> list[float]:
    """Group-mean baseline; no std normalization."""
    baseline = sum(rewards) / len(rewards)
    return [reward - baseline for reward in rewards]


def grpo_loss(advantage, policy_lp, ref_lp, beta_kl: float):
    """REINFORCE with a k3 KL penalty to the frozen reference.

    Returns the scalar loss and the per-sample KL estimate.
    """
    import torch

    log_ratio = ref_lp - policy_lp
    kl = torch.exp(log_ratio) - log_ratio - 1
    return (-(advantage * policy_lp) + beta_kl * kl).mean(), kl


def state_action_scores(
    cache: dict,
    trajectory: Path,
    game_id: str,
    decision_index: int,
    move_ids: list[str],
    seeds: list[int],
    executor,
    hero_policy: str = HERO_POLICY,
    opponent_policy: str = OPPONENT_POLICY,
) -> tuple[dict[str, float], list[dict], int, int]:
    """Mean reward per move from the cache; misses are scored and cached.

    Misses replay the logged decision from its trajectory file, so the
    executor and inline paths score identically.

    Returns (scores, newly computed rows, cache hits, cache misses).
    """
    missing = {
        move: [
            seed
            for seed in seeds
            if (game_id, decision_index, move, seed) not in cache
        ]
        for move in move_ids
    }
    misses = sum(len(wanted) for wanted in missing.values())
    hits = len(move_ids) * len(seeds) - misses

    new_rows = []
    futures = {}
    for move, wanted in missing.items():
        if not wanted:
            continue
        if executor is None:
            new_rows.extend(
                score_decision_moves(
                    str(trajectory),
                    decision_index,
                    [move],
                    wanted,
                    hero_policy,
                    opponent_policy,
                )
            )
        else:
            futures[move] = executor.submit(
                score_decision_moves,
                str(trajectory),
                decision_index,
                [move],
                wanted,
                hero_policy,
                opponent_policy,
            )
    for move in sorted(futures):
        new_rows.extend(futures[move].result())
    for row in new_rows:
        cache[row_key(row)] = row

    scores = {
        move: sum(
            cache[(game_id, decision_index, move, seed)]["reward"] for seed in seeds
        )
        / len(seeds)
        for move in move_ids
    }
    return scores, new_rows, hits, misses


def completion_token_logprobs(model, sequences, attention_mask, prompt_length: int):
    """Completion-token log-probs and mask for right-padded sequences."""
    import torch

    logits = model(input_ids=sequences, attention_mask=attention_mask).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    picked = logprobs.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].clone()
    mask[:, : prompt_length - 1] = 0
    return picked, mask


def sequence_logprobs(model, sequences, attention_mask, prompt_length: int):
    """Sum completion-token log-probs per sequence without length normalization."""
    picked, mask = completion_token_logprobs(
        model, sequences, attention_mask, prompt_length
    )
    return (picked * mask).sum(dim=-1)


def ref_sequence_logprobs(model, sequences, attention_mask, prompt_length: int):
    """Reference log-probs under the frozen "ref" adapter, restoring "policy"."""
    import torch

    model.set_adapter("ref")
    try:
        with torch.no_grad():
            return sequence_logprobs(model, sequences, attention_mask, prompt_length)
    finally:
        model.set_adapter("policy")


def stop_token_ids(model, tokenizer) -> set[int]:
    """Every id generate() stops on, plus the pad id used for batching."""
    configured = model.generation_config.eos_token_id
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if isinstance(configured, int):
        ids.add(configured)
    elif configured is not None:
        ids.update(configured)
    return {token for token in ids if token is not None}


def completion_attention_mask(sequences, prompt_length: int, stop_ids: set[int]):
    """Mask padding after each completion's first stop token, which counts.

    Qwen checkpoints stop generation on more than one id, so masking only
    the tokenizer's EOS would leak padding into the loss for completions
    that ended on a secondary stop token.
    """
    import torch

    mask = torch.ones_like(sequences)
    completions = sequences[:, prompt_length:]
    stops = torch.tensor(sorted(stop_ids), device=sequences.device)
    for index in range(completions.shape[0]):
        stop_positions = torch.isin(completions[index], stops).nonzero()
        if len(stop_positions):
            end = prompt_length + int(stop_positions[0].item()) + 1
            mask[index, end:] = 0
    return mask


def encode_completions(texts: list[str], tokenizer):
    """Tokenize API-sampled texts as training targets, appending EOS."""
    import torch

    rows = [
        tokenizer(text, add_special_tokens=False)["input_ids"]
        + [tokenizer.eos_token_id]
        for text in texts
    ]
    width = max(len(row) for row in rows)
    padded = torch.full((len(rows), width), tokenizer.eos_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = 1
    return padded, mask


def sample_in_process(model, tokenizer, prompt: str, k: int, temperature: float):
    """Sample K completions with plain temperature sampling (no top-k/top-p)."""
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            num_return_sequences=k,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    model.train()
    prompt_length = encoded["input_ids"].shape[1]
    texts = tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)
    attention_mask = completion_attention_mask(
        output, prompt_length, stop_token_ids(model, tokenizer)
    )
    return output, attention_mask, prompt_length, texts


def sample_via_api(
    api_base: str, model_name: str, prompt: str, k: int, temperature: float
) -> list[str]:
    """Sample K completion texts from an OpenAI-compatible vLLM endpoint."""
    request = Request(
        api_base.rstrip("/") + "/completions",
        data=json.dumps(
            {
                "model": model_name,
                "prompt": prompt,
                "n": k,
                "temperature": temperature,
                "max_tokens": MAX_NEW_TOKENS,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        payload = json.load(response)
    texts = [choice["text"] for choice in payload["choices"]]
    if len(texts) != k:
        raise RuntimeError(f"endpoint returned {len(texts)} completions, not {k}")
    return texts


def subset_identity(identity: list[list]) -> str:
    payload = json.dumps(identity, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _assert_finite(node, key: str = "manifest") -> None:
    if isinstance(node, dict):
        for name, value in node.items():
            _assert_finite(value, f"{key}.{name}")
    elif isinstance(node, (list, tuple)):
        for position, value in enumerate(node):
            _assert_finite(value, f"{key}[{position}]")
    elif isinstance(node, float) and not math.isfinite(node):
        raise ValueError(f"non-finite manifest value: {key}")


def write_manifest(path: Path, manifest: dict) -> None:
    """Write run_manifest.json, refusing missing fields or non-finite values."""
    missing = sorted(REQUIRED_MANIFEST_KEYS - set(manifest))
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    _assert_finite(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def package_versions() -> dict[str, str]:
    import peft
    import safetensors
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "safetensors": safetensors.__version__,
    }


def load_policy_model(model_path: str, adapter_path: Path):
    """Load the merged base with a trainable "policy" and frozen "ref" adapter."""
    from peft import PeftModel

    model = load_text_only_qwen35(model_path)
    model = PeftModel.from_pretrained(
        model, str(adapter_path), adapter_name="policy", is_trainable=True
    )
    model.load_adapter(str(adapter_path), adapter_name="ref")
    model.set_adapter("policy")
    frozen_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".policy." not in name
    ]
    if frozen_trainable:
        raise RuntimeError(
            f"non-policy parameters are trainable: {frozen_trainable[:4]}"
        )
    return model


def save_checkpoint(model, output: Path, steps: int) -> Path:
    """Save the policy adapter to output/step_NNNNN and verify it exists."""
    path = output / f"step_{steps:05d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), selected_adapters=["policy"])
    adapter_file = path / "policy" / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise RuntimeError(f"checkpoint {path} contains no adapter tensors")
    return path


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run(args) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.time()

    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, set_seed

    check_fixed_hashseed()
    set_seed(args.seed)
    kernels = assert_fast_linear_attention()

    paths = sorted(Path(args.states).glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no .jsonl trajectories found in {args.states}")

    cache_config = {
        "trajectories": str(args.states),
        "scenarios": args.scenarios,
        "seed": args.seed,
        "hero_policy": HERO_POLICY,
        "opponent_policy": OPPONENT_POLICY,
        **scorer_fingerprint(),
    }
    if args.cache.exists():
        cache, cache_index = checked_cache(args.cache)
        for key in (
            "seed",
            "hero_policy",
            "opponent_policy",
            "scorer_sha256",
            "catanatron_version",
        ):
            if cache_index.get(key) != cache_config[key]:
                raise RuntimeError(
                    f"cache index {key} mismatch: "
                    f"{cache_index.get(key)!r} != {cache_config[key]!r}"
                )
        cache_config = {
            key: value
            for key, value in cache_index.items()
            if key not in ("cache_sha256", "rows", "games", "decisions")
        }
    else:
        cache = {}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy_model(args.model, args.adapter)
    model = model.to("cuda")
    base = model.get_base_model()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    executor = (
        ProcessPoolExecutor(max_workers=args.reward_workers)
        if args.reward_workers > 1
        else None
    )
    states_seen = 0
    degenerate_groups = 0
    samples_total = 0
    invalid_samples = 0
    cache_hits = 0
    cache_misses = 0
    steps = 0
    pending = 0
    reward_values = []
    group_means = []
    kl_values = []
    loss_values = []
    unique_counts = []
    identity = []

    def optimizer_step():
        nonlocal steps, pending
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
        pending = 0
        if steps % args.checkpoint_every == 0:
            save_checkpoint(model, args.output, steps)

    optimizer.zero_grad()
    try:
        for trajectory, game_record, decision in iter_states(
            paths, args.seed, args.max_states
        ):
            game_id = game_record.game_id
            prompt = render_prompt(
                prompt_messages(game_record, decision), tokenizer
            )
            if args.api_base:
                texts = sample_via_api(
                    args.api_base,
                    args.api_model,
                    prompt,
                    args.k,
                    args.temperature,
                )
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                prompt_length = prompt_ids.shape[1]
                completion_ids, completion_mask = encode_completions(texts, tokenizer)
                sequences = torch.cat(
                    [prompt_ids.repeat(args.k, 1), completion_ids], dim=1
                ).to(model.device)
                attention_mask = torch.cat(
                    [
                        torch.ones((args.k, prompt_length), dtype=torch.long),
                        completion_mask,
                    ],
                    dim=1,
                ).to(model.device)
            else:
                sequences, attention_mask, prompt_length, texts = sample_in_process(
                    model, tokenizer, prompt, args.k, args.temperature
                )

            parses = [parse_move(text, decision.legal_actions) for text in texts]
            states_seen += 1
            samples_total += args.k
            invalid_samples += parses.count(None)
            if degenerate(parses):
                degenerate_groups += 1
                continue

            unique = sorted({parsed for parsed in parses if parsed is not None})
            moves = {index: move_id(*decision.legal_actions[index]) for index in unique}
            seeds = scenario_seeds(args.seed, game_id, decision.i, args.scenarios)
            scores, new_rows, hits, misses = state_action_scores(
                cache,
                trajectory,
                game_id,
                decision.i,
                list(moves.values()),
                seeds,
                executor,
            )
            cache_hits += hits
            cache_misses += misses
            if new_rows:
                append_rows(args.cache, new_rows)

            rewards = group_rewards(
                parses, {index: scores[move] for index, move in moves.items()}
            )
            advantage = torch.tensor(
                advantages(rewards), dtype=torch.float32, device=model.device
            )
            policy_lp = sequence_logprobs(
                model, sequences, attention_mask, prompt_length
            )
            ref_lp = ref_sequence_logprobs(
                model, sequences, attention_mask, prompt_length
            )
            loss, kl = grpo_loss(advantage, policy_lp, ref_lp, args.beta_kl)
            if not math.isfinite(loss.item()):
                raise RuntimeError(
                    f"non-finite loss at {game_id} decision {decision.i}"
                )
            (loss / STATES_PER_STEP).backward()

            pending += 1
            identity.append([game_id, decision.i])
            reward_values.extend(rewards)
            group_means.append(sum(rewards) / len(rewards))
            kl_values.append(kl.mean().item())
            loss_values.append(loss.item())
            unique_counts.append(len(unique))
            if pending == STATES_PER_STEP:
                optimizer_step()
                if args.max_steps is not None and steps >= args.max_steps:
                    break
        if pending and not (args.max_steps is not None and steps >= args.max_steps):
            optimizer_step()
    finally:
        if executor is not None:
            executor.shutdown()
        if args.cache.exists():
            write_index(args.cache, cache_config)

    if not identity:
        raise RuntimeError("no trainable groups; every state was degenerate")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output), selected_adapters=["policy"])
    tokenizer.save_pretrained(str(args.output))
    adapter_file = args.output / "policy" / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise RuntimeError("saved adapter contains no tensors")
    with safe_open(adapter_file, framework="pt") as adapter:
        adapter_keys = list(adapter.keys())
    if not adapter_keys:
        raise RuntimeError("saved adapter contains no tensors")

    manifest = {
        "started_unix": started,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "hostname": platform.node(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "model": args.model,
        "adapter": str(args.adapter),
        "adapter_source_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "states": str(args.states),
        "cache": str(args.cache),
        "cache_sha256": sha256(args.cache),
        "config": {
            "k": args.k,
            "scenarios": args.scenarios,
            "temperature": args.temperature,
            "beta_kl": args.beta_kl,
            "lr": args.lr,
            "seed": args.seed,
            "states_per_step": STATES_PER_STEP,
            "max_new_tokens": MAX_NEW_TOKENS,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "max_states": args.max_states,
            "max_steps": args.max_steps,
            "checkpoint_every": args.checkpoint_every,
            "api_base": args.api_base,
            "hero_policy": HERO_POLICY,
            "opponent_policy": OPPONENT_POLICY,
        },
        "kernels": kernels,
        "packages": package_versions(),
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "states_seen": states_seen,
        "groups_trained": len(identity),
        "degenerate_groups": degenerate_groups,
        "degenerate_rate": degenerate_groups / states_seen,
        "samples_total": samples_total,
        "invalid_samples": invalid_samples,
        "invalid_rate": invalid_samples / samples_total,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": (
            cache_hits / (cache_hits + cache_misses)
            if cache_hits + cache_misses
            else None
        ),
        "optimizer_steps": steps,
        "epoch_stats": [
            {
                "groups": len(identity),
                "reward_mean": _mean(reward_values),
                "reward_min": min(reward_values),
                "reward_max": max(reward_values),
                "group_reward_mean": _mean(group_means),
                "kl_mean": _mean(kl_values),
                "loss_mean": _mean(loss_values),
                "mean_unique_actions": _mean([float(count) for count in unique_counts]),
            }
        ],
        "state_subset_sha256": subset_identity(identity),
        "adapter_tensors": len(adapter_keys),
        "adapter_sha256": sha256(adapter_file),
    }
    write_manifest(args.output / "run_manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged base checkpoint")
    parser.add_argument(
        "--states", type=Path, default=Path("data/dagger_traces/r3pairs")
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("data/checkpoints/dpo_fair/dpo_full"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--beta-kl", type=float, default=DEFAULT_BETA_KL)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY
    )
    parser.add_argument(
        "--api-base",
        help="OpenAI-compatible /v1 base URL; sample there instead of in-process",
    )
    parser.add_argument("--api-model", default="dpo-post")
    args = parser.parse_args()

    if args.k <= 1:
        parser.error("--k must be at least 2")
    if args.scenarios <= 0:
        parser.error("--scenarios must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a positive finite number")
    if args.beta_kl < 0:
        parser.error("--beta-kl must be non-negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.reward_workers <= 0:
        parser.error("--reward-workers must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
