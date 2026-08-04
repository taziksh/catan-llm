"""Per-decision GRPO with TRL's GRPOTrainer, rewarded from the engine cache.

Rewards are cache lookups (a miss raises; rebuild the cache), prompts render
exactly as in the DPO data, and ``--no-adapter`` trains from the dagger2
base instead of merging the DPO adapter first.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.parse import MOVE_RE
from catan_llm.replay import replay_model_decisions
from catan_llm.schema import GameRecord
from catan_llm.serialize import move_id
from build_reward_cache import (
    HERO_POLICY,
    OPPONENT_POLICY,
    VAL_EVERY,
    checked_cache,
    is_val_seed,
    scenario_seeds,
    scorer_fingerprint,
)
from run_dpo import (
    assert_fast_linear_attention,
    assert_finite_metrics,
    load_text_only_qwen35,
    package_versions,
    sha256,
)
from run_grpo import prompt_messages, render_prompt

DEFAULT_K = 8
DEFAULT_SCENARIOS = 8
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LR = 2e-6
DEFAULT_SEED = 42
DEFAULT_ADAPTER = Path("data/checkpoints/dpo_fair/dpo_full")
DEFAULT_RUN_BASE = "grpo-r3pairs-fair"
PROMPTS_PER_STEP = 32
COMPLETIONS_PER_DEVICE = 8
MAX_COMPLETION_TOKENS = 32
WANDB_PROJECT = "catan-llm"
REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_SOURCES = (
    "catan_llm/replay.py",
    "catan_llm/determinize.py",
    "catan_llm/simulation.py",
    "scripts/build_reward_cache.py",
    "scripts/run_grpo_trl.py",
)


def build_rows(
    paths, seed: int, max_states: int | None = None
) -> list[dict]:
    """One conversational GRPO row per replayed non-forced decision.

    Shuffled deterministically given seed, so ``max_states`` truncation
    yields an unbiased subset.
    """
    rows = []
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
            decision = replayed.decision
            rows.append(
                {
                    "prompt": prompt_messages(replayed.game_record, decision),
                    "game_id": replayed.game_record.game_id,
                    "decision": decision.i,
                    "legal_moves": [
                        move_id(*action) for action in decision.legal_actions
                    ],
                }
            )
    random.Random(seed).shuffle(rows)
    if max_states is not None:
        rows = rows[:max_states]
    return rows


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if (
        len(completion) == 1
        and completion[0].get("role") == "assistant"
        and isinstance(completion[0].get("content"), str)
    ):
        return completion[0]["content"]
    raise ValueError(f"completion must be one assistant message: {completion!r}")


def parse_completion(text: str, legal_moves: list[str]) -> str | None:
    """parse_move over move-id strings: last legal ``answer:`` token wins."""
    legal = set(legal_moves)
    for token in reversed(MOVE_RE.findall(text)):
        token = token.strip("*`.,;:\"'")
        if token in legal:
            return token
    return None


def cached_value(
    cache: dict, run_seed: int, scenarios: int, game: str, index: int, move: str
) -> float:
    total = 0.0
    for scenario in scenario_seeds(run_seed, game, index, scenarios):
        row = cache.get((game, index, move, scenario))
        if row is None:
            raise RuntimeError(
                f"reward cache missing {game} decision {index} "
                f"move {move} scenario {scenario}"
            )
        total += row["reward"]
    return total / scenarios


def _parses(completions, legal_moves):
    return [
        parse_completion(completion_text(completion), legal)
        for completion, legal in zip(completions, legal_moves, strict=True)
    ]


class TrainerMetrics:
    """Appends scalars to the trainer's metric buffer for the next wandb log."""

    def __init__(self):
        self.trainer = None

    def record(self, name: str, value: float) -> None:
        if self.trainer is not None:
            self.trainer._metrics["train"][name].append(value)


def record_diagnostics(
    metrics: TrainerMetrics,
    cache: dict,
    run_seed: int,
    scenarios: int,
    parses: list,
    game_id,
    decision,
    legal_moves,
) -> None:
    """Per-batch reward ceiling, invalid rate, and unique moves per state."""
    ceilings = [
        max(
            cached_value(cache, run_seed, scenarios, game, index, move)
            for move in legal
        )
        for game, index, legal in zip(
            game_id, decision, legal_moves, strict=True
        )
    ]
    groups = {}
    for move, game, index in zip(parses, game_id, decision, strict=True):
        groups.setdefault((game, index), set())
        if move is not None:
            groups[(game, index)].add(move)
    metrics.record("diagnostics/reward_ceiling", sum(ceilings) / len(ceilings))
    metrics.record("diagnostics/invalid_rate", parses.count(None) / len(parses))
    metrics.record(
        "diagnostics/unique_moves",
        sum(len(moves) for moves in groups.values()) / len(groups),
    )


def make_reward(
    cache: dict,
    run_seed: int,
    scenarios: int,
    metrics: TrainerMetrics | None = None,
):
    def engine_reward(completions, game_id, decision, legal_moves, **kwargs):
        parses = _parses(completions, legal_moves)
        rewards = [
            0.0
            if move is None
            else cached_value(cache, run_seed, scenarios, game, index, move)
            for move, game, index in zip(
                parses, game_id, decision, strict=True
            )
        ]
        if metrics is not None:
            record_diagnostics(
                metrics, cache, run_seed, scenarios,
                parses, game_id, decision, legal_moves,
            )
        return rewards

    return engine_reward


def run_label(base: str, adapter: Path | None) -> str:
    return f"{base}-{'dpo' if adapter is not None else 'dagger2'}"


def init_fields(adapter: Path | None) -> dict:
    if adapter is None:
        return {"init": "dagger2", "adapter": None, "adapter_source_sha256": None}
    return {
        "init": "dpo",
        "adapter": str(adapter),
        "adapter_source_sha256": sha256(adapter / "adapter_model.safetensors"),
    }


def prepare_policy_base(model_path: str, adapter: Path | None):
    model = load_text_only_qwen35(model_path)
    if adapter is None:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()


def grpo_config_kwargs(args) -> dict:
    kwargs = {
        "output_dir": str(args.output),
        "run_name": run_label(args.run_base, args.adapter),
        "seed": args.seed,
        "data_seed": args.seed,
        "num_train_epochs": 1.0,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": COMPLETIONS_PER_DEVICE,
        "gradient_accumulation_steps": (
            PROMPTS_PER_STEP * args.k // COMPLETIONS_PER_DEVICE
        ),
        "num_generations": args.k,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_completion_length": MAX_COMPLETION_TOKENS,
        "loss_type": "dr_grpo",
        "scale_rewards": "none",
        "beta": 0.0,
        "num_iterations": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "tf32": True,
        "logging_steps": 1,
        "log_completions": True,
        "num_completions_to_print": 2,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "report_to": ["wandb"],
        "remove_unused_columns": False,
        "optim": "adamw_torch_fused",
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
    }
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    if args.use_vllm:
        kwargs["use_vllm"] = True
        kwargs["vllm_mode"] = "colocate"
        kwargs["vllm_gpu_memory_utilization"] = 0.25
    return kwargs


def validate_grpo_api(kwargs: dict) -> None:
    from trl import GRPOConfig

    parameters = inspect.signature(GRPOConfig).parameters
    unsupported = sorted(set(kwargs) - set(parameters))
    if unsupported:
        raise RuntimeError(
            f"installed TRL GRPOConfig lacks expected fields: {unsupported}"
        )


def check_cache_index(index: dict, args) -> None:
    """Raise unless the cache was built with this run's exact config."""
    expected = {
        "seed": args.seed,
        "trajectories": str(args.states),
        "val_every": VAL_EVERY,
        "hero_policy": HERO_POLICY,
        "opponent_policy": OPPONENT_POLICY,
        **scorer_fingerprint(),
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise RuntimeError(
                f"cache index {key} mismatch: {index.get(key)!r} != {value!r}"
            )
    if index["scenarios"] < args.scenarios:
        raise RuntimeError(
            f"cache holds {index['scenarios']} scenarios, run needs "
            f"{args.scenarios}"
        )


def preflight_cache_coverage(
    cache: dict, rows: list[dict], run_seed: int, scenarios: int
) -> int:
    """Return the number of cache keys the run needs, raising on any gap."""
    required = 0
    missing = 0
    for row in rows:
        seeds = scenario_seeds(
            run_seed, row["game_id"], row["decision"], scenarios
        )
        for move in row["legal_moves"]:
            for scenario in seeds:
                required += 1
                if (row["game_id"], row["decision"], move, scenario) not in cache:
                    missing += 1
    if missing:
        raise RuntimeError(
            f"reward cache missing {missing} of {required} required keys"
        )
    return required


def subset_identity(rows: list[dict]) -> str:
    identity = [[row["game_id"], row["decision"]] for row in rows]
    payload = json.dumps(identity, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run(args) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    started = time.time()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from safetensors import safe_open
    from transformers import AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    check_fixed_hashseed()
    set_seed(args.seed)
    kernels = assert_fast_linear_attention()

    paths = sorted(Path(args.states).glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no .jsonl trajectories found in {args.states}")
    rows = build_rows(paths, args.seed, args.max_states)
    if not rows:
        raise ValueError("no training rows after validation-game exclusion")

    cache, cache_index = checked_cache(args.cache)
    check_cache_index(cache_index, args)
    cache_keys_required = preflight_cache_coverage(
        cache, rows, args.seed, args.scenarios
    )

    config_kwargs = grpo_config_kwargs(args)
    validate_grpo_api(config_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    render_prompt(rows[0]["prompt"], tokenizer)

    model = prepare_policy_base(args.model, args.adapter)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    metrics = TrainerMetrics()
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**config_kwargs),
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        reward_funcs=make_reward(cache, args.seed, args.scenarios, metrics),
        peft_config=peft_config,
    )
    metrics.trainer = trainer

    train_result = trainer.train()
    assert_finite_metrics(train_result.metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    adapter_path = args.output / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt") as adapter:
        adapter_keys = list(adapter.keys())
    if not adapter_keys:
        raise RuntimeError("saved adapter contains no tensors")

    import wandb

    manifest = {
        "started_unix": started,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "hostname": platform.node(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "source_sha256": {
            name: sha256(REPO_ROOT / name) for name in PROVENANCE_SOURCES
        },
        "model": args.model,
        "base_model_config_sha256": sha256(Path(args.model) / "config.json"),
        **init_fields(args.adapter),
        "states": str(args.states),
        "cache": str(args.cache),
        "cache_sha256": sha256(args.cache),
        "cache_index": cache_index,
        "cache_keys_required": cache_keys_required,
        "config": config_kwargs,
        "k": args.k,
        "scenarios": args.scenarios,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "rows": len(rows),
        "subset_sha256": subset_identity(rows),
        "kernels": kernels,
        "packages": package_versions(),
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "train": {
            key: value
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float))
        },
        "wandb_run": wandb.run.name if wandb.run else None,
        "wandb_url": wandb.run.url if wandb.run else None,
        "adapter_tensors": len(adapter_keys),
        "adapter_sha256": sha256(adapter_path),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged base checkpoint")
    parser.add_argument(
        "--adapter", type=Path, default=DEFAULT_ADAPTER,
        help="DPO adapter merged into the base before training",
    )
    parser.add_argument(
        "--no-adapter", action="store_true",
        help="train directly on the base (GRPO-from-dagger2 ablation)",
    )
    parser.add_argument(
        "--states", type=Path, default=Path("data/dagger_traces/r3pairs")
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-base", default=DEFAULT_RUN_BASE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--use-vllm", action="store_true")
    args = parser.parse_args()

    if args.no_adapter:
        args.adapter = None
    if args.k <= 1:
        parser.error("--k must be at least 2")
    if PROMPTS_PER_STEP * args.k % COMPLETIONS_PER_DEVICE:
        parser.error(
            f"--k must keep {PROMPTS_PER_STEP} prompts per step divisible "
            f"into micro-batches of {COMPLETIONS_PER_DEVICE}"
        )
    if args.scenarios <= 0:
        parser.error("--scenarios must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a positive finite number")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.max_states is not None and args.max_states <= 0:
        parser.error("--max-states must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
