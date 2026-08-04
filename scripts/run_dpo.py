"""Run a controlled LoRA DPO update on a merged Qwen3.5 checkpoint.

The official Qwen3.5 checkpoint contains a vision tower even for text-only
generation. This launcher explicitly loads Qwen3_5ForCausalLM and maps the
merged checkpoint's ``model.language_model.*`` tensors onto the text-only
``model.*`` backbone. That mirrors vLLM's ``--language-model-only`` behavior
and prevents the DPO adapter from touching unused vision parameters.
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


DEFAULT_BETA = 0.1
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_SEED = 42
TEXT_KEY_MAPPING = {r"^model\.language_model\.": "model."}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deterministic_subset(
    rows: list[dict], limit: int | None, seed: int
) -> list[dict]:
    if limit is None or limit >= len(rows):
        return list(rows)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[:limit]]


def subset_identity(rows: list[dict]) -> str:
    identity = [
        [row["game_id"], row["decision"]]
        for row in rows
    ]
    payload = json.dumps(identity, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _assistant_text(messages: list[dict]) -> str:
    if (
        len(messages) != 1
        or messages[0].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise ValueError("completion must be one assistant message")
    return messages[0]["content"]


def render_pair(row: dict, tokenizer) -> dict[str, str]:
    """Render exactly one non-thinking Qwen chat prompt for TRL."""
    prompt = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    chosen = _assistant_text(row["chosen"])
    rejected = _assistant_text(row["rejected"])
    if chosen == rejected:
        raise ValueError(
            f"identical preference completions for {row.get('game_id')}"
        )
    if not prompt.endswith("<think>\n\n</think>\n\n"):
        raise ValueError(
            "unexpected Qwen non-thinking generation prompt suffix"
        )
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def token_length_stats(rendered: list[dict], tokenizer) -> dict:
    prompt_lengths = []
    completion_lengths = []
    eos_extra = 1 if tokenizer.eos_token_id is not None else 0
    for row in rendered:
        prompt_lengths.append(
            len(
                tokenizer(
                    row["prompt"], add_special_tokens=False
                )["input_ids"]
            )
        )
        completion_lengths.extend(
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            + eos_extra
            for text in (row["chosen"], row["rejected"])
        )

    def summarize(values: list[int]) -> dict:
        ordered = sorted(values)

        def percentile(fraction: float) -> int:
            index = math.ceil(fraction * len(ordered)) - 1
            return ordered[max(index, 0)]

        return {
            "min": ordered[0],
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": ordered[-1],
        }

    return {
        "prompt": summarize(prompt_lengths),
        "completion": summarize(completion_lengths),
    }


def assert_fast_linear_attention() -> dict[str, bool]:
    """Refuse paid training through the slow Qwen Gated DeltaNet fallback."""
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    availability = {
        "causal_conv1d": is_causal_conv1d_available(),
        "flash_linear_attention": is_flash_linear_attention_available(),
    }
    if not availability["flash_linear_attention"]:
        raise RuntimeError(
            "flash-linear-attention is required; Qwen3.5's PyTorch "
            "Gated DeltaNet fallback is too slow and memory hungry for this run"
        )
    return availability


def load_text_only_qwen35(model_path: str):
    """Load a full merged Qwen3.5 checkpoint as its text-only causal LM."""
    import torch
    from transformers import (
        AutoConfig,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )

    full_config = AutoConfig.from_pretrained(
        model_path, local_files_only=True
    )
    if not hasattr(full_config, "text_config"):
        raise ValueError("checkpoint does not contain Qwen3.5 text_config")
    text_config = Qwen3_5TextConfig.from_dict(
        full_config.text_config.to_dict()
    )
    text_config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        config=text_config,
        key_mapping=TEXT_KEY_MAPPING,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if model.config.model_type != "qwen3_5_text":
        raise AssertionError(
            f"loaded non-text config: {model.config.model_type}"
        )
    return model


def package_versions() -> dict[str, str]:
    import accelerate
    import datasets
    import peft
    import safetensors
    import torch
    import transformers
    import trl

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "safetensors": safetensors.__version__,
    }


def dpo_config_kwargs(args) -> dict:
    kwargs = {
        "output_dir": str(args.output),
        "seed": args.seed,
        "data_seed": args.seed,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "tf32": True,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "precompute_ref_log_probs": True,
        "logging_steps": 1,
        "eval_strategy": "no",
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": True,
        "optim": "adamw_torch_fused",
        "warmup_ratio": 0.0,
        "lr_scheduler_type": "constant",
    }
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    return kwargs


def validate_dpo_api(kwargs: dict) -> None:
    from trl import DPOConfig

    parameters = inspect.signature(DPOConfig).parameters
    unsupported = sorted(set(kwargs) - set(parameters))
    if unsupported:
        raise RuntimeError(
            f"installed TRL DPOConfig lacks expected fields: {unsupported}"
        )


def metric_snapshot(metrics: dict) -> dict:
    wanted = (
        "loss",
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
        "logits/chosen",
        "logits/rejected",
        "runtime",
        "samples_per_second",
    )
    return {
        key: value
        for key, value in metrics.items()
        if any(key.endswith(name) for name in wanted)
    }


def assert_finite_metrics(metrics: dict) -> None:
    non_finite = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (float, int)) and not math.isfinite(value)
    }
    if non_finite:
        raise RuntimeError(f"non-finite metrics: {non_finite}")


def run(args) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.time()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from safetensors import safe_open
    from transformers import AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    kernels = assert_fast_linear_attention()
    train_rows = deterministic_subset(
        load_jsonl(args.train), args.train_samples, args.seed
    )
    eval_rows = deterministic_subset(
        load_jsonl(args.eval), args.eval_samples, args.seed + 1
    )
    if not train_rows or not eval_rows:
        raise ValueError("train and eval datasets must be non-empty")
    if {row["game_id"] for row in train_rows} & {
        row["game_id"] for row in eval_rows
    }:
        raise ValueError("game leakage between train and eval subsets")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered_train = [render_pair(row, tokenizer) for row in train_rows]
    rendered_eval = [render_pair(row, tokenizer) for row in eval_rows]
    lengths = {
        "train": token_length_stats(rendered_train, tokenizer),
        "eval": token_length_stats(rendered_eval, tokenizer),
    }
    if max(
        lengths["train"]["prompt"]["max"],
        lengths["eval"]["prompt"]["max"],
    ) > args.max_prompt_length:
        raise ValueError(f"prompt exceeds limit: {lengths}")
    if max(
        lengths["train"]["completion"]["max"],
        lengths["eval"]["completion"]["max"],
    ) > args.max_completion_length:
        raise ValueError(f"completion exceeds limit: {lengths}")

    config_kwargs = dpo_config_kwargs(args)
    validate_dpo_api(config_kwargs)
    model = load_text_only_qwen35(args.model)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(**config_kwargs),
        train_dataset=Dataset.from_list(rendered_train),
        eval_dataset=Dataset.from_list(rendered_eval),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    before = trainer.evaluate(metric_key_prefix="before")
    assert_finite_metrics(before)
    train_result = trainer.train()
    assert_finite_metrics(train_result.metrics)
    after = trainer.evaluate(metric_key_prefix="after")
    assert_finite_metrics(after)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    adapter_path = args.output / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt") as adapter:
        adapter_keys = list(adapter.keys())
    if not adapter_keys:
        raise RuntimeError("saved adapter contains no tensors")

    manifest = {
        "schema_version": 1,
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
        "train_path": str(args.train),
        "eval_path": str(args.eval),
        "train_sha256": sha256(args.train),
        "eval_sha256": sha256(args.eval),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_subset_sha256": subset_identity(train_rows),
        "eval_subset_sha256": subset_identity(eval_rows),
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.gradient_accumulation,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "token_lengths": lengths,
        "kernels": kernels,
        "packages": package_versions(),
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "before": metric_snapshot(before),
        "train": metric_snapshot(train_result.metrics),
        "after": metric_snapshot(after),
        "adapter_tensors": len(adapter_keys),
        "adapter_sha256": sha256(adapter_path),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))
