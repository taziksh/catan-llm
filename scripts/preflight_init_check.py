"""Paired rollout-value comparison of two inits over logged decisions."""

import argparse
import json
import random
from pathlib import Path

from catan_llm.determinism import check_fixed_hashseed
from build_reward_cache import checked_cache
from run_grpo import render_prompt
from run_grpo_trl import build_rows, cached_value, check_cache_index, parse_completion

MAX_NEW_TOKENS = 32
BOOTSTRAP_RESAMPLES = 2000


def greedy_moves(model, tokenizer, rows, batch_size: int) -> list[str | None]:
    """Greedy-decode every row's prompt and parse each to a move id."""
    import torch

    moves = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [render_prompt(row["prompt"], tokenizer) for row in batch]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(
            output[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        moves.extend(
            parse_completion(text, row["legal_moves"])
            for text, row in zip(texts, batch, strict=True)
        )
    return moves


def paired_records(
    rows, base_moves, adapter_moves, cache, seed: int, scenarios: int
) -> list[dict]:
    """Value both arms' moves per decision, invalid parses scoring 0."""
    records = []
    for row, base_move, adapter_move in zip(
        rows, base_moves, adapter_moves, strict=True
    ):
        values = {}
        for arm, move in (("base", base_move), ("adapter", adapter_move)):
            values[arm] = (
                cached_value(
                    cache, seed, scenarios, row["game_id"], row["decision"], move
                )
                if move is not None
                else 0.0
            )
        records.append(
            {
                "game_id": row["game_id"],
                "decision": row["decision"],
                "base_move": base_move,
                "adapter_move": adapter_move,
                "base_value": values["base"],
                "adapter_value": values["adapter"],
            }
        )
    return records


def summarize(records: list[dict], seed: int) -> dict:
    """Mean adapter-minus-base value gap with a by-game bootstrap CI."""
    by_game: dict[str, list[float]] = {}
    for record in records:
        diff = record["adapter_value"] - record["base_value"]
        by_game.setdefault(record["game_id"], []).append(diff)
    games = sorted(by_game)
    diffs = [diff for game in games for diff in by_game[game]]
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [
            diff
            for _ in games
            for diff in by_game[rng.choice(games)]
        ]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "decisions": len(records),
        "games": len(games),
        "mean_diff": sum(diffs) / len(diffs),
        "ci95_low": means[int(0.025 * len(means))],
        "ci95_high": means[int(0.975 * len(means)) - 1],
        "base_invalid_rate": sum(
            record["base_move"] is None for record in records
        ) / len(records),
        "adapter_invalid_rate": sum(
            record["adapter_move"] is None for record in records
        ) / len(records),
    }


def run(args) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from run_dpo import load_text_only_qwen35

    check_fixed_hashseed()
    paths = sorted(Path(args.states).glob("*.jsonl"))
    rows = build_rows(paths, args.seed, args.max_states)
    cache, index = checked_cache(args.cache)
    check_cache_index(index, args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_text_only_qwen35(args.model).to("cuda")
    base_moves = greedy_moves(model, tokenizer, rows, args.batch_size)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    adapter_moves = greedy_moves(model, tokenizer, rows, args.batch_size)
    del model
    torch.cuda.empty_cache()

    records = paired_records(
        rows, base_moves, adapter_moves, cache, args.seed, args.scenarios
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return summarize(records, args.seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--states", type=Path, default=Path("data/dagger_traces/r3pairs")
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=4000)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.max_states is not None and args.max_states <= 0:
        parser.error("--max-states must be positive")
    if args.scenarios <= 0:
        parser.error("--scenarios must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
