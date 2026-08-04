"""Build a DPO dataset from raw teacher values and their source trajectories.

Only strict, timeout-free preferences are emitted. The preferred completion is
the teacher's best move and the rejected completion is the move the logged
policy actually played. Whole games are assigned to train or validation by
seed, so no board trajectory appears in both splits.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from catan_llm.schema import DecisionRecord, GameRecord
from catan_llm.serialize import decision_to_prompt, move_id


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _answer(move: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": f"answer: {move}"}]


def make_pair(
    game: GameRecord, decision: DecisionRecord, score: dict
) -> dict | None:
    """Validate one score row and return its strict preference pair."""
    if score.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported score schema: {score.get('schema_version')}")
    if score["game_id"] != game.game_id or decision.game_id != game.game_id:
        raise ValueError("game_id mismatch between score and trajectory")
    if score["seed"] != game.seed:
        raise ValueError(f"seed mismatch for {game.game_id}")
    if score["decision"] != decision.i:
        raise ValueError(
            f"decision mismatch for {game.game_id}: "
            f"{score['decision']} != {decision.i}"
        )

    legal_ids = [move_id(*action) for action in decision.legal_actions]
    logged_id = legal_ids[decision.chosen_action]
    if score["chosen_id"] != logged_id:
        raise ValueError(
            f"logged move mismatch for {game.game_id} decision {decision.i}: "
            f"{score['chosen_id']} != {logged_id}"
        )
    if score["best_id"] not in legal_ids:
        raise ValueError(
            f"teacher move is not legal for {game.game_id} decision {decision.i}"
        )

    if score["any_timeout"] or not score["strict_preference"]:
        return None
    if not score["best_value"] > score["chosen_value"]:
        raise ValueError("strict preference does not have a positive value gap")
    if score["best_id"] == score["chosen_id"]:
        raise ValueError("strict preference has identical chosen and rejected moves")

    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": decision_to_prompt(game, decision)},
        ],
        # DPO terminology: "chosen" is the preferred teacher move. The raw
        # evidence calls the policy's played move "chosen_id", so spell this
        # mapping out here to prevent an easy-to-miss reversal.
        "chosen": _answer(score["best_id"]),
        "rejected": _answer(score["chosen_id"]),
        "game_id": game.game_id,
        "seed": game.seed,
        "decision": decision.i,
        "move_type": score["move_type"],
        "teacher_value_gap": score["best_value"] - score["chosen_value"],
    }


def _trajectory_index(directory: Path) -> dict[str, Path]:
    index = {}
    for path in sorted(directory.glob("*.jsonl")):
        first = path.read_text().splitlines()[0]
        game_id = GameRecord.model_validate_json(first).game_id
        if game_id in index:
            raise ValueError(f"duplicate trajectory game_id: {game_id}")
        index[game_id] = path
    return index


def _load_trajectory(path: Path) -> tuple[GameRecord, dict[int, DecisionRecord]]:
    lines = path.read_text().splitlines()
    game = GameRecord.model_validate_json(lines[0])
    decisions = {
        decision.i: decision
        for decision in (
            DecisionRecord.model_validate_json(line) for line in lines[1:]
        )
    }
    return game, decisions


def build_dataset(
    scores_path: Path,
    trajectories_dir: Path,
    out_dir: Path,
    val_every: int = 10,
) -> dict:
    if val_every <= 1:
        raise ValueError("val_every must be greater than 1")
    trajectory_paths = _trajectory_index(trajectories_dir)
    if not trajectory_paths:
        raise ValueError(f"no trajectories found in {trajectories_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split: out_dir / f"{split}.jsonl" for split in ("train", "val")
    }
    writers = {
        split: path.open("w") for split, path in output_paths.items()
    }
    counts = Counter()
    split_games = {"train": set(), "val": set()}
    current_id = None
    current_game = None
    current_decisions = None
    try:
        with scores_path.open() as scores:
            for line in scores:
                score = json.loads(line)
                counts["input_decisions"] += 1
                if score["any_timeout"]:
                    counts["timed_out"] += 1
                elif score["strict_preference"]:
                    counts["strict_preferences"] += 1
                elif (
                    score["best_id"] != score["chosen_id"]
                    and score["best_value"] == score["chosen_value"]
                ):
                    counts["exact_ties"] += 1
                else:
                    counts["teacher_best_agreements"] += 1

                if score["game_id"] != current_id:
                    try:
                        trajectory_path = trajectory_paths[score["game_id"]]
                    except KeyError as error:
                        raise ValueError(
                            f"missing trajectory for {score['game_id']}"
                        ) from error
                    current_game, current_decisions = _load_trajectory(
                        trajectory_path
                    )
                    current_id = score["game_id"]
                try:
                    decision = current_decisions[score["decision"]]
                except KeyError as error:
                    raise ValueError(
                        f"missing decision {score['decision']} for {current_id}"
                    ) from error

                sample = make_pair(current_game, decision, score)
                if sample is None:
                    continue
                if sample["seed"] is None or sample["seed"] < EVAL_SEED_LIMIT:
                    raise ValueError(
                        f"training pair overlaps eval seeds: {sample['seed']}"
                    )
                split = "val" if sample["seed"] % val_every == 0 else "train"
                writers[split].write(
                    json.dumps(sample, separators=(",", ":")) + "\n"
                )
                counts[f"{split}_pairs"] += 1
                split_games[split].add(sample["game_id"])
    finally:
        for writer in writers.values():
            writer.close()

    if split_games["train"] & split_games["val"]:
        raise AssertionError("game leakage between train and validation")
    if counts["strict_preferences"] != (
        counts["train_pairs"] + counts["val_pairs"]
    ):
        raise AssertionError("not every strict preference was emitted")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "scores": str(scores_path),
        "scores_sha256": _sha256(scores_path),
        "trajectories": str(trajectories_dir),
        "val_every": val_every,
        **dict(counts),
        "train_games": len(split_games["train"]),
        "val_games": len(split_games["val"]),
        "train_sha256": _sha256(output_paths["train"]),
        "val_sha256": _sha256(output_paths["val"]),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-every", type=int, default=10)
    args = parser.parse_args()
    check_fixed_hashseed()
    metadata = build_dataset(
        Path(args.scores),
        Path(args.trajectories),
        Path(args.out),
        args.val_every,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
