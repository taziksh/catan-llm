"""Precompute engine-rollout rewards for per-decision GRPO.

One JSONL row per (game_id, decision, move_id, scenario_seed): the logged
decision is replayed, the hidden world is determinized from the hero's
information set, the move is forced, and scripted bots finish the game.
Scenario seeds are identical across the moves of one decision (paired
common random numbers) and derive exactly as in the label-stability audit.
The continuation RNG is seeded from a domain-separated derivation of the
scenario seed, so future dice/steals come from a stream independent of the
one that sampled the hidden world, while both are shared across the
candidate moves of a decision.
The cache is append-safe: rebuilding skips keys already present, and a
sidecar index records row counts, the build config, and the file hash.
"""

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

from catan_llm.determinism import EVAL_SEED_LIMIT, check_fixed_hashseed
from catan_llm.determinize import determinize
from catan_llm.extract import to_action
from catan_llm.replay import ReplayedDecision, replay_model_decisions
from catan_llm.schema import GameRecord, Player
from catan_llm.serialize import move_id
from catan_llm.simulation import GameOutcome, rollout_action
from run_dpo import sha256

DEFAULT_SCENARIOS = 8
DEFAULT_SEED = 42
HERO_POLICY = "alpha_beta"
OPPONENT_POLICY = "value_function"
VP_CAP = 10
VAL_EVERY = 10


def is_val_seed(seed: int | None) -> bool:
    """True for DPO-validation games, per build_dpo_dataset's seed split."""
    return seed is not None and seed % VAL_EVERY == 0


def scorer_fingerprint() -> dict:
    """Content identity of the code and engine that define rewards."""
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for name in (
        "catan_llm/replay.py",
        "catan_llm/determinize.py",
        "catan_llm/simulation.py",
        "scripts/build_reward_cache.py",
    ):
        digest.update((root / name).read_bytes())
    return {
        "scorer_sha256": digest.hexdigest(),
        "catanatron_version": version("catanatron"),
    }


def trajectory_header(path: Path) -> GameRecord:
    with path.open() as handle:
        header = GameRecord.model_validate_json(handle.readline())
    if header.seed is None or header.seed < EVAL_SEED_LIMIT:
        raise ValueError(f"{path.name}: trajectory seed in eval range: {header.seed}")
    return header


def scenario_seeds(seed: int, game_id: str, decision: int, count: int) -> list[int]:
    """Unique per-decision scenario seeds, identical to the audit's derivation.

    Lower counts are prefixes of higher ones, and string seeding does not
    depend on hash randomization.
    """
    rng = random.Random(f"{seed}:{game_id}:{decision}")
    seeds = []
    seen = set()
    while len(seeds) < count:
        candidate = rng.getrandbits(32)
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)
    return seeds


def continuation_seed(scenario_seed: int) -> int:
    """Continuation-stream seed, domain-separated from the world sampling."""
    return random.Random(f"{scenario_seed}:continuation").getrandbits(63)


def continuation_reward(outcome: GameOutcome, hero: Player) -> float:
    """Project reward: won + 0.1 * min(vp, 10) / 10. Truncation keeps the VP term."""
    won = outcome.winner == hero
    return float(won) + 0.1 * min(outcome.victory_points[hero], VP_CAP) / VP_CAP


def row_key(row: dict) -> tuple:
    return (row["game_id"], row["decision"], row["move_id"], row["scenario_seed"])


def hero_color(game_record: GameRecord) -> Player:
    heroes = [color for color, kind in game_record.seats.items() if kind == "llm"]
    if len(heroes) != 1:
        raise ValueError(f"expected exactly one llm seat, found {len(heroes)}")
    return heroes[0]


def playable_move_ids(replayed: ReplayedDecision) -> list[str]:
    catan_map = replayed.game.state.board.map
    return [
        move_id(*to_action(action, catan_map))
        for action in replayed.game.playable_actions
    ]


def score_moves(
    replayed: ReplayedDecision,
    move_ids: list[str],
    seeds: list[int],
    hero_policy: str = HERO_POLICY,
    opponent_policy: str = OPPONENT_POLICY,
) -> list[dict]:
    """Score moves over paired determinized scenarios.

    Each seed samples one hidden world; every requested move is forced in
    that same world and the continuation reuses the seed, so outcome
    differences across moves reflect the moves alone.
    """
    game = replayed.game
    hero = hero_color(replayed.game_record)
    colors = [Player(color.value) for color in game.state.colors]
    hero_index = colors.index(hero)
    policies = {
        color: hero_policy if color == hero else opponent_policy
        for color in colors
    }
    rows = []
    for seed in seeds:
        world = determinize(game, hero_index, seed)
        future_seed = continuation_seed(seed)
        catan_map = world.state.board.map
        indices = {
            move_id(*to_action(action, catan_map)): index
            for index, action in enumerate(world.playable_actions)
        }
        for move in move_ids:
            if move not in indices:
                raise ValueError(f"move not legal in sampled world: {move}")
            outcome = rollout_action(
                world, indices[move], policies, seed=future_seed
            )
            rows.append(
                {
                    "game_id": replayed.game_record.game_id,
                    "decision": replayed.decision.i,
                    "move_id": move,
                    "scenario_seed": seed,
                    "won": outcome.winner == hero,
                    "hero_vp": outcome.victory_points[hero],
                    "truncated": outcome.truncated,
                    "turns": outcome.turns,
                    "reward": continuation_reward(outcome, hero),
                }
            )
    return rows


def score_decision_moves(
    path: str,
    decision_index: int,
    move_ids: list[str],
    seeds: list[int],
    hero_policy: str = HERO_POLICY,
    opponent_policy: str = OPPONENT_POLICY,
) -> list[dict]:
    """Replay one logged decision and score moves. Safe in a worker process."""
    for replayed in replay_model_decisions(Path(path)):
        if replayed.decision.i == decision_index:
            return score_moves(
                replayed, move_ids, seeds, hero_policy, opponent_policy
            )
    raise ValueError(f"{path}: decision {decision_index} not replayed")


def load_cache(path: Path) -> dict[tuple, dict]:
    """Read cache rows keyed (game_id, decision, move_id, scenario_seed)."""
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row_key(row)] = row
    return rows


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def index_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".index.json")


def write_index(cache_path: Path, config: dict) -> dict:
    rows = load_cache(cache_path)
    index = {
        "cache_sha256": sha256(cache_path),
        "rows": len(rows),
        "games": len({key[0] for key in rows}),
        "decisions": len({key[:2] for key in rows}),
        **config,
    }
    index_path(cache_path).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    return index


def checked_cache(cache_path: Path) -> tuple[dict[tuple, dict], dict]:
    """Load a cache and its index, refusing a stale index hash."""
    index = json.loads(index_path(cache_path).read_text())
    if index["cache_sha256"] != sha256(cache_path):
        raise RuntimeError(
            f"{cache_path}: index hash does not match the cache file; "
            "rebuild the index"
        )
    return load_cache(cache_path), index


def _check_config(existing: dict, config: dict) -> None:
    """Refuse appends whose seed, policies, or scorer differ from the cache."""
    for key in (
        "seed",
        "hero_policy",
        "opponent_policy",
        "scorer_sha256",
        "catanatron_version",
    ):
        if existing.get(key) != config[key]:
            raise RuntimeError(
                f"cache index {key} mismatch: "
                f"{existing.get(key)!r} != {config[key]!r}"
            )


def score_game(
    path: str,
    scenarios: int,
    seed: int,
    max_decisions: int | None,
    hero_policy: str,
    opponent_policy: str,
    skip: frozenset,
) -> list[dict]:
    """Score the missing rows of one trajectory. Safe in a worker process."""
    with open(path) as handle:
        header = GameRecord.model_validate_json(handle.readline())
    if header.seed is None or header.seed < EVAL_SEED_LIMIT:
        raise ValueError(f"{path}: trajectory seed in eval range: {header.seed}")

    rows = []
    for produced, replayed in enumerate(replay_model_decisions(Path(path))):
        if max_decisions is not None and produced >= max_decisions:
            break
        decision = replayed.decision.i
        moves = playable_move_ids(replayed)
        for scenario_seed in scenario_seeds(
            seed, header.game_id, decision, scenarios
        ):
            missing = [
                move for move in moves
                if (decision, move, scenario_seed) not in skip
            ]
            if missing:
                rows.extend(
                    score_moves(
                        replayed,
                        missing,
                        [scenario_seed],
                        hero_policy,
                        opponent_policy,
                    )
                )
    return rows


def _score_game_star(args):
    return score_game(*args)


def build_cache(
    trajectories: Path,
    out: Path,
    scenarios: int = DEFAULT_SCENARIOS,
    seed: int = DEFAULT_SEED,
    workers: int = 1,
    max_games: int | None = None,
    max_decisions: int | None = None,
    hero_policy: str = HERO_POLICY,
    opponent_policy: str = OPPONENT_POLICY,
) -> dict:
    paths = sorted(Path(trajectories).glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no .jsonl trajectories found in {trajectories}")
    headers = {path: trajectory_header(path) for path in paths}
    paths = [path for path in paths if not is_val_seed(headers[path].seed)]
    if max_games is not None:
        paths = paths[:max_games]
    by_game = {}
    for path in paths:
        game_id = headers[path].game_id
        if game_id in by_game:
            raise ValueError(
                f"duplicate trajectories for {game_id}: "
                f"{by_game[game_id].name}, {path.name}"
            )
        by_game[game_id] = path

    config = {
        "trajectories": str(trajectories),
        "scenarios": scenarios,
        "seed": seed,
        "val_every": VAL_EVERY,
        "hero_policy": hero_policy,
        "opponent_policy": opponent_policy,
        **scorer_fingerprint(),
    }
    if index_path(out).exists():
        _check_config(json.loads(index_path(out).read_text()), config)
    elif out.exists():
        raise RuntimeError(f"{out}: cache file has no index")
    cached = load_cache(out)

    worker_args = []
    for path in paths:
        skip = frozenset(
            key[1:] for key in cached if key[0] == headers[path].game_id
        )
        worker_args.append(
            (str(path), scenarios, seed, max_decisions,
             hero_policy, opponent_policy, skip)
        )
    added = 0
    if workers == 1:
        for rows in map(_score_game_star, worker_args):
            append_rows(out, rows)
            added += len(rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_score_game_star, args) for args in worker_args
            ]
            for future in as_completed(futures):
                rows = future.result()
                append_rows(out, rows)
                added += len(rows)
    return {**write_index(out, config), "rows_added": added}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-games", type=int)
    parser.add_argument(
        "--max-decisions",
        type=int,
        help="score at most this many decisions per game (canary)",
    )
    args = parser.parse_args()

    check_fixed_hashseed()
    if args.scenarios <= 0:
        parser.error("--scenarios must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    index = build_cache(
        args.trajectories,
        args.out,
        scenarios=args.scenarios,
        seed=args.seed,
        workers=args.workers,
        max_games=args.max_games,
        max_decisions=args.max_decisions,
    )
    print(json.dumps(index, indent=2, sort_keys=True))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
