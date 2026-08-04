"""Rewrite a reward cache's .index.json after a build was killed partway.

Config fields default from the existing index; a value that differs from
the existing index is refused without --force, so a casual rebuild cannot
silently re-stamp the cache with wrong provenance. A partial trailing line
left by a killed append is dropped before hashing.
"""

import argparse
import json
from pathlib import Path

from build_reward_cache import index_path, write_index
from run_dpo import sha256

CONFIG_KEYS = (
    "trajectories",
    "scenarios",
    "seed",
    "val_every",
    "hero_policy",
    "opponent_policy",
    "scorer_sha256",
    "catanatron_version",
)


def repair_truncated_tail(cache: Path) -> int:
    """Drop an unparseable partial trailing line; returns bytes removed.

    A killed append can only truncate the final line. A parseable tail that
    merely lost its newline is terminated instead of dropped.
    """
    data = cache.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        json.loads(tail)
    except json.JSONDecodeError:
        cache.write_bytes(data[:cut])
        return len(tail)
    cache.write_bytes(data + b"\n")
    return 0


def resolve_config(existing: dict | None, overrides: dict, force: bool) -> dict:
    """Merge CLI overrides over the existing index config, refusing drift."""
    config = {}
    for key in CONFIG_KEYS:
        override = overrides.get(key)
        recorded = existing.get(key) if existing else None
        if override is None and recorded is None:
            raise ValueError(f"{key} is not in the existing index; pass it explicitly")
        if (
            override is not None
            and recorded is not None
            and override != recorded
            and not force
        ):
            raise ValueError(
                f"{key} {override!r} differs from existing index value "
                f"{recorded!r}; pass --force to re-pin deliberately"
            )
        config[key] = override if override is not None else recorded
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--trajectories")
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--hero-policy")
    parser.add_argument("--opponent-policy")
    parser.add_argument("--scorer-sha256")
    parser.add_argument("--catanatron-version")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.cache.exists():
        parser.error(f"{args.cache}: no such cache file")

    removed = repair_truncated_tail(args.cache)
    if removed:
        print(f"dropped {removed}-byte partial trailing line")

    existing = None
    if index_path(args.cache).exists():
        existing = json.loads(index_path(args.cache).read_text())
        if (
            not removed
            and existing["cache_sha256"] == sha256(args.cache)
            and not args.force
        ):
            parser.error(f"{index_path(args.cache)}: already matches the cache")

    overrides = {
        "trajectories": args.trajectories,
        "scenarios": args.scenarios,
        "seed": args.seed,
        "val_every": args.val_every,
        "hero_policy": args.hero_policy,
        "opponent_policy": args.opponent_policy,
        "scorer_sha256": args.scorer_sha256,
        "catanatron_version": args.catanatron_version,
    }
    try:
        config = resolve_config(existing, overrides, args.force)
    except ValueError as error:
        parser.error(str(error))

    index = write_index(args.cache, config)
    if existing is not None:
        print(f"rows  {existing['rows']} -> {index['rows']}")
        print(f"games {existing['games']} -> {index['games']}")
    print(json.dumps(index, indent=2, sort_keys=True))
    print(f"-> {index_path(args.cache)}")


if __name__ == "__main__":
    main()
