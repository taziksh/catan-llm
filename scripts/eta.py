"""Estimate completion time of the local reward-cache fill.

Each run snapshots (timestamp, row count) to a state file in $HOME, then
extrapolates from measured throughput: rate is computed between the first
and latest snapshot, and the target row count comes from the measured mean
of the 120 baseline games (407,096 rows) scaled to all 270 scorable games.
Rows land in whole-game bursts, so spans under an hour are noise.

Usage: python3 scripts/eta.py
"""

import json
import time
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "data/reward_cache/local_fill.jsonl"
STATE = Path.home() / ".local_fill_eta.json"
BASE_ROWS = 407_096
BASE_GAMES = 120
TOTAL_GAMES = 270


def main() -> None:
    games = set()
    rows = 0
    with CACHE.open() as handle:
        for line in handle:
            rows += 1
            games.add(json.loads(line)["game_id"])
    now = time.time()
    hist = json.loads(STATE.read_text()) if STATE.exists() else []
    hist.append([now, rows])
    STATE.write_text(json.dumps(hist))

    target = BASE_ROWS / BASE_GAMES * TOTAL_GAMES
    print(f"games {len(games)}/{TOTAL_GAMES} done, {TOTAL_GAMES - len(games)} left")
    print(f"rows {rows:,} / ~{target:,.0f}")

    if len(hist) < 2:
        print("baseline planted -- run again in an hour or more")
        return
    (t0, r0), (t1, r1) = hist[0], hist[-1]
    span = t1 - t0
    if span < 3600:
        print(f"span {span / 60:.0f} min -- too short for a real rate, run again later")
        return
    rate = (r1 - r0) / span
    if rate <= 0:
        print(f"no rows in {span / 3600:.1f}h -- builder stalled?")
        return
    eta = now + (target - rows) / rate
    print(
        f"rate {rate * 3600:,.0f} rows/h over {span / 3600:.1f}h span"
        f" -> ETA {time.strftime('%a %H:%M', time.localtime(eta))}"
    )


if __name__ == "__main__":
    main()
