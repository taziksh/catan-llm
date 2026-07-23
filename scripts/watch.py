"""Steps through a logged trajectory, verifying each decision against a replay."""

import argparse
import sys
import tempfile
from pathlib import Path

from catanatron import Game

from catan_llm.bots import BOTS, COLORS
from catan_llm.determinism import require_fixed_hashseed
from catan_llm.extract import TrajectoryAccumulator
from catan_llm.schema import ActionType, Building, DecisionRecord, GameRecord
from catan_llm.serialize import decision_to_prompt


def replay_lines(game_rec):
    seats = {player.value: bot for player, bot in game_rec.seats.items()}
    players = [BOTS[seats[c.value]](c) for c in COLORS if c.value in seats]
    game = Game(players, seed=game_rec.seed)
    game.id = game_rec.game_id
    accumulator = TrajectoryAccumulator(tempfile.mkdtemp())
    game.play(accumulators=[accumulator])
    return accumulator.path.read_text().splitlines()


RESOURCE_EMOJI = {"WOOD": "🪵", "BRICK": "🧱", "SHEEP": "🐑", "WHEAT": "🌾", "ORE": "🪨"}


def emojify(value):
    if isinstance(value, str):
        return RESOURCE_EMOJI.get(value, value)
    if isinstance(value, list):
        return "[" + ", ".join(str(emojify(v)) for v in value) + "]"
    return value


def fmt_hand(hand):
    return " ".join(f"{RESOURCE_EMOJI[resource.value]}{n}" for resource, n in hand.items())


def fmt_payload(payload):
    return "" if payload is None else f" {emojify(payload)}"


def show_legal(legal, chosen):
    rows = [
        f" {'>' if idx == chosen else ' '} {idx:3d}. {t.value}{fmt_payload(p)}"
        for idx, (t, p) in enumerate(legal)
    ]
    if len(rows) <= 14:
        print("\n".join(rows))
        return
    print("\n".join(rows[:12]))
    if chosen >= 12:
        print("   ...")
        print(rows[chosen])
    print(f"   ... {len(rows)} total")


def show_decision(rec, seats, verify_status):
    print("=" * 72)
    status = f" · verify {verify_status}" if verify_status else ""
    print(
        f"[{rec.i}] turn {rec.turn} · {rec.phase.value} · "
        f"{rec.actor.value} ({seats[rec.actor]}){status}"
    )
    buildings = ", ".join(
        f"{node}:{color.value[0]}{'S' if kind == Building.SETTLEMENT else 'C'}"
        for node, color, kind in rec.board.buildings
    )
    roads = ", ".join(f"{u}-{v}:{color.value[0]}" for u, v, color in rec.board.roads)
    print(f"buildings: {buildings or '-'}")
    print(f"roads: {roads or '-'}")
    print(f"🦹 @ tile {rec.board.robber}")
    for player, ps in rec.players.items():
        badges = (" LR" if ps.has_longest_road else "") + (
            " LA" if ps.has_largest_army else ""
        )
        devs = "".join(
            f" {card.value[:4]}x{n}" for card, n in ps.devs_in_hand.items() if n
        )
        print(
            f"{player.value:6s} {fmt_hand(ps.hand)} | "
            f"vp {ps.vps_public}({ps.vps_actual}) | road {ps.road_len}{badges}{devs}"
        )
    print(f"bank   {fmt_hand(rec.bank.resources)} | dev {rec.bank.dev_cards_left}")
    show_legal(rec.legal_actions, rec.chosen_action)
    if rec.result is not None:
        chosen_type, _ = rec.legal_actions[rec.chosen_action]
        icon = "🎲 " if chosen_type == ActionType.ROLL else ""
        print(f"result: {icon}{emojify(rec.result)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--prompt", action="store_true", help="show the model prompt")
    parser.add_argument("--check", action="store_true", help="verify all, no stepping")
    args = parser.parse_args()

    lines = Path(args.file).read_text().splitlines()
    game_rec = GameRecord.model_validate_json(lines[0])
    decisions = [DecisionRecord.model_validate_json(line) for line in lines[1:]]

    replayed = None
    if not args.no_verify:
        try:
            replayed = replay_lines(game_rec)
        except KeyError as e:
            print(f"replay-verify off: no bot for seat {e}")

    if args.check:
        if replayed is None:
            sys.exit("--check needs a replay")
        mismatches = [
            i for i, (a, b) in enumerate(zip(lines, replayed)) if a != b
        ] + list(range(min(len(lines), len(replayed)), max(len(lines), len(replayed))))
        print(f"{args.file}: {len(decisions)} decisions, {len(mismatches)} mismatches")
        sys.exit(1 if mismatches else 0)

    seats = ", ".join(f"{p.value}={bot}" for p, bot in game_rec.seats.items())
    print(f"{game_rec.game_id} · seed {game_rec.seed} · {seats}")
    outcome = game_rec.outcome
    print(f"{len(decisions)} decisions · winner {outcome.winner and outcome.winner.value} in {outcome.turns} turns")

    i = 0
    while 0 <= i < len(decisions):
        verify_status = None
        if replayed:
            verify_status = "ok" if replayed[i + 1] == lines[i + 1] else "MISMATCH"
        if args.prompt:
            print(decision_to_prompt(game_rec, decisions[i]))
        else:
            show_decision(decisions[i], game_rec.seats, verify_status)
        try:
            key = input("[enter]=next, <i>=jump, q=quit> ").strip()
        except EOFError:
            break
        if key == "q":
            break
        i = int(key) if key.isdigit() else i + 1
    print(f"outcome: {outcome.model_dump_json()}")


if __name__ == "__main__":
    require_fixed_hashseed()
    main()
