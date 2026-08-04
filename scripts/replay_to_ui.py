"""Loads a trajectory into the catanatron web database for its replay UI."""

import argparse
from pathlib import Path

from catanatron.web.models import GameState, metadata, upsert_game_state
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from catan_llm.determinism import require_fixed_hashseed
from catan_llm.replay import replay_steps
from catan_llm.replay_metadata import save_replay_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/replays.sqlite"),
        help="sqlite file the catanatron web API reads",
    )
    parser.add_argument("--model", help="model name to display in the replay browser")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/replays.models.json"),
        help="model identity sidecar read by the replay browser",
    )
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{args.db.resolve()}")
    metadata.create_all(engine)

    game_id = None
    states = 0
    with Session(engine) as session:
        for step in replay_steps(args.trajectory):
            if game_id is None:
                game_id = step.game_record.game_id
                session.query(GameState).filter_by(uuid=game_id).delete()
                session.commit()
            upsert_game_state(step.game, session)
            states += 1

    if args.model:
        save_replay_model(args.metadata, game_id, args.model)

    print(f"{game_id}: {states} states -> {args.db}")
    print(f"replay: http://localhost:3000/replays/{game_id}")


if __name__ == "__main__":
    require_fixed_hashseed()
    main()
