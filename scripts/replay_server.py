"""Serves the catanatron web API plus an index of loaded replays."""

import os
import pickle
import tempfile
from pathlib import Path

from catanatron.state_functions import get_actual_victory_points
from flask import jsonify, request, send_file
from sqlalchemy import func

from catan_llm.determinism import require_fixed_hashseed
from catan_llm.replay import replay_steps
from catan_llm.replay_metadata import load_replay_models, save_replay_model

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = Path(__file__).with_name("replay_index.html")
WATCH_HTML = Path(__file__).with_name("replay_watch.html")
DEFAULT_DB = ROOT / "data" / "replays.sqlite"
DEFAULT_METADATA = ROOT / "data" / "replays.models.json"

def display_model_name(model: str | None) -> str | None:
    """Remove a provider prefix while preserving the recorded mode label."""
    if model is None:
        return None
    return model.rsplit("/", 1)[-1]


def parse_game_id(game_id: str, seat_count: int) -> tuple[list[str] | None, int | None]:
    """Read the seat lineup and seed out of a game id."""
    stem, _, tail = game_id.rpartition("_s")
    if not stem:
        return None, None
    seed = tail.split("_", 1)[0]
    kinds = stem.split("-")
    if not seed.isdigit() or len(kinds) != seat_count:
        return None, None
    return kinds, int(seed)


def create_server():
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{DEFAULT_DB}")

    from catanatron.web import create_app
    from catanatron.web.models import GameState, database_session, upsert_game_state

    app = create_app()

    @app.get("/")
    def index():
        return send_file(INDEX_HTML)

    @app.get("/replays/<game_id>")
    def watch_replay(game_id):
        return send_file(WATCH_HTML)

    @app.get("/api/replays")
    def list_replays():
        games = []
        replay_models = load_replay_models(DEFAULT_METADATA)
        with database_session() as session:
            latest = (
                session.query(
                    GameState.uuid.label("uuid"),
                    func.max(GameState.state_index).label("last_index"),
                    func.max(GameState.id).label("load_order"),
                )
                .group_by(GameState.uuid)
                .subquery()
            )
            rows = (
                session.query(
                    GameState,
                    latest.c.last_index,
                    latest.c.load_order,
                )
                .join(latest, GameState.id == latest.c.load_order)
                .all()
            )
            for final, last_index, load_order in rows:
                uuid = final.uuid
                game = pickle.loads(final.pickle_data)
                colors = [color.value for color in game.state.colors]
                winner = game.winning_color()
                kinds, seed = parse_game_id(uuid, len(colors))
                model = replay_models.get(uuid)
                model_color = None
                if kinds and "llm" in kinds:
                    model_color = colors[kinds.index("llm")]
                games.append({
                    "game_id": uuid,
                    "load_order": load_order,
                    "states": last_index + 1,
                    "turns": game.state.num_turns,
                    "colors": colors,
                    "seat_kinds": kinds,
                    "model": model,
                    "model_label": display_model_name(model),
                    "model_color": model_color,
                    "seed": seed,
                    "victory_points": {
                        color.value: get_actual_victory_points(game.state, color)
                        for color in game.state.colors
                    },
                    "winner": winner.value if winner else None,
                })
        games.sort(key=lambda entry: entry["load_order"], reverse=True)
        return jsonify(games)

    @app.post("/api/replays")
    def upload_replay():
        upload = request.files.get("trajectory")
        if upload is None or not upload.filename:
            return jsonify({"error": "no trajectory file in the request"}), 400

        game_id = None
        states = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / Path(upload.filename).name
            upload.save(path)
            try:
                with database_session() as session:
                    for step in replay_steps(path):
                        if game_id is None:
                            game_id = step.game_record.game_id
                            session.query(GameState).filter_by(uuid=game_id).delete()
                            session.commit()
                        upsert_game_state(step.game, session)
                        states += 1
            except (ValueError, RuntimeError) as error:
                if game_id is not None:
                    with database_session() as session:
                        session.query(GameState).filter_by(uuid=game_id).delete()
                        session.commit()
                return jsonify({"error": str(error)}), 400

        model = request.form.get("model", "").strip()
        if model:
            save_replay_model(DEFAULT_METADATA, game_id, model)

        return jsonify({"game_id": game_id, "states": states})

    return app


if __name__ == "__main__":
    require_fixed_hashseed()
    create_server().run(port=5001)
