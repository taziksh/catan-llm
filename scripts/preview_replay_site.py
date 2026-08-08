"""Preview a generated replay site with its local R2 artifacts."""

import argparse
import re
from pathlib import Path

from flask import Flask, Response, send_file, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "data" / "replay_cloudflare"
SAFE_GAME_ID = re.compile(r"[A-Za-z0-9_.-]+")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    return parser.parse_args()


def create_preview(export_root: Path) -> Flask:
    export_root = export_root.resolve()
    site = export_root / "site"
    r2 = export_root / "r2"
    if not (export_root / ".catan-replay-export").is_file():
        raise ValueError(f"not a generated replay export: {export_root}")

    app = Flask(__name__, static_folder=None)

    @app.get("/api/replays")
    def catalog():
        return send_file(r2 / "catalog.json", mimetype="application/json")

    @app.get("/api/replay-data/<game_id>")
    def replay_data(game_id):
        if SAFE_GAME_ID.fullmatch(game_id) is None:
            return {"error": "invalid game id"}, 400
        path = r2 / "games" / f"{game_id}.json.gz"
        if not path.is_file():
            return {"error": "replay not found"}, 404
        response = send_file(path, mimetype="application/json")
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/replays/<game_id>")
    def replay(game_id):
        return send_file(site / "replay.html")

    @app.get("/replays")
    @app.get("/replays/")
    def replay_catalog():
        return send_file(site / "replays" / "index.html")

    @app.get("/replays/watch")
    def replay_watch():
        return send_file(site / "replays" / "watch.html")

    @app.get("/")
    def index():
        return send_file(site / "index.html")

    @app.get("/<path:asset>")
    def static_asset(asset):
        candidate = site / asset
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            return Response("not found", status=404)
        return send_from_directory(site, candidate.relative_to(site))

    return app


if __name__ == "__main__":
    args = parse_args()
    create_preview(args.export).run(host=args.host, port=args.port)
