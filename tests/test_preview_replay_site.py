import gzip
import json

from scripts.preview_replay_site import create_preview


def preview_export(tmp_path):
    root = tmp_path / "export"
    (root / "site" / "viewer").mkdir(parents=True)
    (root / "site" / "replays").mkdir()
    (root / "r2" / "games").mkdir(parents=True)
    (root / ".catan-replay-export").write_text("generated\n")
    (root / "site" / "index.html").write_text("home")
    (root / "site" / "replays" / "index.html").write_text("catalog")
    (root / "site" / "replays" / "watch.html").write_text("watch shell")
    (root / "site" / "replay.html").write_text("replay shell")
    (root / "site" / "viewer" / "index.html").write_text("viewer")
    (root / "r2" / "catalog.json").write_text('[{"game_id":"game-a"}]')
    payload = gzip.compress(json.dumps([{"state_index": 0}]).encode())
    (root / "r2" / "games" / "game-a.json.gz").write_bytes(payload)
    return root, payload


def test_preview_serves_site_catalog_and_compressed_replay(tmp_path):
    root, payload = preview_export(tmp_path)
    client = create_preview(root).test_client()

    assert client.get("/").text == "home"
    assert client.get("/replays").text == "catalog"
    assert client.get("/replays/watch?game_id=game-a").text == "watch shell"
    assert client.get("/replays/game-a").text == "replay shell"
    assert client.get("/viewer/").text == "viewer"
    assert client.get("/api/replays").get_json() == [{"game_id": "game-a"}]
    response = client.get("/api/replay-data/game-a")
    assert response.headers["Content-Encoding"] == "gzip"
    assert response.data == payload


def test_preview_rejects_missing_replay(tmp_path):
    root, _ = preview_export(tmp_path)
    response = create_preview(root).test_client().get("/api/replay-data/missing")
    assert response.status_code == 404
