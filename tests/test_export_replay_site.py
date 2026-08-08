import gzip
import json
import sqlite3

import pytest

from scripts.deploy_replay_site import (
    deploy_pages,
    load_verified_objects,
    upload_command,
    upload_r2,
)
from scripts.export_replay_site import (
    HOME_HTML,
    INDEX_HTML,
    WATCH_HTML,
    game_bundle,
    insert_meta,
    publication_exclusion,
    structural_error,
    write_site_shell,
)


def catalog_entry(model="real-eval", winner="RED"):
    return {"model": model, "winner": winner}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((4, 4, 0, 3), None),
        ((3, 3, 1, 3), "missing-initial-state"),
        ((4, 3, 0, 2), "duplicate-state-index"),
        ((3, 3, 0, 3), "state-index-gap"),
    ],
)
def test_structural_error(values, expected):
    assert structural_error(*values) == expected


@pytest.mark.parametrize("model", [None, "smoke", "v2_probe", "swap-screen"])
def test_diagnostic_runs_are_excluded_by_default(model):
    assert publication_exclusion(
        catalog_entry(model),
        None,
        include_diagnostics=False,
        include_models=[],
        exclude_models=[],
    ) == "diagnostic-or-unlabeled"


def test_explicit_model_selection_can_include_diagnostic_run():
    assert publication_exclusion(
        catalog_entry("v2_probe"),
        None,
        include_diagnostics=False,
        include_models=["*_probe"],
        exclude_models=[],
    ) is None


def test_integrity_and_winner_checks_precede_model_filters():
    options = {
        "include_diagnostics": True,
        "include_models": [],
        "exclude_models": [],
    }
    assert (
        publication_exclusion(catalog_entry(), "state-index-gap", **options)
        == "state-index-gap"
    )
    assert (
        publication_exclusion(catalog_entry(winner=None), None, **options)
        == "no-winner"
    )


def test_game_bundle_is_deterministic_and_valid_json():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "create table game_states "
        "(id integer primary key, uuid text, state_index integer, state text)"
    )
    states = [{"state_index": index, "value": index * 2} for index in range(3)]
    connection.executemany(
        "insert into game_states(uuid,state_index,state) values(?,?,?)",
        [("game-a", state["state_index"], json.dumps(state)) for state in states],
    )

    first, raw_bytes = game_bundle(connection, "game-a", 3)
    second, _ = game_bundle(connection, "game-a", 3)

    assert first == second
    assert json.loads(gzip.decompress(first)) == states
    assert raw_bytes == len(gzip.decompress(first))


def test_game_bundle_rejects_state_gaps():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "create table game_states "
        "(id integer primary key, uuid text, state_index integer, state text)"
    )
    connection.executemany(
        "insert into game_states(uuid,state_index,state) values(?,?,?)",
        [("game-a", 0, "{}"), ("game-a", 2, "{}")],
    )

    with pytest.raises(ValueError, match="non-contiguous"):
        game_bundle(connection, "game-a", 3)


def test_insert_meta_adds_hosted_mode_after_viewport():
    html = (
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Replay</title>"
    )
    result = insert_meta(html, "replay-read-only", "true")
    assert '<meta name="replay-read-only" content="true">' in result


def test_hosted_navigation_has_project_home_and_scoped_replay_routes():
    assert 'href="/replays"' in HOME_HTML.read_text()
    assert "/replays/watch?game_id=" in INDEX_HTML.read_text()
    assert 'id="all-replays"' in WATCH_HTML.read_text()


def test_write_site_shell_builds_home_catalog_watch_and_legacy_redirect(tmp_path):
    write_site_shell(tmp_path)

    assert "Learning to play the whole game" in (tmp_path / "index.html").read_text()
    assert 'name="replay-read-only"' in (
        tmp_path / "replays" / "index.html"
    ).read_text()
    assert 'name="replay-viewer-url"' in (
        tmp_path / "replays" / "watch.html"
    ).read_text()
    assert (tmp_path / "replay.html").is_file()
    assert (tmp_path / "_redirects").read_text() == (
        "/replay /replays/watch 301\n"
    )


def test_deploy_verifies_manifest_and_sets_gzip_metadata(tmp_path):
    root = tmp_path / "export"
    path = root / "r2" / "games" / "game-a.json.gz"
    path.parent.mkdir(parents=True)
    (root / ".catan-replay-export").write_text("generated\n")
    path.write_bytes(b"payload")
    report = {
        "objects": [
            {
                "key": "games/game-a.json.gz",
                "bytes": 7,
                "sha256": (
                    "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"
                ),
            }
        ]
    }
    (root / "export-report.json").write_text(json.dumps(report))

    [(item, verified_path)] = load_verified_objects(root)
    command = upload_command("bucket", item, verified_path)

    assert verified_path == path
    assert "bucket/games/game-a.json.gz" in command
    assert "--content-encoding=gzip" in command


def test_deploy_rejects_changed_artifact(tmp_path):
    root = tmp_path / "export"
    path = root / "r2" / "catalog.json"
    path.parent.mkdir(parents=True)
    (root / ".catan-replay-export").write_text("generated\n")
    path.write_text("changed")
    (root / "export-report.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"key": "catalog.json", "bytes": 7, "sha256": "wrong"}
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_objects(root)


def test_deploy_rejects_object_path_outside_export(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    (root / ".catan-replay-export").write_text("generated\n")
    (root / "export-report.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"key": "../../secret", "bytes": 0, "sha256": "unused"}
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="unsafe object key"):
        load_verified_objects(root)


def test_r2_catalog_is_uploaded_after_all_game_objects(monkeypatch, tmp_path):
    uploaded = []

    def record_upload(bucket, item, path, dry_run):
        uploaded.append(item["key"])
        return item["key"]

    monkeypatch.setattr("scripts.deploy_replay_site.upload_one", record_upload)
    objects = [
        ({"key": "catalog.json"}, tmp_path / "catalog.json"),
        ({"key": "games/a.json.gz"}, tmp_path / "a.json.gz"),
        ({"key": "games/b.json.gz"}, tmp_path / "b.json.gz"),
    ]

    upload_r2("bucket", objects, jobs=2, dry_run=False)

    assert set(uploaded[:-1]) == {"games/a.json.gz", "games/b.json.gz"}
    assert uploaded[-1] == "catalog.json"


def test_pages_deploy_requires_compiled_viewer(tmp_path):
    with pytest.raises(ValueError, match="compiled replay viewer"):
        deploy_pages(tmp_path, "project", dry_run=True)
