"""Build a read-only Cloudflare Pages + R2 replay site.

The source SQLite database stores both JSON and pickle copies of every state.
This exporter publishes only JSON, bundled and gzip-compressed once per game.
Diagnostic runs are excluded by default but remain untouched in the source DB.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from catanatron.state_functions import get_actual_victory_points

from catan_llm.replay_metadata import load_replay_models
from scripts.replay_server import display_model_name, parse_game_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "replays.sqlite"
DEFAULT_METADATA = ROOT / "data" / "replays.models.json"
DEFAULT_OUTPUT = ROOT / "data" / "replay_cloudflare"
DEFAULT_CATANATRON_UI = ROOT.parent / "catanatron" / "ui"
INDEX_HTML = Path(__file__).with_name("replay_index.html")
WATCH_HTML = Path(__file__).with_name("replay_watch.html")
HOME_HTML = Path(__file__).with_name("replay_home.html")
EXPORT_MARKER = ".catan-replay-export"
SAFE_GAME_ID = re.compile(r"[A-Za-z0-9_.-]+")
DIAGNOSTIC_MARKERS = ("smoke", "probe", "swap-screen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catanatron-ui", type=Path, default=DEFAULT_CATANATRON_UI)
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="include smoke, probe, swap-screen, and unlabeled games",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        metavar="GLOB",
        help="include only matching model/run labels; repeatable",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        metavar="GLOB",
        help="exclude matching model/run labels; repeatable",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="export only the newest N selected games (useful for smoke tests)",
    )
    parser.add_argument(
        "--skip-viewer-build",
        action="store_true",
        help="omit the Catanatron UI build",
    )
    return parser.parse_args()


def structural_error(
    row_count: int, distinct_states: int, first_state: int, last_state: int
) -> str | None:
    if first_state != 0:
        return "missing-initial-state"
    if row_count != distinct_states:
        return "duplicate-state-index"
    if distinct_states != last_state + 1:
        return "state-index-gap"
    return None


def is_diagnostic_model(model: str | None) -> bool:
    if model is None:
        return True
    normalized = model.lower()
    return any(marker in normalized for marker in DIAGNOSTIC_MARKERS)


def matches_any(value: str | None, patterns: list[str]) -> bool:
    candidate = value or ""
    return any(fnmatch.fnmatchcase(candidate, pattern) for pattern in patterns)


def publication_exclusion(
    entry: dict,
    integrity_error: str | None,
    *,
    include_diagnostics: bool,
    include_models: list[str],
    exclude_models: list[str],
) -> str | None:
    if integrity_error is not None:
        return integrity_error
    if entry["winner"] is None:
        return "no-winner"

    model = entry["model"]
    if matches_any(model, exclude_models):
        return "excluded-model"
    if include_models:
        return None if matches_any(model, include_models) else "not-selected-model"
    if not include_diagnostics and is_diagnostic_model(model):
        return "diagnostic-or-unlabeled"
    return None


def catalog_rows(
    connection: sqlite3.Connection, replay_models: dict[str, str]
) -> tuple[list[dict], dict[str, str | None]]:
    rows = connection.execute(
        """
        WITH stats AS (
            SELECT
                uuid,
                COUNT(*) AS row_count,
                COUNT(DISTINCT state_index) AS distinct_states,
                MIN(state_index) AS first_state,
                MAX(state_index) AS last_state
            FROM game_states
            GROUP BY uuid
        ),
        latest AS (
            SELECT states.uuid, MAX(states.id) AS final_id
            FROM game_states AS states
            JOIN stats
              ON stats.uuid = states.uuid
             AND stats.last_state = states.state_index
            GROUP BY states.uuid
        )
        SELECT
            stats.uuid,
            stats.row_count,
            stats.distinct_states,
            stats.first_state,
            stats.last_state,
            latest.final_id,
            final.pickle_data
        FROM stats
        JOIN latest ON latest.uuid = stats.uuid
        JOIN game_states AS final ON final.id = latest.final_id
        ORDER BY latest.final_id DESC
        """
    ).fetchall()

    entries = []
    integrity = {}
    for (
        game_id,
        row_count,
        distinct_states,
        first_state,
        last_state,
        load_order,
        pickle_data,
    ) in rows:
        integrity[game_id] = structural_error(
            row_count, distinct_states, first_state, last_state
        )
        game = pickle.loads(pickle_data)
        colors = [color.value for color in game.state.colors]
        winner = game.winning_color()
        seat_kinds, seed = parse_game_id(game_id, len(colors))
        model = replay_models.get(game_id)
        model_color = None
        if seat_kinds and "llm" in seat_kinds:
            model_color = colors[seat_kinds.index("llm")]
        entries.append(
            {
                "game_id": game_id,
                "load_order": load_order,
                "states": last_state + 1,
                "turns": game.state.num_turns,
                "colors": colors,
                "seat_kinds": seat_kinds,
                "model": model,
                "model_label": display_model_name(model),
                "model_color": model_color,
                "seed": seed,
                "victory_points": {
                    color.value: get_actual_victory_points(game.state, color)
                    for color in game.state.colors
                },
                "winner": winner.value if winner else None,
            }
        )
    return entries, integrity


def deterministic_gzip(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=6, mtime=0)


def game_bundle(
    connection: sqlite3.Connection, game_id: str, expected_states: int
) -> tuple[bytes, int]:
    rows = connection.execute(
        """
        SELECT state_index, state
        FROM game_states
        WHERE uuid = ?
        ORDER BY state_index, id
        """,
        (game_id,),
    ).fetchall()
    indices = [row[0] for row in rows]
    if indices != list(range(expected_states)):
        raise ValueError(f"{game_id}: non-contiguous or duplicate state indices")
    raw = ("[" + ",".join(row[1] for row in rows) + "]").encode()
    return deterministic_gzip(raw), len(raw)


def insert_meta(html: str, name: str, content: str) -> str:
    marker = '<meta name="viewport" content="width=device-width, initial-scale=1">'
    if marker not in html:
        raise ValueError("expected viewport meta tag not found")
    tag = f'<meta name="{name}" content="{content}">'
    return html.replace(marker, f"{marker}\n{tag}", 1)


def replace_exact(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {description}, found {count}")
    return text.replace(old, new, 1)


def patch_catanatron_ui(ui_root: Path) -> None:
    configuration = ui_root / "src" / "configuration.ts"
    text = configuration.read_text()
    text = replace_exact(
        text,
        'import.meta.env.CTRON_API_URL || "http://localhost:5001"',
        "import.meta.env.CTRON_API_URL || window.location.origin",
        "Catanatron API default",
    )
    configuration.write_text(text)

    app = ui_root / "src" / "App.tsx"
    text = app.read_text()
    text = replace_exact(
        text,
        'BrowserRouter as Router',
        'HashRouter as Router',
        "Catanatron router import",
    )
    app.write_text(text)

    api_client = ui_root / "src" / "utils" / "apiClient.ts"
    text = api_client.read_text()
    start = text.index("export async function getState(")
    end = text.index("/** action=undefined", start)
    replacement = '''const replayCache = new Map<string, Promise<GameState[]>>();

function replayStates(gameId: string): Promise<GameState[]> {
  let pending = replayCache.get(gameId);
  if (!pending) {
    pending = axios
      .get<GameState[]>(
        `${API_URL}/api/replay-data/${encodeURIComponent(gameId)}`
      )
      .then((response) => response.data);
    replayCache.set(gameId, pending);
  }
  return pending;
}

export async function getState(
  gameId: string,
  stateIndex: StateIndex = "latest"
): Promise<GameState> {
  const states = await replayStates(gameId);
  if (stateIndex === "latest") return states[states.length - 1];
  const parsed = Number(stateIndex);
  const state = states[parsed];
  if (!state || state.state_index !== parsed) {
    throw new Error(`Replay state ${parsed} not found for ${gameId}`);
  }
  return state;
}

'''
    api_client.write_text(text[:start] + replacement + text[end:])


def build_viewer(source: Path, destination: Path) -> None:
    if not (source / "package.json").is_file():
        raise FileNotFoundError(f"Catanatron UI not found: {source}")
    if not (source / "node_modules").is_dir():
        raise FileNotFoundError(f"Catanatron UI dependencies not installed: {source}")

    with tempfile.TemporaryDirectory(prefix="catan-replay-ui-") as temp_dir:
        worktree = Path(temp_dir) / "ui"
        shutil.copytree(
            source,
            worktree,
            ignore=shutil.ignore_patterns("node_modules", "dist", ".git"),
        )
        os.symlink(source / "node_modules", worktree / "node_modules")
        patch_catanatron_ui(worktree)
        subprocess.run(
            ["yarn", "--ignore-engines", "build", "--base=/viewer/"],
            cwd=worktree,
            check=True,
        )
        shutil.copytree(worktree / "dist", destination)


def prepare_output(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not (output / EXPORT_MARKER).is_file():
            raise ValueError(f"refusing to replace unmarked directory: {output}")
        shutil.rmtree(output)
    output.mkdir()
    (output / EXPORT_MARKER).write_text("generated by scripts/export_replay_site.py\n")
    return output


def write_site_shell(site_dir: Path) -> None:
    """Refresh the source-controlled shell without rebuilding replay bundles."""
    site_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = site_dir / "replays"
    replay_dir.mkdir(exist_ok=True)
    index_html = insert_meta(
        INDEX_HTML.read_text(), "replay-read-only", "true"
    )
    watch_html = insert_meta(
        WATCH_HTML.read_text(), "replay-viewer-url", "/viewer"
    )
    (site_dir / "index.html").write_text(HOME_HTML.read_text())
    (replay_dir / "index.html").write_text(index_html)
    (replay_dir / "watch.html").write_text(watch_html)
    (site_dir / "replay.html").write_text(watch_html)
    (site_dir / "_redirects").write_text("/replay /replays/watch 301\n")


def export(args: argparse.Namespace) -> dict:
    database = args.database.resolve()
    metadata = args.metadata.resolve()
    output = prepare_output(args.output)
    site_dir = output / "site"
    r2_dir = output / "r2"
    games_dir = r2_dir / "games"
    site_dir.mkdir()
    games_dir.mkdir(parents=True)

    replay_models = load_replay_models(metadata)
    connection = sqlite3.connect(database)
    try:
        entries, integrity = catalog_rows(connection, replay_models)
        selected = []
        excluded = []
        for entry in entries:
            reason = publication_exclusion(
                entry,
                integrity[entry["game_id"]],
                include_diagnostics=args.include_diagnostics,
                include_models=args.include_model,
                exclude_models=args.exclude_model,
            )
            if reason is None:
                selected.append(entry)
            else:
                excluded.append({"game_id": entry["game_id"], "reason": reason})
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            for entry in selected[args.limit :]:
                excluded.append(
                    {"game_id": entry["game_id"], "reason": "export-limit"}
                )
            selected = selected[: args.limit]

        objects = []
        source_json_bytes = 0
        compressed_bytes = 0
        for position, entry in enumerate(selected, 1):
            game_id = entry["game_id"]
            if SAFE_GAME_ID.fullmatch(game_id) is None:
                raise ValueError(f"unsafe game id for object key: {game_id!r}")
            bundle, raw_bytes = game_bundle(connection, game_id, entry["states"])
            relative = Path("games") / f"{game_id}.json.gz"
            destination = r2_dir / relative
            destination.write_bytes(bundle)
            digest = hashlib.sha256(bundle).hexdigest()
            objects.append(
                {
                    "key": relative.as_posix(),
                    "bytes": len(bundle),
                    "sha256": digest,
                }
            )
            source_json_bytes += raw_bytes
            compressed_bytes += len(bundle)
            if position % 25 == 0 or position == len(selected):
                print(f"bundled {position}/{len(selected)} games", flush=True)
    finally:
        connection.close()

    catalog_payload = json.dumps(
        selected, ensure_ascii=False, separators=(",", ":")
    ).encode()
    catalog_path = r2_dir / "catalog.json"
    catalog_path.write_bytes(catalog_payload)
    objects.append(
        {
            "key": "catalog.json",
            "bytes": len(catalog_payload),
            "sha256": hashlib.sha256(catalog_payload).hexdigest(),
        }
    )

    write_site_shell(site_dir)
    if not args.skip_viewer_build:
        build_viewer(args.catanatron_ui.resolve(), site_dir / "viewer")

    exclusions = Counter(item["reason"] for item in excluded)
    report = {
        "version": 1,
        "database": str(database),
        "metadata": str(metadata),
        "included_games": len(selected),
        "excluded_games": len(excluded),
        "exclusions": dict(sorted(exclusions.items())),
        "source_json_bytes": source_json_bytes,
        "compressed_game_bytes": compressed_bytes,
        "r2_object_count": len(objects),
        "objects": objects,
        "excluded": excluded,
    }
    (output / "export-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: report[key] for key in (
        "included_games",
        "excluded_games",
        "exclusions",
        "source_json_bytes",
        "compressed_game_bytes",
        "r2_object_count",
    )}, indent=2))
    return report


if __name__ == "__main__":
    export(parse_args())
