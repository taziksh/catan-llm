"""Persistent model labels for replay trajectories."""

import json
from pathlib import Path


def load_replay_models(path: Path) -> dict[str, str]:
    """Return game-id to model-name mappings from a small JSON sidecar."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(
        isinstance(game_id, str) and isinstance(model, str)
        for game_id, model in data.items()
    ):
        raise ValueError(f"{path}: expected a string-to-string JSON object")
    return data


def save_replay_model(path: Path, game_id: str, model: str) -> None:
    """Atomically add or update one replay's model identity."""
    models = load_replay_models(path)
    models[game_id] = model
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
