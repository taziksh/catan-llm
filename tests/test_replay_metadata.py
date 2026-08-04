import json

import pytest

from catan_llm.replay_metadata import load_replay_models, save_replay_model
from scripts.replay_server import display_model_name


def test_replay_model_metadata_round_trip(tmp_path):
    path = tmp_path / "replays.models.json"
    save_replay_model(path, "game-a", "gpt-5.6-sol")
    save_replay_model(path, "game-b", "claude-fable-5")

    assert load_replay_models(path) == {
        "game-a": "gpt-5.6-sol",
        "game-b": "claude-fable-5",
    }
    assert json.loads(path.read_text())["game-a"] == "gpt-5.6-sol"


def test_replay_model_metadata_rejects_non_string_values(tmp_path):
    path = tmp_path / "replays.models.json"
    path.write_text('{"game-a": 5}')

    with pytest.raises(ValueError, match="string-to-string"):
        load_replay_models(path)


def test_replay_model_display_names():
    assert (
        display_model_name("openai/gpt-5.6-sol (thinking)")
        == "gpt-5.6-sol (thinking)"
    )
    assert display_model_name("claude-fable-5") == "claude-fable-5"
    assert display_model_name("provider/new-model") == "new-model"
