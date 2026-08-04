import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_dpo import (
    DEFAULT_BETA,
    TEXT_KEY_MAPPING,
    deterministic_subset,
    render_pair,
    subset_identity,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return (
            "<|im_start|>system\nsystem<|im_end|>\n"
            "<|im_start|>user\nstate<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def _row(index):
    return {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "state"},
        ],
        "chosen": [{"role": "assistant", "content": f"answer: best:{index}"}],
        "rejected": [
            {"role": "assistant", "content": f"answer: played:{index}"}
        ],
        "game_id": f"game-{index // 2}",
        "decision": index,
    }


def test_controlled_defaults_and_text_key_mapping():
    assert DEFAULT_BETA == 0.1
    assert TEXT_KEY_MAPPING == {r"^model\.language_model\.": "model."}


def test_render_pair_preserves_direction_and_disables_thinking():
    rendered = render_pair(_row(3), FakeTokenizer())

    assert rendered["chosen"] == "answer: best:3"
    assert rendered["rejected"] == "answer: played:3"
    assert rendered["prompt"].endswith("<think>\n\n</think>\n\n")


def test_render_pair_rejects_identical_completions():
    row = _row(3)
    row["rejected"] = row["chosen"]

    with pytest.raises(ValueError, match="identical"):
        render_pair(row, FakeTokenizer())


def test_canary_subset_is_reproducible_and_recorded():
    rows = [_row(index) for index in range(20)]
    first = deterministic_subset(rows, 7, 42)
    second = deterministic_subset(rows, 7, 42)

    assert first == second
    assert len(first) == 7
    assert subset_identity(first) == subset_identity(second)
    assert first != rows[:7]
