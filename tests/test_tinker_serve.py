import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from tinker_serve import split_message


def test_typed_parts_split_into_reasoning_and_answer():
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "count resources"},
            {"type": "thinking", "thinking": "settlement wins"},
            {"type": "text", "text": "answer: 9"},
        ],
    }
    assert split_message(message) == ("count resources\nsettlement wins", "answer: 9")


def test_parts_other_than_thinking_and_text_are_dropped():
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan"},
            {"type": "tool_call", "name": "noop"},
            {"type": "text", "text": "answer: 2"},
        ],
    }
    assert split_message(message) == ("plan", "answer: 2")


def test_string_with_think_block_is_split():
    message = {"role": "assistant", "content": "<think>weigh options</think>answer: 3"}
    assert split_message(message) == ("weigh options", "answer: 3")


def test_string_without_think_block_is_all_answer():
    message = {"role": "assistant", "content": "answer: 0"}
    assert split_message(message) == ("", "answer: 0")
