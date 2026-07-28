"""Parses a model reply into a chosen option index."""

import re

from catan_llm.serialize import move_id

ANSWER_RE = re.compile(r"answer\s*:\s*(\d+)", re.IGNORECASE)
MOVE_RE = re.compile(r"answer\s*:\s*(\S+)", re.IGNORECASE)


def parse_answer(text: str, n_options: int):
    """Returns the chosen index, or None if unparseable or out of range."""
    matches = ANSWER_RE.findall(text)
    if matches:
        index = int(matches[-1])
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        last = lines[-1].rstrip(".") if lines else ""
        if not last.isdigit():
            return None
        index = int(last)
    return index if 0 <= index < n_options else None


def parse_move(text: str, legal_actions):
    """Returns the chosen index, or None if unparseable or not a legal move."""
    ids = {move_id(t, p): i for i, (t, p) in enumerate(legal_actions)}
    for token in reversed(MOVE_RE.findall(text)):
        token = token.strip("*`.,;:\"'")
        if token in ids:
            return ids[token]
    return None
