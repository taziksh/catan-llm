"""Parses a model reply into a chosen option index."""

import re

ANSWER_RE = re.compile(r"answer\s*:\s*(\d+)", re.IGNORECASE)


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
