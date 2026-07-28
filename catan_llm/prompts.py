"""Prompt versions, append-only. Never rename a version; only the two
pointer lines at the bottom change.

v1: unbounded reasoning invitation; serializer had no edges line and no
    trading rule.
v2: "reason briefly" bound; serializer adds the board edge list and the
    bank/port-only trading sentence (3459636).
v3: options named by move id instead of a per-state index; serializer drops
    the move name from the option note.
"""

SYSTEM_PROMPT_V1 = (
    "You are playing Settlers of Catan. Each turn you receive the full game "
    "state and a numbered list of your legal options. You may reason first; "
    'only your final "answer: <option number>" line counts.'
)

SYSTEM_PROMPT_V2 = (
    "You are playing Settlers of Catan. Each turn you receive the full game "
    "state and a numbered list of your legal options. You may reason briefly "
    'first. Only your final "answer: <option number>" line counts.'
)

SYSTEM_PROMPT_V3 = (
    "You are playing Settlers of Catan. Each turn you receive the full game "
    "state and a list of your legal moves. You may reason briefly first. "
    'Only your final "answer: <move id>" line counts.'
)

PROMPT_VERSION = "v3"
SYSTEM_PROMPT = SYSTEM_PROMPT_V3
