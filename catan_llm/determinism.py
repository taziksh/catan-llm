"""Helpers for reproducible runs."""

import os
import sys

EVAL_SEED_LIMIT = 10_000
"""Seeds below this are reserved for evaluation; training seeds start here."""


def fixed_hashseed():
    """Whether the interpreter started with PYTHONHASHSEED=0."""
    return os.environ.get("PYTHONHASHSEED") == "0" and not sys.flags.hash_randomization


def require_fixed_hashseed():
    """Re-execs with PYTHONHASHSEED=0; it must be set before interpreter start."""
    if not fixed_hashseed():
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def check_fixed_hashseed():
    """Raises unless the interpreter started with PYTHONHASHSEED=0.

    Bot tie-breaks and playable_actions order depend on hash randomization, so
    without it the same seed plays out differently across processes. For hosts
    that cannot re-exec (e.g. the eval harness).
    """
    if not fixed_hashseed():
        raise RuntimeError(
            "games are only seed-deterministic with PYTHONHASHSEED=0; "
            "set it before launching this process"
        )
