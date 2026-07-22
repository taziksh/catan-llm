"""Helpers for reproducible runs."""

import os
import sys


def require_fixed_hashseed():
    """Re-execs with PYTHONHASHSEED=0; it must be set before interpreter start."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
