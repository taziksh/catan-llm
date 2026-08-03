"""Package the canonical catan_llm source with the Prime environment."""

import shutil
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ENV_ROOT = Path(__file__).resolve().parent
CANONICAL_PACKAGE = ENV_ROOT.parents[1] / "catan_llm"
PACKAGED_COPY = ENV_ROOT / "catan_llm"


def sync_catan_llm() -> None:
    if not CANONICAL_PACKAGE.is_dir():
        # A pulled Prime source archive already contains the packaged copy.
        if PACKAGED_COPY.is_dir():
            return
        raise RuntimeError(f"missing canonical package: {CANONICAL_PACKAGE}")

    if PACKAGED_COPY.is_symlink():
        PACKAGED_COPY.unlink()
    elif PACKAGED_COPY.exists():
        shutil.rmtree(PACKAGED_COPY)

    shutil.copytree(
        CANONICAL_PACKAGE,
        PACKAGED_COPY,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        sync_catan_llm()
