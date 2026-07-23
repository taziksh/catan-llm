"""Stateless chat harness: every segment sees only the current user turn.

Decision prompts are self-contained, so `resume()` launches on the new turn
alone instead of replaying the accumulated conversation (the base default).
Each decision becomes its own branch in the trace graph — one training sample
per decision — and per-game input cost stays linear in decisions.

A package of its own (harness id "catan_v1_harness"): plugin ids resolve while
catan_v1's env config defaults are built, so the harness cannot live inside
that package without a circular import.
"""

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.null import NullHarness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages

__all__ = ["StatelessHarness"]


class StatelessHarness(NullHarness):
    async def resume(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        messages: Messages,
    ) -> ProgramResult:
        return await self.launch(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data.model_copy(update={"prompt": messages}),
        )
