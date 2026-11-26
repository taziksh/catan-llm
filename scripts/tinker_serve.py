"""Serves a Tinker checkpoint or base model behind a local OpenAI-compatible endpoint.

Two prompt modes: a tinker_cookbook renderer name (pinned chat format for our
finetunes), or ``--renderer hf`` to use the tokenizer's own chat template with
its defaults (thinking on for GLM / Kimi). In hf mode a ``<think>...</think>``
block is returned as ``reasoning_content`` and only the text after it as
``content``, matching how hosted thinking models appear to the harness.
"""

import argparse
import asyncio
import json
import logging
import time

import tinker
from aiohttp import web
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

CONTEXT_LIMIT = 16384
REQUEST_TIMEOUT = 1200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer id if it differs from --base-model")
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--renderer", required=True, help="cookbook renderer name, or 'hf'")
    parser.add_argument("--stop-tokens", default="", help="comma-separated stop token strings (hf mode)")
    parser.add_argument(
        "--template-kwargs",
        default="{}",
        help='JSON kwargs for apply_chat_template (hf mode), e.g. {"reasoning_effort": "low"} or {"thinking": false}',
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    class Session:
        """A Tinker service client, its sampling client, and the requests in flight on it."""

        def __init__(self):
            self.service = tinker.ServiceClient()
            if args.model_path:
                self.sampler = self.service.create_sampling_client(model_path=args.model_path)
            else:
                self.sampler = self.service.create_sampling_client(base_model=args.base_model)
            self.dead = False
            self.tasks = set()

        def kill(self, reason):
            """Cancels in-flight requests and stops the heartbeat so the server drops the session."""
            if self.dead:
                return
            self.dead = True
            print(f"{time.strftime('%H:%M:%S')} session dead ({reason}); cancelling {len(self.tasks)} in-flight", flush=True)
            for task in self.tasks:
                task.cancel()
            self.service._get_session_holder().close()

    state = {"session": None}
    lock = asyncio.Lock()

    async def get_session():
        async with lock:
            if state["session"] is None or state["session"].dead:
                state["session"] = await asyncio.to_thread(Session)
            return state["session"]

    class HeartbeatWatch(logging.Handler):
        """Kills the current session when Tinker logs a failed session heartbeat."""

        def __init__(self, loop):
            super().__init__(level=logging.INFO)
            self.loop = loop

        def emit(self, record):
            message = record.getMessage()
            if ("heartbeat failed" in message or "has finished" in message) and state["session"]:
                self.loop.call_soon_threadsafe(state["session"].kill, message[:80])

    tokenizer = get_tokenizer(args.tokenizer or args.base_model)
    if args.renderer == "hf":
        stop_strings = [s for s in args.stop_tokens.split(",") if s]
        stop_strings.append(tokenizer.eos_token)
        stop_ids = [tokenizer.convert_tokens_to_ids(s) for s in stop_strings]

        template_kwargs = json.loads(args.template_kwargs)

        def build_prompt(messages):
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, **template_kwargs
            )
            return types.ModelInput.from_ints(list(ids["input_ids"]))
    else:
        renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.base_model)
        stop_ids = [s for s in renderer.get_stop_sequences() if isinstance(s, int)]
        stop_strings = [tokenizer.decode([s]) for s in stop_ids]
        stop_strings += [s for s in renderer.get_stop_sequences() if isinstance(s, str)]
        build_prompt = renderer.build_generation_prompt

    async def chat_completions(request):
        body = await request.json()
        prompt = build_prompt(body["messages"])
        prompt_tokens = len(prompt.to_ints())
        max_tokens = min(body.get("max_tokens") or CONTEXT_LIMIT, CONTEXT_LIMIT - prompt_tokens)
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=body.get("temperature", 0.0),
            top_p=body.get("top_p", 1.0),
            stop=stop_ids,
        )
        session = await get_session()
        task = asyncio.ensure_future(
            session.sampler.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
        )
        session.tasks.add(task)
        try:
            result = await asyncio.wait_for(task, timeout=REQUEST_TIMEOUT)
        except asyncio.CancelledError:
            return web.json_response({"error": {"message": "session replaced"}}, status=503)
        except asyncio.TimeoutError:
            print(f"{time.strftime('%H:%M:%S')} sample timed out after {REQUEST_TIMEOUT}s", flush=True)
            return web.json_response({"error": {"message": "sample timed out"}}, status=503)
        except Exception as e:
            session.kill(f"{type(e).__name__}: {str(e)[:120]}")
            return web.json_response({"error": {"message": f"{type(e).__name__}: {e}"}}, status=503)
        finally:
            session.tasks.discard(task)
        sequence = result.sequences[0]
        text = tokenizer.decode(sequence.tokens)
        for stop in stop_strings:
            text = text.split(stop)[0]
        reasoning, sep, content = text.rpartition("</think>")
        if not sep:
            reasoning, content = "", text
        reasoning = reasoning.removeprefix("<think>").strip()
        message = {"role": "assistant", "content": content.strip()}
        if reasoning:
            message["reasoning_content"] = reasoning
        finish = "length" if len(sequence.tokens) >= max_tokens else "stop"
        print(
            f"{time.strftime('%H:%M:%S')} prompt={prompt_tokens} completion={len(sequence.tokens)} "
            f"{finish} :: {content.strip().splitlines()[-1][:80] if content.strip() else '<empty>'}",
            flush=True,
        )
        return web.json_response(
            {
                "id": "tinker-shim",
                "created": int(time.time()),
                "object": "chat.completion",
                "model": args.served_name,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(sequence.tokens),
                    "total_tokens": prompt_tokens + len(sequence.tokens),
                },
            }
        )

    async def models(request):
        return web.json_response(
            {"object": "list", "data": [{"id": args.served_name, "object": "model"}]}
        )

    async def watch_heartbeat(app):
        holder_log = logging.getLogger("tinker.lib.internal_client_holder")
        holder_log.setLevel(logging.INFO)
        holder_log.addHandler(HeartbeatWatch(asyncio.get_running_loop()))
        await get_session()

    app = web.Application()
    app.on_startup.append(watch_heartbeat)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", models)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
