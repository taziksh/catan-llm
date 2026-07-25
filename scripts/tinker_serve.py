"""Serves a Tinker checkpoint behind a local OpenAI-compatible endpoint."""

import argparse
import time

import tinker
from aiohttp import web
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    service = tinker.ServiceClient()
    if args.model_path:
        sampler = service.create_sampling_client(model_path=args.model_path)
    else:
        sampler = service.create_sampling_client(base_model=args.base_model)
    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(
        args.renderer, tokenizer, model_name=args.base_model
    )

    async def chat_completions(request):
        body = await request.json()
        prompt = renderer.build_generation_prompt(body["messages"])
        params = types.SamplingParams(
            max_tokens=body.get("max_tokens", 2000),
            temperature=body.get("temperature", 0.0),
            top_p=body.get("top_p", 1.0),
        )
        result = await sampler.sample_async(
            prompt=prompt, sampling_params=params, num_samples=1
        )
        sequence = result.sequences[0]
        text = tokenizer.decode(sequence.tokens)
        for stop in renderer.get_stop_sequences():
            if isinstance(stop, int):
                stop = tokenizer.decode([stop])
            text = text.split(stop)[0]
        return web.json_response(
            {
                "id": "tinker-shim",
                "created": int(time.time()),
                "object": "chat.completion",
                "model": args.served_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.to_ints()),
                    "completion_tokens": len(sequence.tokens),
                    "total_tokens": len(prompt.to_ints()) + len(sequence.tokens),
                },
            }
        )

    async def models(request):
        return web.json_response(
            {"object": "list", "data": [{"id": args.served_name, "object": "model"}]}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", models)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
