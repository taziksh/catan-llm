"""End-to-end catan-v1 rollouts against a scripted local model server.

Runs the real verifiers stack (serving, interception, harness program,
runtime); the "model" is an HTTP server that always answers "answer: 0".
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from catan_llm.determinism import fixed_hashseed

requires_fixed_hashseed = pytest.mark.skipif(
    not fixed_hashseed(),
    reason="needs PYTHONHASHSEED=0 (see catan_llm.determinism)",
)


class _ScriptedModel(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append(body["messages"])
        payload = json.dumps(
            {
                "id": "chatcmpl-0",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer: 0"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def scripted_model():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedModel)
    server.calls = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


@requires_fixed_hashseed
def test_two_rollouts_stateless_and_distinct_trajectories(
    scripted_model, tmp_path, monkeypatch
):
    from verifiers.v1 import EvalClientConfig
    from verifiers.v1.clients import ModelContext, resolve_client
    from verifiers.v1.types import SamplingConfig

    from catan_v1.taskset import CatanEnv, CatanEnvConfig

    monkeypatch.setenv("FAKE_API_KEY", "fake")
    traj_dir = tmp_path / "traj"
    env = CatanEnv(
        CatanEnvConfig(taskset={"id": "catan_v1"}, trajectory_dir=str(traj_dir))
    )
    port = scripted_model.server_address[1]
    client = resolve_client(
        EvalClientConfig(
            base_url=f"http://127.0.0.1:{port}/v1", api_key_var="FAKE_API_KEY"
        )
    )
    ctx = ModelContext(client=client, model="scripted", sampling=SamplingConfig())

    async def run():
        async with env.serving():
            (task,) = env.taskset.select(1, False)
            slots = env.slots(task, n=2)
            return await asyncio.gather(*(env.run_slot(s, ctx) for s in slots))

    episodes = asyncio.run(run())

    for episode in episodes:
        assert episode.ok, [str(e) for e in episode.errors]
        (trace,) = episode.traces
        assert "reward_win" in trace.rewards and "reward_vp" in trace.rewards
        assert trace.metrics["invalid_rate"] == 0.0
        assert trace.metrics["decisions"] > 0

    # Stateless: every model call carries the system prompt and one user turn.
    assert scripted_model.calls
    for messages in scripted_model.calls:
        assert [m["role"] for m in messages] == ["system", "user"]

    # Two rollouts of one seed write two distinct trajectory files. The
    # lineup prefix follows turn order (seed-shuffled), so only its parts
    # are fixed.
    files = sorted(traj_dir.glob("*.jsonl"))
    assert len(files) == 2 and files[0].name != files[1].name
    for path in files:
        lineup, seed_and_suffix = path.stem.rsplit("_s", 1)
        assert sorted(lineup.split("-")) == ["llm"] + ["value_function"] * 3
        assert seed_and_suffix.startswith("0_")

    # Concurrent rollouts of one seed with a deterministic model must be the
    # same game (the engine isolates each game's rng stream); only the
    # per-rollout game_id differs.
    def records(path):
        out = []
        for line in path.read_text().splitlines():
            record = json.loads(line)
            record.pop("game_id")
            out.append(record)
        return out

    assert records(files[0]) == records(files[1])
