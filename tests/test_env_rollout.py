"""End-to-end catan-v1 rollouts against a scripted local model server.

Runs the real verifiers stack (serving, interception, harness program,
runtime). The "model" is an HTTP server that always picks the first option.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import verifiers as vf
from catan_v1 import load_environment
from catan_v1.legacy import DEFAULT_MAX_TURNS
from catan_v1.taskset import CatanEnv, CatanEnvConfig, CatanTasksetConfig
from pydantic import ValidationError
from verifiers.v1 import EvalClientConfig
from verifiers.v1.clients import ModelContext, resolve_client
from verifiers.v1.types import SamplingConfig

from catan_llm.determinism import fixed_hashseed

requires_fixed_hashseed = pytest.mark.skipif(
    not fixed_hashseed(),
    reason="needs PYTHONHASHSEED=0 (see catan_llm.determinism)",
)


def _first_option(messages):
    prompt = messages[-1]["content"]
    line = prompt.split("YOUR OPTIONS\n", 1)[1].splitlines()[0]
    return line.split(" (")[0]


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
                        "message": {
                            "role": "assistant",
                            "content": f"answer: {_first_option(body['messages'])}",
                        },
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


@requires_fixed_hashseed
def test_hosted_loader_completes_full_stateless_game(scripted_model, monkeypatch):
    """Exercise the exact load_environment API used by Prime hosted eval/training."""
    monkeypatch.setenv("FAKE_API_KEY", "fake")
    env = load_environment(num_seeds=1)
    port = scripted_model.server_address[1]
    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=f"http://127.0.0.1:{port}/v1",
        api_key_var="FAKE_API_KEY",
    )
    output = asyncio.run(
        env.run_rollout(
            env.get_dataset()[0],
            client=client,
            model="scripted",
            sampling_args={"max_tokens": 32, "temperature": 0.0},
        )
    )

    assert output.get("error") is None
    assert output["is_completed"]
    assert output["stop_condition"] == "has_final_env_response"
    assert output["reward"] >= 0.0
    assert output["decisions"] > 0
    assert output["invalid_rate"] == 0.0
    assert output["game_length"] > 0
    assert output["info"]["catan"]["seed"] == 0
    assert output["info"]["catan"]["env_version"]

    # The legacy hosted adapter must preserve the existing stateless prompt
    # contract: one system message plus the current board, never full history.
    assert scripted_model.calls
    for messages in scripted_model.calls:
        assert [message["role"] for message in messages] == ["system", "user"]


def test_hosted_loader_keeps_train_and_eval_seeds_disjoint():
    eval_env = load_environment(seed_start=0, num_seeds=3)
    train_env = load_environment(seed_start=10_000, num_seeds=3)

    assert eval_env.max_turns == DEFAULT_MAX_TURNS == 500
    assert [row["info"]["seed"] for row in eval_env.get_dataset()] == [0, 1, 2]
    assert [row["info"]["seed"] for row in train_env.get_dataset()] == [
        10_000,
        10_001,
        10_002,
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed_start": -1}, "non-negative"),
        (
            {"seed_start": 9_999, "num_seeds": 2},
            "crosses into training seeds",
        ),
        (
            {"seed_start": 0, "num_seeds": 10_001},
            "crosses into training seeds",
        ),
        ({"num_seeds": 0}, "at least 1"),
        ({"invalid_retries": -1}, "non-negative"),
        ({"vp_coef": -0.1}, "non-negative"),
        ({"max_turns": 0}, "at least 1"),
        ({"timeout_seconds": 0}, "positive or None"),
        ({"seats": "agent,value_function,value_function"}, "exactly four"),
        ({"seats": "agent,agent,value_function,value_function"}, "exactly one"),
        (
            {"seats": "agent,value_function,value_function,not_a_bot"},
            "unknown seat kind",
        ),
    ],
)
def test_hosted_loader_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        load_environment(**({"num_seeds": 1} | kwargs))


def test_hosted_loader_requires_fixed_hashseed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        load_environment(num_seeds=1)


def test_v1_config_rejects_removed_dagger_options():
    with pytest.raises(ValidationError, match="dagger_beta"):
        CatanEnvConfig(dagger_beta=0.1)


def test_v1_config_validates_lineups_and_seed_partition():
    config = CatanEnvConfig(
        seats="agent, value_function, value_function, value_function"
    )
    assert config.seats == "agent,value_function,value_function,value_function"

    with pytest.raises(ValidationError, match="at least one agent"):
        CatanEnvConfig(seats="random,random,random,random")
    assert CatanTasksetConfig(seed_start=5).seed_start == 5
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        CatanTasksetConfig(seed_start=-1)
