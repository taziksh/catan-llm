import argparse
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import run_grpo_trl
from build_reward_cache import scenario_seeds, scorer_fingerprint
from run_grpo_trl import (
    COMPLETIONS_PER_DEVICE,
    DEFAULT_ADAPTER,
    DEFAULT_K,
    DEFAULT_LR,
    DEFAULT_RUN_BASE,
    DEFAULT_SCENARIOS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    MAX_COMPLETION_TOKENS,
    PROMPTS_PER_STEP,
    build_rows,
    check_cache_index,
    completion_text,
    grpo_config_kwargs,
    init_fields,
    make_reward,
    parse_completion,
    preflight_cache_coverage,
    prepare_policy_base,
    run_label,
)
from test_reward_cache import record_llm_game

from catan_llm.parse import parse_move
from catan_llm.replay import replay_model_decisions
from catan_llm.schema import ActionType, DecisionRecord
from catan_llm.serialize import move_id

DATA = Path(__file__).parent.parent / "data"
FAIR_PAIRS = DATA / "dpo" / "r3pairs_fair" / "train.jsonl"
TRACES = DATA / "dagger_traces" / "r3pairs"


@pytest.fixture(scope="module")
def llm_trace(tmp_path_factory):
    return record_llm_game(tmp_path_factory.mktemp("grpo_trl_trace"))


def _reseed(trace: Path, out_dir: Path, seed, name=None) -> Path:
    lines = trace.read_text().splitlines()
    header = json.loads(lines[0])
    header["seed"] = seed
    out = out_dir / (name or trace.name)
    out.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")
    return out


def _args(**overrides):
    values = {
        "output": Path("out"),
        "run_base": DEFAULT_RUN_BASE,
        "adapter": DEFAULT_ADAPTER,
        "states": Path("data/dagger_traces/r3pairs"),
        "seed": DEFAULT_SEED,
        "learning_rate": DEFAULT_LR,
        "k": DEFAULT_K,
        "scenarios": DEFAULT_SCENARIOS,
        "temperature": DEFAULT_TEMPERATURE,
        "max_steps": None,
        "use_vllm": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_approved_defaults():
    assert DEFAULT_K == 8
    assert DEFAULT_SCENARIOS == 8
    assert DEFAULT_TEMPERATURE == 1.0
    assert DEFAULT_LR == 2e-6
    assert DEFAULT_SEED == 42
    assert PROMPTS_PER_STEP == 32
    assert MAX_COMPLETION_TOKENS == 32

    kwargs = grpo_config_kwargs(_args())
    assert kwargs["loss_type"] == "dr_grpo"
    assert kwargs["scale_rewards"] == "none"
    assert kwargs["beta"] == 0.0
    assert kwargs["num_iterations"] == 1
    assert kwargs["num_generations"] == 8
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["learning_rate"] == 2e-6
    assert kwargs["max_completion_length"] == 32
    assert kwargs["num_train_epochs"] == 1.0
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 100
    assert kwargs["save_total_limit"] == 2
    assert kwargs["log_completions"] is True
    assert "reward_weights" not in kwargs
    assert "max_steps" not in kwargs
    assert "use_vllm" not in kwargs


def test_batch_arithmetic_gives_32_prompts_per_step():
    kwargs = grpo_config_kwargs(_args())
    completions_per_step = (
        kwargs["per_device_train_batch_size"]
        * kwargs["gradient_accumulation_steps"]
    )
    assert completions_per_step == PROMPTS_PER_STEP * DEFAULT_K
    assert completions_per_step % kwargs["num_generations"] == 0
    assert completions_per_step // kwargs["num_generations"] == PROMPTS_PER_STEP
    assert kwargs["per_device_train_batch_size"] == COMPLETIONS_PER_DEVICE


def test_config_optional_knobs():
    assert grpo_config_kwargs(_args(max_steps=2))["max_steps"] == 2
    vllm = grpo_config_kwargs(_args(use_vllm=True))
    assert vllm["use_vllm"] is True
    assert vllm["vllm_mode"] == "colocate"


def test_run_label_and_init_fields(tmp_path):
    assert run_label("grpo-r3pairs-fair", DEFAULT_ADAPTER) == (
        "grpo-r3pairs-fair-dpo"
    )
    assert run_label("grpo-r3pairs-fair", None) == "grpo-r3pairs-fair-dagger2"
    assert grpo_config_kwargs(_args())["run_name"] == "grpo-r3pairs-fair-dpo"
    assert grpo_config_kwargs(_args(adapter=None))["run_name"] == (
        "grpo-r3pairs-fair-dagger2"
    )

    assert init_fields(None) == {
        "init": "dagger2",
        "adapter": None,
        "adapter_source_sha256": None,
    }
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"tensors")
    fields = init_fields(adapter_dir)
    assert fields["init"] == "dpo"
    assert fields["adapter"] == str(adapter_dir)
    assert len(fields["adapter_source_sha256"]) == 64


def test_prepare_policy_base_skips_merge_without_adapter(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        run_grpo_trl, "load_text_only_qwen35", lambda path: sentinel
    )
    monkeypatch.delitem(sys.modules, "peft", raising=False)
    assert prepare_policy_base("base", None) is sentinel


def test_prepare_policy_base_merges_adapter(monkeypatch):
    base = object()
    merged = object()
    calls = {}

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, path):
            calls["args"] = (model, path)
            return types.SimpleNamespace(merge_and_unload=lambda: merged)

    monkeypatch.setattr(
        run_grpo_trl, "load_text_only_qwen35", lambda path: base
    )
    monkeypatch.setitem(
        sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel)
    )
    assert prepare_policy_base("base", Path("dpo")) is merged
    assert calls["args"] == (base, "dpo")


LEGAL = [
    (ActionType.END_TURN, None),
    (ActionType.BUILD_SETTLEMENT, 5),
    (ActionType.BUILD_ROAD, (3, 4)),
    (ActionType.BUILD_CITY, 9),
]


def test_parse_completion_parity_with_parse_move():
    ids = [move_id(*action) for action in LEGAL]
    texts = [
        "answer: settlement:5",
        "I will build.\n**answer: road:3-4**",
        "answer: end_turn\nanswer: settlement:5",
        "answer: city:9.",
        "answer: 1",
        "answer: settlement:99",
        "no move at all",
        "",
    ]
    for text in texts:
        index = parse_move(text, LEGAL)
        expected = ids[index] if index is not None else None
        assert parse_completion(text, ids) == expected, text


def test_completion_text_formats():
    assert completion_text("answer: roll") == "answer: roll"
    assert completion_text(
        [{"role": "assistant", "content": "answer: roll"}]
    ) == "answer: roll"
    with pytest.raises(ValueError, match="assistant"):
        completion_text([{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError, match="assistant"):
        completion_text([])


def test_reward_shares_scores_and_zeroes_invalid():
    seeds = scenario_seeds(42, "game-a", 3, 2)
    cache = {
        ("game-a", 3, "settlement:5", seed): {"reward": reward}
        for seed, reward in zip(seeds, (0.4, 0.6))
    }
    reward = make_reward(cache, 42, 2)
    completions = [
        [{"role": "assistant", "content": "answer: settlement:5"}],
        [{"role": "assistant", "content": "answer: settlement:5"}],
        [{"role": "assistant", "content": "pass"}],
    ]
    assert reward(
        completions=completions,
        game_id=["game-a"] * 3,
        decision=[3] * 3,
        legal_moves=[["settlement:5", "end_turn"]] * 3,
    ) == pytest.approx([0.5, 0.5, 0.0])


def test_reward_raises_on_cache_miss():
    seeds = scenario_seeds(42, "game-a", 3, 2)
    cache = {("game-a", 3, "settlement:5", seeds[0]): {"reward": 0.4}}
    reward = make_reward(cache, 42, 2)
    completions = [[{"role": "assistant", "content": "answer: settlement:5"}]]
    with pytest.raises(RuntimeError, match="missing"):
        reward(
            completions=completions,
            game_id=["game-a"],
            decision=[3],
            legal_moves=[["settlement:5"]],
        )
    with pytest.raises(RuntimeError, match="missing"):
        make_reward(cache, 7, 1)(
            completions=completions,
            game_id=["game-a"],
            decision=[3],
            legal_moves=[["settlement:5"]],
        )


def test_reward_records_diagnostics_into_trainer_metrics():
    seeds = scenario_seeds(42, "game-a", 3, 1)
    cache = {
        ("game-a", 3, "settlement:5", seeds[0]): {"reward": 0.6},
        ("game-a", 3, "end_turn", seeds[0]): {"reward": 0.2},
    }
    metrics = run_grpo_trl.TrainerMetrics()
    metrics.trainer = types.SimpleNamespace(
        _metrics={"train": defaultdict(list)}
    )
    reward = make_reward(cache, 42, 1, metrics)
    completions = [
        [{"role": "assistant", "content": "answer: settlement:5"}],
        [{"role": "assistant", "content": "answer: end_turn"}],
        [{"role": "assistant", "content": "pass"}],
    ]
    rewards = reward(
        completions=completions,
        game_id=["game-a"] * 3,
        decision=[3] * 3,
        legal_moves=[["settlement:5", "end_turn"]] * 3,
    )
    assert rewards == pytest.approx([0.6, 0.2, 0.0])

    recorded = metrics.trainer._metrics["train"]
    assert recorded["diagnostics/reward_ceiling"] == [pytest.approx(0.6)]
    assert recorded["diagnostics/invalid_rate"] == [pytest.approx(1 / 3)]
    assert recorded["diagnostics/unique_moves"] == [pytest.approx(2.0)]

    unbound = run_grpo_trl.TrainerMetrics()
    unbound.record("diagnostics/invalid_rate", 0.0)


def _index(**overrides):
    index = {
        "seed": 42,
        "scenarios": 8,
        "trajectories": "data/dagger_traces/r3pairs",
        "val_every": 10,
        "hero_policy": "alpha_beta",
        "opponent_policy": "value_function",
        **scorer_fingerprint(),
    }
    index.update(overrides)
    return index


def test_check_cache_index_refuses_mismatches():
    check_cache_index(_index(), _args())
    check_cache_index(_index(scenarios=16), _args())
    bad = {
        "seed": 7,
        "trajectories": "elsewhere",
        "val_every": 5,
        "hero_policy": "value_function",
        "opponent_policy": "alpha_beta",
        "scorer_sha256": "0" * 64,
        "catanatron_version": "0.0.0",
    }
    for key, value in bad.items():
        with pytest.raises(RuntimeError, match=key):
            check_cache_index(_index(**{key: value}), _args())
    with pytest.raises(RuntimeError, match="scenarios"):
        check_cache_index(_index(scenarios=4), _args())


def test_preflight_cache_coverage():
    rows = [{"game_id": "g", "decision": 1, "legal_moves": ["end_turn", "roll"]}]
    seeds = scenario_seeds(42, "g", 1, 2)
    cache = {
        ("g", 1, move, seed): {"reward": 0.1}
        for move in ("end_turn", "roll")
        for seed in seeds
    }
    assert preflight_cache_coverage(cache, rows, 42, 2) == 4

    del cache[("g", 1, "roll", seeds[1])]
    with pytest.raises(RuntimeError, match="missing 1 of 4"):
        preflight_cache_coverage(cache, rows, 42, 2)


def test_build_rows_is_deterministic_and_complete(llm_trace):
    first = build_rows([llm_trace], 42)
    second = build_rows([llm_trace], 42)
    assert first == second
    assert len(first) > 3

    replayed = list(replay_model_decisions(llm_trace))
    assert {(row["game_id"], row["decision"]) for row in first} == {
        (item.game_record.game_id, item.decision.i) for item in replayed
    }
    assert [row["decision"] for row in first] != [
        item.decision.i for item in replayed
    ]
    by_decision = {item.decision.i: item.decision for item in replayed}
    for row in first:
        decision = by_decision[row["decision"]]
        assert row["legal_moves"] == [
            move_id(*action) for action in decision.legal_actions
        ]
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"

    assert build_rows([llm_trace], 42, max_states=3) == first[:3]


def test_build_rows_excludes_val_games(llm_trace, tmp_path):
    val_trace = _reseed(llm_trace, tmp_path, 10_020, name="val-game.jsonl")
    assert build_rows([val_trace], 42) == []

    rows = build_rows([llm_trace, val_trace], 42)
    assert rows == build_rows([llm_trace], 42)


def test_build_rows_refuses_eval_seeds(llm_trace, tmp_path):
    for bad_seed in (42, None):
        doctored = _reseed(
            llm_trace, tmp_path, bad_seed, name=f"seed_{bad_seed}.jsonl"
        )
        with pytest.raises(ValueError, match="eval"):
            build_rows([doctored], 42)


@pytest.mark.skipif(
    not (FAIR_PAIRS.exists() and TRACES.exists()),
    reason="repo DPO data not present",
)
def test_prompt_parity_with_dpo_fair_data():
    with FAIR_PAIRS.open() as handle:
        pair = json.loads(handle.readline())
    trace = TRACES / f"{pair['game_id']}.jsonl"
    row = next(
        row
        for row in build_rows([trace], 42)
        if row["decision"] == pair["decision"]
    )
    assert row["prompt"] == pair["prompt"]

    lines = trace.read_text().splitlines()
    decision = next(
        DecisionRecord.model_validate_json(line)
        for line in lines[1:]
        if json.loads(line)["i"] == pair["decision"]
    )
    chosen = move_id(*decision.legal_actions[decision.chosen_action])
    assert chosen in row["legal_moves"]


def test_validate_grpo_api_against_installed_trl():
    pytest.importorskip("trl")
    run_grpo_trl.validate_grpo_api(grpo_config_kwargs(_args(use_vllm=True)))
