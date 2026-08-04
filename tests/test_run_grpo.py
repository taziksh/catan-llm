import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_grpo import (
    DEFAULT_BETA_KL,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_K,
    DEFAULT_LR,
    DEFAULT_SCENARIOS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    GRAD_CLIP_NORM,
    MAX_NEW_TOKENS,
    STATES_PER_STEP,
    advantages,
    completion_token_logprobs,
    completion_attention_mask,
    degenerate,
    group_rewards,
    grpo_loss,
    iter_states,
    prompt_messages,
    ref_sequence_logprobs,
    render_prompt,
    save_checkpoint,
    sequence_logprobs,
    write_manifest,
)
from test_reward_cache import record_llm_game

from catan_llm.parse import parse_move
from catan_llm.replay import replay_model_decisions
from catan_llm.schema import ActionType, DecisionRecord, GameRecord

DATA = Path(__file__).parent.parent / "data"
FAIR_PAIRS = DATA / "dpo" / "r3pairs_fair" / "train.jsonl"
TRACES = DATA / "dagger_traces" / "r3pairs"


@pytest.fixture(scope="module")
def llm_trace(tmp_path_factory):
    return record_llm_game(tmp_path_factory.mktemp("grpo_trace"))


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).eval()


def test_approved_defaults():
    assert DEFAULT_K == 8
    assert DEFAULT_SCENARIOS == 8
    assert DEFAULT_TEMPERATURE == 1.0
    assert DEFAULT_BETA_KL == 5e-3
    assert DEFAULT_LR == 2e-6
    assert DEFAULT_SEED == 42
    assert STATES_PER_STEP == 32
    assert MAX_NEW_TOKENS == 32
    assert GRAD_CLIP_NORM == 1.0
    assert DEFAULT_CHECKPOINT_EVERY == 50


class FakeTokenizer:
    def __init__(self, suffix="<think>\n\n</think>\n\n"):
        self.suffix = suffix

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return "prompt" + self.suffix


def _messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "state"},
    ]


def test_render_prompt_disables_thinking():
    assert render_prompt(_messages(), FakeTokenizer()).endswith(
        "<think>\n\n</think>\n\n"
    )


def test_render_prompt_refuses_thinking_template():
    with pytest.raises(ValueError, match="suffix"):
        render_prompt(_messages(), FakeTokenizer(suffix="<think>\n"))


@pytest.mark.skipif(
    not (FAIR_PAIRS.exists() and TRACES.exists()),
    reason="repo DPO data not present",
)
def test_prompt_parity_with_dpo_fair_data():
    with FAIR_PAIRS.open() as handle:
        row = json.loads(handle.readline())
    lines = (TRACES / f"{row['game_id']}.jsonl").read_text().splitlines()
    game = GameRecord.model_validate_json(lines[0])
    decision = next(
        DecisionRecord.model_validate_json(line)
        for line in lines[1:]
        if json.loads(line)["i"] == row["decision"]
    )
    assert prompt_messages(game, decision) == row["prompt"]


def test_group_rewards_share_scores_and_zero_invalid():
    legal = [(ActionType.END_TURN, None), (ActionType.BUILD_SETTLEMENT, 5)]
    parses = [
        parse_move("answer: settlement:5", legal),
        parse_move("no move here", legal),
    ]
    assert parses == [1, None]
    assert group_rewards(parses, {1: 0.9}) == [0.9, 0.0]
    assert group_rewards([0, 0, 1, None], {0: 0.5, 1: 0.2}) == [
        0.5,
        0.5,
        0.2,
        0.0,
    ]


def test_advantages_use_group_mean_baseline():
    assert advantages([0.5, 0.5, 0.2, 0.0]) == pytest.approx([0.2, 0.2, -0.1, -0.3])
    assert sum(advantages([1.1, 0.3, 0.0, 0.7])) == pytest.approx(0.0)


def test_degenerate_groups_are_skipped_before_scoring():
    assert degenerate([2, 2, 2, 2])
    assert degenerate([None, None])
    assert not degenerate([2, None])
    assert not degenerate([1, 2, 2, 2])


def test_sequence_logprobs_match_hand_computation():
    model = tiny_model()
    sequences = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])

    with torch.no_grad():
        logits = model(input_ids=sequences, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    expected = torch.stack(
        [
            logprobs[0, 2, sequences[0, 3]] + logprobs[0, 3, sequences[0, 4]],
            logprobs[1, 2, sequences[1, 3]],
        ]
    )

    with torch.no_grad():
        result = sequence_logprobs(model, sequences, attention_mask, 3)
    assert torch.allclose(result, expected, atol=1e-5)

    with_grad = sequence_logprobs(model, sequences, attention_mask, 3)
    assert with_grad.requires_grad


def test_completion_token_logprobs_expose_only_completion_positions():
    model = tiny_model()
    sequences = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])

    with torch.no_grad():
        token_logprobs, mask = completion_token_logprobs(
            model, sequences, attention_mask, 3
        )
        summed = sequence_logprobs(model, sequences, attention_mask, 3)

    assert mask.tolist() == [[0, 0, 1, 1], [0, 0, 1, 0]]
    assert torch.allclose((token_logprobs * mask).sum(dim=-1), summed)


class AdapterSwitchRecorder:
    def __init__(self, model):
        self._model = model
        self.calls = []
        self.active = "policy"

    def set_adapter(self, name):
        self.calls.append(name)
        self.active = name

    def __call__(self, **kwargs):
        return self._model(**kwargs)


def test_save_checkpoint_writes_policy_adapter(tmp_path):
    peft = pytest.importorskip("peft")
    LoraConfig, get_peft_model = peft.LoraConfig, peft.get_peft_model

    model = get_peft_model(
        tiny_model(),
        LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"]),
        adapter_name="policy",
    )
    path = save_checkpoint(model, tmp_path, 50)
    assert path == tmp_path / "step_00050"
    assert (path / "policy" / "adapter_model.safetensors").exists()


def test_ref_logprobs_switch_adapters_without_grad():
    model = tiny_model()
    recorder = AdapterSwitchRecorder(model)
    sequences = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(sequences)

    ref = ref_sequence_logprobs(recorder, sequences, attention_mask, 3)

    assert recorder.calls == ["ref", "policy"]
    assert recorder.active == "policy"
    assert not ref.requires_grad
    with torch.no_grad():
        direct = sequence_logprobs(model, sequences, attention_mask, 3)
    assert torch.allclose(ref, direct)


def test_ref_logprobs_restore_policy_adapter_on_error():
    def boom(**kwargs):
        raise RuntimeError("boom")

    recorder = AdapterSwitchRecorder(boom)
    sequences = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError, match="boom"):
        ref_sequence_logprobs(recorder, sequences, torch.ones_like(sequences), 2)
    assert recorder.calls == ["ref", "policy"]


def test_completion_attention_mask_stops_after_first_stop_token():
    sequences = torch.tensor([[4, 5, 6, 9, 9], [4, 5, 6, 8, 8], [4, 5, 7, 6, 6]])
    mask = completion_attention_mask(sequences, 2, stop_ids={9, 7})
    assert mask.tolist() == [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ]


class GenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


def test_stop_token_ids_include_generation_config_and_pad():
    from run_grpo import stop_token_ids

    class Model:
        generation_config = GenerationConfig([9, 7])

    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 3

    assert stop_token_ids(Model(), Tokenizer()) == {9, 7, 3}

    class IntConfigModel:
        generation_config = GenerationConfig(9)

    assert stop_token_ids(IntConfigModel(), Tokenizer()) == {9, 3}


def _manifest():
    return {
        "config": {"k": 8, "beta_kl": 5e-3},
        "packages": {"torch": "test"},
        "states_seen": 4,
        "degenerate_rate": 0.25,
        "invalid_rate": 0.1,
        "cache_hit_rate": 1.0,
        "optimizer_steps": 1,
        "epoch_stats": [{"loss_mean": 0.5, "reward_mean": 0.2}],
        "adapter_sha256": "abc",
    }


def test_write_manifest_roundtrip(tmp_path):
    path = tmp_path / "run_manifest.json"
    manifest = _manifest()
    write_manifest(path, manifest)
    assert json.loads(path.read_text()) == manifest


def test_write_manifest_requires_fields(tmp_path):
    manifest = _manifest()
    del manifest["adapter_sha256"]
    with pytest.raises(ValueError, match="missing"):
        write_manifest(tmp_path / "run_manifest.json", manifest)


def test_write_manifest_refuses_non_finite(tmp_path):
    manifest = _manifest()
    manifest["epoch_stats"][0]["loss_mean"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        write_manifest(tmp_path / "run_manifest.json", manifest)


def test_iter_states_is_deterministic_given_seed(llm_trace):
    first = [(str(path), decision.i) for path, _, decision in iter_states([llm_trace], 42)]
    second = [(str(path), decision.i) for path, _, decision in iter_states([llm_trace], 42)]
    assert first == second
    assert len(first) > 3

    expected = [replayed.decision.i for replayed in replay_model_decisions(llm_trace)]
    assert sorted(index for _, index in first) == sorted(expected)
    assert [index for _, index in first] != expected

    truncated = [
        (str(path), decision.i)
        for path, _, decision in iter_states([llm_trace], 42, max_states=3)
    ]
    assert truncated == first[:3]


def _reseed(trace, out_dir, seed, name):
    lines = trace.read_text().splitlines()
    header = json.loads(lines[0])
    header["seed"] = seed
    out = out_dir / name
    out.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n")
    return out


def test_iter_states_refuses_eval_range_seeds(llm_trace, tmp_path):
    for bad_seed in (42, None):
        doctored = _reseed(llm_trace, tmp_path, bad_seed, f"seed_{bad_seed}.jsonl")
        with pytest.raises(ValueError, match="eval"):
            iter_states([doctored], 42)


def test_iter_states_skips_val_games(llm_trace, tmp_path):
    val_trace = _reseed(llm_trace, tmp_path, 10_020, "val-game.jsonl")
    assert iter_states([val_trace], 42) == []
    assert iter_states([llm_trace, val_trace], 42) == iter_states([llm_trace], 42)


def test_iter_states_interleaves_games(llm_trace, tmp_path_factory):
    other = record_llm_game(tmp_path_factory.mktemp("grpo_interleave"), seed=10_021)
    names = [path.name for path, _, _ in iter_states([llm_trace, other], 42)]
    runs = 1 + sum(a != b for a, b in zip(names, names[1:]))
    assert set(names) == {llm_trace.name, other.name}
    assert runs > 2


def test_grpo_loss_kl_gradient_vanishes_at_reference():
    param = torch.zeros(4, requires_grad=True)
    policy_lp = param - 5.0
    ref_lp = policy_lp.detach().clone()
    advantage = torch.zeros(4)
    loss, _ = grpo_loss(advantage, policy_lp, ref_lp, beta_kl=0.5)
    loss.backward()
    assert torch.allclose(param.grad, torch.zeros(4), atol=1e-7)


def test_grpo_loss_kl_pulls_toward_reference():
    for offset in (1.0, -1.0):
        param = torch.zeros(2, requires_grad=True)
        policy_lp = param - 5.0
        ref_lp = policy_lp.detach() + offset
        loss, _ = grpo_loss(torch.zeros(2), policy_lp, ref_lp, beta_kl=0.5)
        loss.backward()
        step = -param.grad
        assert torch.all(step * offset > 0), (offset, param.grad)
