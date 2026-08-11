import hashlib
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from catanatron import Color

from catan_llm.schema import Player
from catan_llm.simulation import GameOutcome
from catan_llm.whole_game import (
    DEFAULT_INVALID_RETRIES,
    DEFAULT_VP_COEF,
    DecisionSample,
    Rollout,
    SampledCompletion,
    assign_group_advantages,
    group_advantages,
    make_rollouts,
    rollout_games,
    terminal_reward,
    trainable_samples,
    training_seeds,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_whole_game_grpo import (
    DEFAULT_GROUP_SIZE,
    DEFAULT_GROUPS_PER_UPDATE,
    DEFAULT_TEMPERATURE,
    DEFAULT_UPDATES,
    _batch_token_logprobs,
    checkpoint_state,
    clipped_surrogate_loss,
    record_old_logprobs,
    retire_optimizer,
    run_config,
    save_update,
    update_policy,
    validate_resume,
)
from test_run_grpo import tiny_model


def test_approved_rollout_shape_and_reward_defaults():
    assert DEFAULT_GROUP_SIZE == 8
    assert DEFAULT_GROUPS_PER_UPDATE == 4
    assert DEFAULT_UPDATES == 25
    assert DEFAULT_GROUP_SIZE * DEFAULT_GROUPS_PER_UPDATE * DEFAULT_UPDATES == 800
    assert DEFAULT_TEMPERATURE == 1.0
    assert DEFAULT_INVALID_RETRIES == 1
    assert DEFAULT_VP_COEF == 0.1


def test_checkpoint_resume_uses_the_next_zero_based_update_index():
    assert checkpoint_state(1) == {
        "completed_update": 1,
        "next_update_index": 1,
        "optimizer_retained": True,
    }
    assert checkpoint_state(25)["next_update_index"] == 25


def test_checkpoint_is_complete_and_never_overwritten(tmp_path):
    class Model:
        def save_pretrained(self, path, selected_adapters):
            assert selected_adapters == ["policy"]
            policy = Path(path) / "policy"
            policy.mkdir(parents=True)
            (policy / "adapter_model.safetensors").write_bytes(b"adapter")

    class Optimizer:
        def state_dict(self):
            return {"state": {}, "param_groups": []}

    checkpoint = save_update(Model(), Optimizer(), tmp_path, update=1)

    assert checkpoint == tmp_path / "checkpoints" / "update_001"
    assert (checkpoint / "optimizer.pt").exists()
    state = json.loads((checkpoint / "state.json").read_text())
    assert state["completed_update"] == 1
    assert state["next_update_index"] == 1
    assert state["optimizer_retained"] is True
    assert not list((tmp_path / "checkpoints").glob("*.partial-*"))
    with pytest.raises(FileExistsError, match="overwrite"):
        save_update(Model(), Optimizer(), tmp_path, update=1)

    retire_optimizer(checkpoint)
    assert not (checkpoint / "optimizer.pt").exists()
    state = json.loads((checkpoint / "state.json").read_text())
    assert state["optimizer_retained"] is False
    retire_optimizer(checkpoint)


def _args(tmp_path, **overrides):
    values = {
        "model": "/workspace/merged3",
        "resume_from": tmp_path / "checkpoints" / "update_001",
        "updates": 25,
        "groups_per_update": 4,
        "group_size": 8,
        "temperature": 1.0,
        "lr": 1e-6,
        "clip_epsilon": 0.2,
        "policy_epochs": 1,
        "seed": 42,
        "seed_start": 10_000,
        "invalid_retries": 1,
        "max_turns": 500,
        "vp_coef": 0.1,
        "max_prompt_tokens": 8192,
        "sampler": "inprocess",
    }
    values.update(overrides)
    return Namespace(**values)


def test_resume_requires_exact_latest_checkpoint_and_config(tmp_path):
    args = _args(tmp_path)
    args.resume_from.mkdir(parents=True)
    policy = args.resume_from / "policy"
    policy.mkdir()
    adapter = policy / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    adapter_sha = hashlib.sha256(b"adapter").hexdigest()
    state = checkpoint_state(1) | {"adapter_sha256": adapter_sha}
    manifest = {
        "status": "failed",
        "model": args.model,
        "initial_adapter_sha256": "initial",
        "config": run_config(args),
        "last_completed_update": 1,
        "updates": [{"update": 1, "checkpoint": str(args.resume_from)}],
    }

    validate_resume(manifest, state, args, "initial")

    changed = _args(tmp_path, lr=2e-6)
    with pytest.raises(RuntimeError, match="resume config mismatch: learning_rate"):
        validate_resume(manifest, state, changed, "initial")

    older = _args(tmp_path, resume_from=tmp_path / "checkpoints" / "update_000")
    with pytest.raises(RuntimeError, match="checkpoint path"):
        validate_resume(manifest, state, older, "initial")


def test_terminal_reward_keeps_vp_for_losses_and_truncations():
    loss = _rollout(10_000, 0, None)
    loss.outcome = GameOutcome(
        Player.BLUE,
        {Player.RED: 7, Player.BLUE: 10, Player.WHITE: 6, Player.ORANGE: 5},
        80,
    )
    truncated = _rollout(10_000, 1, None)
    truncated.outcome = GameOutcome(
        None,
        {Player.RED: 6, Player.BLUE: 7, Player.WHITE: 5, Player.ORANGE: 4},
        500,
    )
    win = _rollout(10_000, 2, None)
    win.outcome = GameOutcome(
        Player.RED,
        {Player.RED: 10, Player.BLUE: 6, Player.WHITE: 5, Player.ORANGE: 4},
        70,
    )

    assert terminal_reward(loss) == pytest.approx(0.07)
    assert terminal_reward(truncated) == pytest.approx(0.06)
    assert terminal_reward(win) == pytest.approx(1.1)


def test_group_advantages_are_centered_and_std_normalized():
    values = group_advantages([0.02, 0.06, 0.08, 1.1])
    assert sum(values) == pytest.approx(0.0)
    assert torch.tensor(values).std(correction=0).item() == pytest.approx(1.0)
    assert values[0] < values[1] < values[2] < values[3]
    assert group_advantages([0.05] * 8) == [0.0] * 8


def _rollout(seed, index, reward):
    return Rollout(
        seed=seed,
        rollout_index=index,
        game=None,
        hero_color=Color.RED,
        fallback_rng=random.Random(index),
        reward=reward,
    )


def test_group_advantages_are_separate_and_failures_are_excluded():
    rollouts = [
        _rollout(10_000, 0, 0.02),
        _rollout(10_000, 1, 0.08),
        _rollout(10_000, 2, None),
        _rollout(10_001, 0, 0.05),
        _rollout(10_001, 1, 0.05),
    ]

    stats = assign_group_advantages(rollouts)

    assert stats == {"groups": 2, "degenerate_groups": 1, "failed_games": 1}
    assert [rollouts[0].advantage, rollouts[1].advantage] == pytest.approx([-1, 1])
    assert rollouts[2].advantage == 0.0
    assert rollouts[3].advantage == rollouts[4].advantage == 0.0


def test_every_valid_decision_inherits_its_game_advantage():
    first = _rollout(10_000, 0, 0.02)
    second = _rollout(10_000, 1, 0.08)
    first.samples = [DecisionSample([1], [2]), DecisionSample([3], [4])]
    second.samples = [DecisionSample([5], [6])]
    assign_group_advantages([first, second])

    trainable = trainable_samples([first, second])

    assert [
        (rollout.advantage, sample.prompt_ids) for rollout, sample in trainable
    ] == [
        (pytest.approx(-1), [1]),
        (pytest.approx(-1), [3]),
        (pytest.approx(1), [5]),
    ]


def _decision_samples():
    return [
        DecisionSample(prompt_ids=[1, 2, 3], completion_ids=[4, 5]),
        DecisionSample(prompt_ids=[6, 7, 8, 9, 10, 11], completion_ids=[12]),
        DecisionSample(prompt_ids=[13, 14], completion_ids=[15, 16, 17, 18]),
    ]


def test_batched_logprobs_match_single_sample():
    model = tiny_model()
    samples = _decision_samples()

    batched = _batch_token_logprobs(model, samples)
    singles = [_batch_token_logprobs(model, [sample])[0] for sample in samples]

    for together, alone, sample in zip(batched, singles, samples, strict=True):
        assert together.shape == alone.shape == (len(sample.completion_ids),)
        assert torch.allclose(together, alone, atol=1e-5)
    assert batched[0].requires_grad


def _trainable(samples):
    rollouts = [
        _rollout(10_001, index, 0.05) for index in range(len(samples))
    ]
    for rollout, advantage in zip(rollouts, (1.0, -0.5, 0.25), strict=True):
        rollout.advantage = advantage
    return list(zip(rollouts, samples, strict=True))


def test_update_policy_micro_batch_matches_single_sample():
    torch.manual_seed(0)
    samples = _decision_samples()

    metrics = {}
    for micro_batch in (1, 3):
        model = tiny_model()
        trainable = _trainable([DecisionSample(s.prompt_ids, s.completion_ids) for s in samples])
        record_old_logprobs(model, trainable, micro_batch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        metrics[micro_batch] = update_policy(
            model,
            optimizer,
            trainable,
            games_in_batch=3,
            policy_epochs=1,
            clip_epsilon=0.2,
            micro_batch=micro_batch,
        )

    first, batched = metrics[1][0], metrics[3][0]
    assert batched["loss_mean"] == pytest.approx(first["loss_mean"], abs=1e-6)
    assert batched["ratio_mean"] == pytest.approx(first["ratio_mean"], abs=1e-6)
    assert batched["clip_fraction"] == first["clip_fraction"]
    assert batched["grad_norm"] == pytest.approx(first["grad_norm"], rel=1e-4)


@pytest.mark.parametrize(
    ("advantage", "ratio", "expected_loss"),
    [
        (1.0, 2.0, -1.2),
        (-1.0, 2.0, 2.0),
        (1.0, 0.5, -0.5),
        (-1.0, 0.5, 0.8),
    ],
)
def test_clipped_surrogate_matches_ppo(advantage, ratio, expected_loss):
    old = torch.tensor(-2.0)
    new = old + torch.log(torch.tensor(ratio))
    loss, actual_ratio = clipped_surrogate_loss(new, old, torch.tensor(advantage), 0.2)
    assert actual_ratio.item() == pytest.approx(ratio)
    assert loss.item() == pytest.approx(expected_loss)


def test_clipped_surrogate_averages_completion_tokens():
    old = torch.tensor([-2.0, -2.0])
    new = old + torch.log(torch.tensor([2.0, 0.5]))

    loss, ratios = clipped_surrogate_loss(new, old, torch.tensor(1.0), 0.2)

    assert ratios.tolist() == pytest.approx([2.0, 0.5])
    assert loss.item() == pytest.approx(-(1.2 + 0.5) / 2)


def _first_legal(prompt: str) -> str:
    line = prompt.split("YOUR OPTIONS\n", 1)[1].splitlines()[0]
    return line.split(" (", 1)[0]


def _completion(text: str) -> SampledCompletion:
    return SampledCompletion(text=text, prompt_ids=[1, 2], completion_ids=[3])


def test_training_seeds_skip_validation_boards():
    assert training_seeds(10_000, 0, 4) == [10_001, 10_002, 10_003, 10_004]
    assert training_seeds(10_000, 8, 4) == [10_009, 10_011, 10_012, 10_013]
    assert training_seeds(10_000, 0, 12) == training_seeds(10_000, 0, 8) + training_seeds(10_000, 8, 4)
    assert all(seed % 10 != 0 for seed in training_seeds(10_000, 0, 100))


def test_rollout_uses_stateless_valid_decisions():
    rollouts = make_rollouts([10_001], group_size=2, run_seed=42)

    def complete(prompts):
        return [_completion(f"answer: {_first_legal(prompt)}") for prompt in prompts]

    rollout_games(rollouts, complete, max_turns=2)

    assert all(rollout.outcome is not None for rollout in rollouts)
    assert all(rollout.outcome.truncated for rollout in rollouts)
    assert all(rollout.decision_states > 0 for rollout in rollouts)
    assert all(len(rollout.samples) == rollout.decision_states for rollout in rollouts)
    assert all(rollout.invalid_replies == 0 for rollout in rollouts)
    assert all(rollout.reward is not None for rollout in rollouts)


def test_invalid_reply_gets_one_retry_and_no_training_sample():
    (rollout,) = make_rollouts([10_001], group_size=1, run_seed=42)

    def complete(prompts):
        return [
            _completion(
                f"answer: {_first_legal(prompt)}"
                if prompt.startswith("Your last reply")
                else "not a move"
            )
            for prompt in prompts
        ]

    rollout_games([rollout], complete, invalid_retries=1, max_turns=2)

    assert rollout.outcome is not None
    assert rollout.invalid_replies == rollout.decision_states
    assert len(rollout.samples) == rollout.decision_states


def test_second_invalid_uses_fallback_without_creating_model_action():
    (rollout,) = make_rollouts([10_001], group_size=1, run_seed=42)

    rollout_games(
        [rollout],
        lambda prompts: [_completion("not a move") for _ in prompts],
        invalid_retries=1,
        max_turns=2,
    )

    assert rollout.outcome is not None
    assert rollout.invalid_replies == 2 * rollout.decision_states
    assert rollout.samples == []
