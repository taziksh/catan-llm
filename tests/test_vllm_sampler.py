import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_grpo import THINK_SUFFIX

EOS = 7
PAD = 0


class FakeTokenizer:
    eos_token_id = EOS
    pad_token_id = PAD

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"] + THINK_SUFFIX

    def __call__(self, prompt, add_special_tokens=False):
        return {"input_ids": [ord(char) % 50 + 10 for char in prompt]}

    def decode(self, token_ids, skip_special_tokens=True):
        return "answer: " + ",".join(map(str, token_ids))


class FakeCompletion:
    def __init__(self, token_ids, stop_reason):
        self.token_ids = token_ids
        self.stop_reason = stop_reason


class FakeRequestOutput:
    def __init__(self, completion):
        self.outputs = [completion]


class FakeLLM:
    completions = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.generate_calls = []
        self.sleep_calls = 0
        self.wake_calls = 0

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.generate_calls.append(
            {"prompts": prompts, "params": params, "lora_request": lora_request}
        )
        return [FakeRequestOutput(completion) for completion in FakeLLM.completions]

    def sleep(self, level):
        self.sleep_calls += 1

    def wake_up(self):
        self.wake_calls += 1


@pytest.fixture
def sampler(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    fake_inputs = types.ModuleType("vllm.inputs")
    fake_inputs.TokensPrompt = lambda prompt_token_ids: {
        "prompt_token_ids": prompt_token_ids
    }
    fake_lora = types.ModuleType("vllm.lora.request")
    fake_lora.LoRARequest = lambda name, id, path: (name, id, path)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_inputs)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora)

    import transformers

    monkeypatch.setattr(
        transformers.GenerationConfig,
        "from_pretrained",
        classmethod(lambda cls, path: types.SimpleNamespace(eos_token_id=[EOS])),
    )
    from run_whole_game_grpo import VLLMSampler

    return VLLMSampler("fake-model", FakeTokenizer(), 1.0, run_seed=42)


def test_prompt_ids_pass_through_exactly(sampler):
    FakeLLM.completions = [FakeCompletion([11, 12, EOS], None)]
    result = sampler(["hello"])[0]
    prompt = "hello" + THINK_SUFFIX
    expected = [ord(char) % 50 + 10 for char in prompt]
    assert result.prompt_ids == expected
    sent = sampler.llm.generate_calls[0]["prompts"][0]["prompt_token_ids"]
    assert sent == expected


def test_stop_token_appended_when_stripped(sampler):
    FakeLLM.completions = [FakeCompletion([11, 12], EOS)]
    result = sampler(["hello"])[0]
    assert result.completion_ids == [11, 12, EOS]


def test_stop_token_not_duplicated(sampler):
    FakeLLM.completions = [FakeCompletion([11, 12, EOS], EOS)]
    result = sampler(["hello"])[0]
    assert result.completion_ids == [11, 12, EOS]


def test_eos_finish_kept_as_is(sampler):
    FakeLLM.completions = [FakeCompletion([11, EOS], None)]
    result = sampler(["hello"])[0]
    assert result.completion_ids == [11, EOS]


def test_too_long_prompt_raises(sampler):
    sampler.max_prompt_tokens = 3
    with pytest.raises(RuntimeError, match="truncation is forbidden"):
        sampler(["hello"])


def test_distinct_seeds_per_prompt_and_calls(sampler):
    FakeLLM.completions = [
        FakeCompletion([11, EOS], None),
        FakeCompletion([12, EOS], None),
    ]
    sampler(["one", "two"])
    first = [params["seed"] for params in sampler.llm.generate_calls[0]["params"]]
    sampler(["one", "two"])
    second = [params["seed"] for params in sampler.llm.generate_calls[1]["params"]]
    assert len(set(first)) == 2
    assert set(first).isdisjoint(second)


def test_sleep_wake_idempotent(sampler):
    sampler.wake()
    assert sampler.llm.wake_calls == 0
    sampler.sleep()
    sampler.sleep()
    assert sampler.llm.sleep_calls == 1
    sampler.wake()
    sampler.wake()
    assert sampler.llm.wake_calls == 1


def test_load_adapter_bumps_version(sampler):
    sampler.load_adapter(Path("ckpt/update_001/policy"))
    first = sampler.lora_request
    sampler.load_adapter(Path("ckpt/update_002/policy"))
    second = sampler.lora_request
    assert first[1] == 1 and second[1] == 2
    assert second[2].endswith("update_002/policy")
