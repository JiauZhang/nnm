import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnm.models.qwen2 import Qwen2LM

pytestmark = pytest.mark.pretrained


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize(
    "prompts",
    [
        ["Hello, how are you?", " What about Python?"],
        ["Hello", " Tell me more."],
        ["What is the capital of France?", " What about Germany?"],
    ],
)
@torch.no_grad()
def test_qwen2_pretrained(require_model_path, source_repo, prompts, use_cache):
    nnm_model = Qwen2LM.from_pretrained(require_model_path)
    nnm_model.eval()

    weight_repo = os.path.expanduser(source_repo(require_model_path))
    hf_model = AutoModelForCausalLM.from_pretrained(weight_repo, torch_dtype=torch.float32, device_map="cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(weight_repo)

    if not use_cache:
        input_ids = tokenizer(prompts[0], return_tensors="pt")["input_ids"]
        expected = hf_model(input_ids).logits
        actual = nnm_model(input_ids)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        assert tokenizer.decode(expected.argmax(dim=-1)[0]) == tokenizer.decode(actual.argmax(dim=-1)[0])

    hf_model_gen = AutoModelForCausalLM.from_pretrained(weight_repo, torch_dtype=torch.float32, device_map="cpu").eval()
    hf_model_gen.config._attn_implementation = "eager"
    hf_model_gen.config.use_cache = use_cache

    hf_input_ids = tokenizer(prompts[0], return_tensors="pt")["input_ids"]
    nnm_input_ids = hf_input_ids.clone()
    hf_past_key_values = None

    for turn_idx, prompt in enumerate(prompts):
        if turn_idx > 0:
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            hf_input_ids = torch.cat([hf_input_ids, input_ids], dim=1)
            nnm_input_ids = torch.cat([nnm_input_ids, input_ids], dim=1)
        else:
            input_ids = hf_input_ids

        if use_cache:
            hf_out = hf_model_gen(input_ids, past_key_values=hf_past_key_values, use_cache=True)
            hf_past_key_values = hf_out.past_key_values
            hf_logits = hf_out.logits
            nnm_model(input_ids)
        else:
            hf_out = hf_model_gen(hf_input_ids, use_cache=False)
            hf_logits = hf_out.logits
            nnm_model(nnm_input_ids)

        next_token = hf_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        hf_input_ids = torch.cat([hf_input_ids, next_token], dim=1)
        nnm_input_ids = torch.cat([nnm_input_ids, next_token], dim=1)

        for _ in range(1, 10):
            if use_cache:
                hf_out = hf_model_gen(hf_input_ids[:, -1:], past_key_values=hf_past_key_values, use_cache=True)
                hf_past_key_values = hf_out.past_key_values
                hf_logits = hf_out.logits
                nnm_model(nnm_input_ids[:, -1:])
            else:
                hf_out = hf_model_gen(hf_input_ids, use_cache=False)
                hf_logits = hf_out.logits
                nnm_model(nnm_input_ids)

            next_token = hf_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            hf_input_ids = torch.cat([hf_input_ids, next_token], dim=1)
            nnm_input_ids = torch.cat([nnm_input_ids, next_token], dim=1)

    assert tokenizer.decode(hf_input_ids[0]) == tokenizer.decode(nnm_input_ids[0])
