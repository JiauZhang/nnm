import json

import torch
from conippets.config import Config
from safetensors.torch import save_file
from torch import nn

from nnm.pretrained import PretrainedModel


class TinyModel(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.scale = config.get("scale", 1.0)
        self.first = nn.Linear(config.hidden, config.hidden, bias=False)
        self.second = nn.Linear(config.hidden, config.out, bias=False)

    def forward(self, x):
        return self.second(self.first(x)) * self.scale


def write_shards(path, state):
    names = sorted(state)
    split = len(names) // 2
    shards = [names[:split], names[split:]]
    weight_map = {}
    total_size = 0
    for index, keys in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        payload = {}
        for key in keys:
            tensor = state[key].contiguous()
            payload[key] = tensor
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()
        save_file(payload, str(path / filename))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )


def test_low_cpu_mem_usage_loads_sharded_weights(tmp_path):
    config = Config(hidden=8, out=4, scale=2.0)
    (tmp_path / "config.json").write_text(json.dumps(dict(config)))
    source = TinyModel(config).eval()
    expected = {key: value.clone() for key, value in source.state_dict().items()}
    write_shards(tmp_path, expected)

    model = TinyModel.from_pretrained(
        str(tmp_path),
        weights={"device": "cpu", "low_cpu_mem_usage": True},
    ).eval()

    assert not any(parameter.is_meta for parameter in model.parameters())
    assert model.scale == 2.0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_low_cpu_mem_usage_honors_strict(tmp_path):
    config = Config(hidden=8, out=4, scale=2.0)
    (tmp_path / "config.json").write_text(json.dumps(dict(config)))
    source = TinyModel(config)
    state = source.state_dict()
    write_shards(tmp_path, {key: value for key, value in state.items() if key != "second.weight"})

    try:
        TinyModel.from_pretrained(
            str(tmp_path),
            weights={"device": "cpu", "low_cpu_mem_usage": True, "strict": True},
        )
    except RuntimeError as error:
        assert "second.weight" in str(error)
    else:
        raise AssertionError("strict loading should reject missing keys")


def test_config_overlay_applied(tmp_path):
    saved_config = Config(hidden=8, out=4, scale=1.0)
    (tmp_path / "config.json").write_text(json.dumps(dict(saved_config)))
    source = TinyModel(saved_config)
    write_shards(tmp_path, source.state_dict())

    model = TinyModel.from_pretrained(
        str(tmp_path),
        source=Config(),
        weights=Config(device="cpu"),
        config=Config(scale=3.0),
    ).eval()

    assert model.scale == 3.0
    for key, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)


def test_local_meta_repo(tmp_path):
    config = Config(hidden=8, out=4, scale=4.0)
    (tmp_path / "config.json").write_text(json.dumps(dict(config)))
    source = TinyModel(config)
    save_file({key: value.contiguous() for key, value in source.state_dict().items()}, str(tmp_path / "local.safetensors"))
    (tmp_path / "nnm_weights.json").write_text(
        json.dumps({"version": 1, "weights": ["local.safetensors"]})
    )

    model = TinyModel.from_pretrained(str(tmp_path), weights={"device": "cpu"}).eval()

    assert model.scale == 4.0
    for key, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)
