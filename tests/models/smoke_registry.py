from dataclasses import dataclass
from typing import Any, Callable

import torch
from conippets.config import Config
from torch import nn


@dataclass(frozen=True)
class SmokeCase:
    family: str
    arch: str
    build: Callable[[], nn.Module]
    inputs: Callable[[], tuple[Any, ...]]
    check: Callable[[nn.Module, Any], None]
    call: Callable[[nn.Module], Any] | None = None

    @property
    def id(self):
        return f"{self.family}-{self.arch}"


def make(module, name, **config):
    def build():
        cls = getattr(__import__(module, fromlist=[name]), name)
        return cls(Config(**config))

    return build


def from_config(module, name, config_factory):
    def build():
        cls = getattr(__import__(module, fromlist=[name]), name)
        return cls(config_factory())

    return build


def shape(*dims):
    def check(model, output):
        assert tuple(output.shape) == dims

    return check


def run_case(case):
    model = case.build().eval()
    with torch.no_grad():
        output = case.call(model) if case.call is not None else model(*case.inputs())
    case.check(model, output)
