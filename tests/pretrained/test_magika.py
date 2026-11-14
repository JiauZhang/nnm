import magika
import pytest
import torch

from nnm.models.magika import Magika

pytestmark = pytest.mark.pretrained


@pytest.fixture(scope="module")
def model(load_pretrained):
    return load_pretrained(Magika)


@torch.no_grad()
@pytest.mark.parametrize(
    "raw_bytes",
    [
        b"function log(msg) {console.log(msg);}",
        b"#include <stdio.h>\nint main() { return 0; }",
        b"<html><body>hello</body></html>",
        b'import os\nfor root, dirs, files in os.walk("/tmp"):\n    for f in files:\n        print(f)',
        r'<?php echo "hello"; phpinfo(); ?>'.encode(),
        b"body { color: red; background: blue; font-size: 16px; }",
        b'#!/bin/bash\necho "hello"\nls -la\npwd\ncat /tmp/test.txt',
        b'#!/usr/bin/env node\nconsole.log("hello");\nconst x = 1;\nmodule.exports = x;',
        b"pragma solidity ^0.8.0;\ncontract Test {\n    uint256 public value;\n}",
    ],
)
def test_magika_pretrained(model, raw_bytes):
    reference = magika.Magika()
    x = torch.tensor(bytearray(raw_bytes), dtype=torch.uint8).reshape(1, -1)
    probs = model(x).numpy().flatten()

    pred_idx = int(probs.argmax())
    nnm_label = model.labels[pred_idx]
    nnm_score = float(probs[pred_idx])

    ref_result = reference.identify_bytes(raw_bytes)
    assert nnm_label == ref_result.output.label
    assert abs(nnm_score - ref_result.score) < 1e-3
