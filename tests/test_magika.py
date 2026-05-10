import os
import torch
import pytest
import numpy as np
import magika
from nnm.models.magika import Magika


@torch.no_grad()
@pytest.mark.parametrize('raw_bytes', [
    b'function log(msg) {console.log(msg);}',
    b'#include <stdio.h>\nint main() { return 0; }',
    b'<html><body>hello</body></html>',
    b'import os\nfor root, dirs, files in os.walk("/tmp"):\n    for f in files:\n        print(f)',
    r'<?php echo "hello"; phpinfo(); ?>'.encode(),
    b'body { color: red; background: blue; font-size: 16px; }',
    b'#!/bin/bash\necho "hello"\nls -la\npwd\ncat /tmp/test.txt',
    b'#!/usr/bin/env node\nconsole.log("hello");\nconst x = 1;\nmodule.exports = x;',
    b'pragma solidity ^0.8.0;\ncontract Test {\n    uint256 public value;\n}',
])
def test_magika_pretrained(model_path, raw_bytes):
    assert model_path is not None, '--model-path is required'
    nnm_model = Magika.from_pretrained(model_path)
    nnm_model.eval()

    ref_model = magika.Magika()

    x = torch.tensor(bytearray(raw_bytes), dtype=torch.uint8).reshape(1, -1)
    probs = nnm_model(x).numpy().flatten()

    pred_idx = int(probs.argmax())
    nnm_label = nnm_model.labels[pred_idx]
    nnm_score = float(probs[pred_idx])

    ref_result = ref_model.identify_bytes(raw_bytes)
    ref_label = ref_result.output.label
    ref_score = ref_result.score

    assert nnm_label == ref_label, (
        f'Label mismatch: nnm={nnm_label}, ref={ref_label}'
    )
    assert abs(nnm_score - ref_score) < 1e-3, (
        f'Score mismatch for {nnm_label}: nnm={nnm_score:.4f}, ref={ref_score:.4f}'
    )