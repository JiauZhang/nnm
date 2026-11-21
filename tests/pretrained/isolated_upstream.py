import subprocess
import sys

import torch


def run_isolated_upstream(worker_source, variables, output_path):
    """Run upstream reference code in a fresh subprocess.

    Reference implementations (depth_anything_v2 / depth_pro_2) need sys.path and
    sys.modules injections that must never leak into the pytest process, so every
    upstream comparison runs in its own interpreter. `variables` are substituted
    into the worker source before execution. The worker writes the reference
    output tensor to output_path; we load and return it.
    """
    for placeholder, value in variables.items():
        worker_source = worker_source.replace(placeholder, repr(str(value)))
    script = output_path.with_suffix(".py")
    script.write_text(worker_source)
    command = [sys.executable, "-I", str(script), str(output_path)]
    subprocess.run(command, check=True)
    return torch.load(output_path, weights_only=True)
