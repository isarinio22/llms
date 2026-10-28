import sys

import pytest
import torch

from llms.colab import run_stage
from llms.utils import set_seed


def test_set_seed_does_not_query_cuda(monkeypatch):
    """torch.cuda.is_available() poisons fork; vLLM then fails to start its engine process."""
    def boom():
        raise AssertionError("set_seed must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(torch.cuda, "device_count", boom)
    set_seed(3)
    a = torch.rand(2)
    set_seed(3)
    assert torch.equal(a, torch.rand(2))


def test_colab_run_stage_raises_on_failure(capsys):
    with pytest.raises(RuntimeError, match="failed"):
        run_stage("no_such_stage", "configs/smoke.yaml", sys.executable)
    assert "invalid choice" in capsys.readouterr().out
