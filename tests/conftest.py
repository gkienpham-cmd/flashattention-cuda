"""pytest fixtures + path setup shared across the test suite.

The repo isn't pip-installed during the journey (we JIT-compile in Colab), so we put the repo
root on sys.path here. We also centralize the "skip if this GPU can't do it" guard so arch-gated
tests (fp8 on a T4, etc.) skip cleanly instead of erroring.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def requires_cuda():
    """Skip marker: most tests need a real GPU (Colab T4 / rented box)."""
    import torch
    return pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="needs a CUDA GPU (runs on Colab T4 / rented box)")
