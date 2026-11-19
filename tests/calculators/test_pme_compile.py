"""
``torch.compile`` behaviour tests for ``PMECalculator``.

The mesh path reads the mesh shape as Python ints, so ``torch.compile`` captures the
forward with the mesh-shape CPU syncs as the only graph breaks (no per-axis
``int(tensor)`` calls, no data-dependent jumps).
"""

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

import torchpme

sys.path.append(str(Path(__file__).parents[1]))
from helpers import DEVICES, DTYPES


@contextmanager
def _suppress_torch_internal_warnings():
    """
    Silence torch-internal warnings emitted while compiling/executing the
    ``torch.compile`` graph (e.g. complex-op codegen fallback, deprecation notes).
    These vary by torch version and are out of our control; the suite runs under
    ``filterwarnings = ["error"]``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# TorchInductor falls back to eager for the complex FFT ops and emits a (harmless)
# perf warning, which the suite's ``filterwarnings = ["error"]`` would otherwise turn
# into a failure. Hard lowering errors are not warnings and would still fail.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Torchinductor does not support code generation for complex operators"
    ),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]

SMEARING = 1.0
MESH_SPACING = 0.5


def _system(device, dtype, n=60, length=8.0):
    """A small random periodic system; SR neighbor list is empty (LR-only test)."""
    torch.manual_seed(0)
    cell = torch.eye(3, dtype=dtype, device=device) * length
    positions = torch.rand((n, 3), dtype=dtype, device=device) * length
    charges = ((torch.rand(n, dtype=dtype, device=device) - 0.5) * 2.0).unsqueeze(-1)
    neighbor_indices = torch.zeros((0, 2), dtype=torch.int64, device=device)
    neighbor_distances = torch.zeros((0,), dtype=dtype, device=device)
    periodic = torch.tensor([True, True, True], device=device)
    return charges, cell, positions, neighbor_indices, neighbor_distances, periodic


def _calculator(device, dtype):
    calc = torchpme.PMECalculator(
        potential=torchpme.CoulombPotential(smearing=SMEARING),
        mesh_spacing=MESH_SPACING,
    )
    calc.to(device=device, dtype=dtype)
    return calc


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_pme_torch_compile_default(device, dtype):
    """``torch.compile(mode="default")`` matches eager within fp tolerance."""
    calc = _calculator(device, dtype)
    charges, cell, pos, ni, nd, per = _system(device, dtype)
    eager = calc.forward(charges, cell, pos, ni, nd, periodic=per)

    compiled = torch.compile(calc.forward, mode="default")
    with _suppress_torch_internal_warnings():
        out = compiled(charges, cell, pos, ni, nd, periodic=per)

    atol = 1e-5 if dtype == torch.float32 else 1e-10
    assert_close(out, eager, rtol=1e-5, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_pme_compile_graph_breaks_only_mesh_shape_sync(device):
    """
    The only graph breaks left are the mesh-shape CPU syncs.

    The mesh shape ``(nx, ny, nz)`` is data-dependent on ``cell``, so extracting it as
    Python ints (``tolist``) is an unavoidable sync. Everything else (k-vectors,
    interpolation, FFT filter, corrections) must capture without breaking.
    """
    import torch._dynamo as dynamo

    dtype = torch.float64
    calc = _calculator(device, dtype)
    charges, cell, pos, ni, nd, per = _system(device, dtype)

    dynamo.reset()
    with _suppress_torch_internal_warnings():
        explanation = dynamo.explain(calc.forward)(
            charges, cell, pos, ni, nd, periodic=per
        )

    for reason in explanation.break_reasons:
        text = str(reason.reason).lower()
        assert ("data dependent" in text) or ("tolist" in text), (
            f"unexpected graph-break reason: {reason.reason}"
        )
        frames = " ".join(f"{f.filename}:{f.name}" for f in reason.user_stack)
        assert "update" in frames or "_generate_kvectors" in frames, (
            f"graph break outside the mesh-shape extraction: {frames}"
        )
