"""
``torch.compile`` regression/behaviour tests for ``PMECalculator``.

After the functional refactor of ``PMECalculator._compute_kspace`` (no per-forward
state mutation, mesh shape extracted as a single Python-int sync), ``torch.compile``
captures the forward without the previous fragmentation, works with
``mode="reduce-overhead"`` (CUDA Graphs), and preserves autograd w.r.t. ``cell``.
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
    ``torch.compile`` graph (e.g. complex-op codegen fallback, deprecation and
    non-leaf ``.grad`` notes from compiled autograd). These vary by torch version and
    are out of our control; the suite runs under ``filterwarnings = ["error"]``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# The mesh calculators run complex FFTs; TorchInductor falls back to eager for the
# complex ops and emits a (harmless) perf warning, which the suite's
# ``filterwarnings = ["error"]`` would otherwise turn into a failure. Hard lowering
# errors are not warnings and would still fail. The second filter silences a
# torch-internal deprecation warning emitted while compiling the ``linalg_det``
# backward on some torch versions (out of our control).
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


def _calculator(device, dtype, ns_mesh=None):
    calc = torchpme.PMECalculator(
        potential=torchpme.CoulombPotential(smearing=SMEARING),
        mesh_spacing=MESH_SPACING,
        ns_mesh=ns_mesh,
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
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.filterwarnings(
    "ignore:Torchinductor does not support code generation for complex operators"
)
def test_pme_torch_compile_reduce_overhead(device, dtype):
    """
    ``mode="reduce-overhead"`` (CUDA Graphs on GPU) runs and matches eager.

    This mode is incompatible with per-forward state mutation; it only works because
    the refactored ``_compute_kspace`` writes nothing to ``self``.
    """
    calc = _calculator(device, dtype)
    charges, cell, pos, ni, nd, per = _system(device, dtype)
    eager = calc.forward(charges, cell, pos, ni, nd, periodic=per)

    compiled = torch.compile(calc.forward, mode="reduce-overhead")
    # Call a few times: CUDA Graphs warm up / record over the first calls.
    with _suppress_torch_internal_warnings():
        for _ in range(3):
            out = compiled(charges, cell, pos, ni, nd, periodic=per)

    atol = 1e-5 if dtype == torch.float32 else 1e-10
    assert_close(out.cpu(), eager.cpu(), rtol=1e-5, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_pme_compile_graph_breaks_only_mesh_sync(device):
    """
    The only graph breaks left are the single mesh-shape CPU sync.

    The mesh shape ``(nx, ny, nz)`` is data-dependent on ``cell``, so extracting it as
    Python ints is one unavoidable sync. Everything else (interpolation, FFT filter,
    corrections) must capture without breaking. We assert the break count is small and
    that every break is that scalar-extraction sync.
    """
    import torch._dynamo as dynamo

    dtype = torch.float64
    calc = _calculator(device, dtype)
    charges, cell, pos, ni, nd, per = _system(device, dtype)

    dynamo.reset()
    explanation = dynamo.explain(calc.forward)(charges, cell, pos, ni, nd, periodic=per)

    # One tolist() sync; allow a small margin across torch versions.
    assert explanation.graph_break_count <= 3, (
        f"unexpected graph breaks: {explanation.graph_break_count}"
    )
    for reason in explanation.break_reasons:
        text = str(reason.reason).lower()
        assert (
            ("item" in text)
            or ("tolist" in text)
            or ("local_scalar" in text)
            or ("data dependent" in text)
        ), f"unexpected graph-break reason: {reason.reason}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_pme_compile_ad_through_cell(device, dtype):
    """Autograd w.r.t. ``cell`` is preserved and matches eager under compile."""
    calc = _calculator(device, dtype)
    charges, cell, pos, ni, nd, per = _system(device, dtype)

    def energy(forward, cell_):
        return forward(charges, cell_, pos, ni, nd, periodic=per).sum()

    cell_e = cell.clone().requires_grad_(True)
    grad_eager = torch.autograd.grad(energy(calc.forward, cell_e), cell_e)[0]

    compiled = torch.compile(calc.forward, mode="default")
    cell_c = cell.clone().requires_grad_(True)
    with _suppress_torch_internal_warnings():
        grad_compiled = torch.autograd.grad(energy(compiled, cell_c), cell_c)[0]

    assert torch.isfinite(grad_compiled).all()
    assert grad_compiled.abs().sum() > 0
    atol = 1e-4 if dtype == torch.float32 else 1e-9
    assert_close(grad_compiled, grad_eager, rtol=1e-4, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_pme_compile_fluctuating_cell(device):
    """
    A changed cell changes the energy; revisiting a cell gives the same energy
    (no stale per-forward state leaks across calls under compile).
    """
    dtype = torch.float64
    calc = _calculator(device, dtype)
    charges, cell, pos, ni, nd, per = _system(device, dtype)

    compiled = torch.compile(calc.forward, mode="default")

    with _suppress_torch_internal_warnings():
        e1 = compiled(charges, cell, pos, ni, nd, periodic=per)
        e2 = compiled(charges, cell * 1.05, pos, ni, nd, periodic=per)
        e1_again = compiled(charges, cell, pos, ni, nd, periodic=per)

    assert not torch.allclose(e1, e2)
    assert_close(e1, e1_again, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_pme_fixed_ns_mesh_matches_adaptive(device, dtype):
    """
    A fixed ``ns_mesh`` equal to what ``get_ns_mesh`` would derive gives bit-identical
    energies (eager) and preserves AD w.r.t. ``cell``.
    """
    charges, cell, pos, ni, nd, per = _system(device, dtype)
    ns = torchpme.lib.get_ns_mesh(cell, MESH_SPACING).tolist()

    adaptive = _calculator(device, dtype)
    fixed = _calculator(device, dtype, ns_mesh=(ns[0], ns[1], ns[2]))

    e_adaptive = adaptive.forward(charges, cell, pos, ni, nd, periodic=per)
    e_fixed = fixed.forward(charges, cell, pos, ni, nd, periodic=per)
    assert_close(e_fixed, e_adaptive, rtol=0.0, atol=0.0)

    cell_g = cell.clone().requires_grad_(True)
    grad = torch.autograd.grad(
        fixed.forward(charges, cell_g, pos, ni, nd, periodic=per).sum(), cell_g
    )[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_pme_fixed_ns_mesh_zero_graph_breaks(device):
    """With a fixed ``ns_mesh`` the forward is captured as a single graph."""
    import torch._dynamo as dynamo

    dtype = torch.float64
    charges, cell, pos, ni, nd, per = _system(device, dtype)
    ns = torchpme.lib.get_ns_mesh(cell, MESH_SPACING).tolist()
    calc = _calculator(device, dtype, ns_mesh=(ns[0], ns[1], ns[2]))

    dynamo.reset()
    explanation = dynamo.explain(calc.forward)(charges, cell, pos, ni, nd, periodic=per)
    assert explanation.graph_break_count == 0, (
        f"unexpected graph breaks: {explanation.break_reasons}"
    )
    assert explanation.graph_count == 1

    compiled = torch.compile(calc.forward, mode="default")
    with _suppress_torch_internal_warnings():
        out = compiled(charges, cell, pos, ni, nd, periodic=per)
    eager = calc.forward(charges, cell, pos, ni, nd, periodic=per)
    assert_close(out, eager, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("ns_mesh", [(16, 16), (16, 16, 0), (16, -8, 16)])
def test_pme_fixed_ns_mesh_invalid(ns_mesh):
    match = "`ns_mesh` must be three positive integers"
    with pytest.raises(ValueError, match=match):
        torchpme.PMECalculator(
            potential=torchpme.CoulombPotential(smearing=SMEARING),
            mesh_spacing=MESH_SPACING,
            ns_mesh=ns_mesh,
        )


def test_pme_fixed_ns_mesh_scriptable():
    """TorchScript still scripts the calculator with a fixed mesh shape."""
    calc = torchpme.PMECalculator(
        potential=torchpme.CoulombPotential(smearing=SMEARING),
        mesh_spacing=MESH_SPACING,
        ns_mesh=(16, 16, 16),
    )
    scripted = torch.jit.script(calc)
    charges, cell, pos, ni, nd, per = _system("cpu", torch.float64)
    e_script = scripted(charges, cell, pos, ni, nd, periodic=per)
    e_eager = calc.forward(charges, cell, pos, ni, nd, periodic=per)
    assert_close(e_script, e_eager, rtol=0.0, atol=0.0)
