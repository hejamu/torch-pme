"""Tests for `kspace_filter` classes"""

import pytest
import torch

from torchpme.lib import (
    KSpaceFilter,
    KSpaceKernel,
    MeshInterpolator,
    generate_kvectors_for_mesh,
)


class TestKernel:
    class DemoKernel(KSpaceKernel):
        def __init__(self, param: float):
            super().__init__()
            self.param = param

        @torch.jit.export
        def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
            return torch.exp(-k_sq / self.param)

    class NoopKernel(KSpaceKernel):
        def __init__(self):
            super().__init__()

        @torch.jit.export
        def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(k_sq)

    def test_kernel_subclassing(self):
        # check that one can define and use a kernel
        my_krn = self.DemoKernel(1.0)
        k_sq = torch.arange(0, 10, 0.01)

        my_krn.kernel_from_k_sq(k_sq)

    def test_kernel_jitting(self):
        # pytorch
        my_krn = self.DemoKernel(1.0)
        k_sq = torch.arange(0, 10, 0.01)
        filter = my_krn.kernel_from_k_sq(k_sq)

        # jitted
        jit_krn = torch.jit.script(my_krn)
        jit_filter = jit_krn.kernel_from_k_sq(k_sq)

        assert torch.allclose(filter, jit_filter)


class TestFilter:
    cell1 = torch.randn((3, 3))
    cell2 = torch.randn((3, 3))
    ns1 = torch.tensor([3, 4, 5])
    ns2 = torch.tensor([4, 2, 1])

    mykernel = TestKernel.DemoKernel(1.0)
    myfilter1 = KSpaceFilter(cell1, ns1, mykernel)
    myfilter2 = KSpaceFilter(cell2, ns2, mykernel)
    mymesh1 = MeshInterpolator(cell1, ns1, 3, method="P3M")
    mymesh2 = MeshInterpolator(cell2, ns2, 3, method="Lagrange")
    points = torch.tensor([[1.0, 2, 3], [0, 1, 1]])
    weights = torch.tensor([[-0.1], [0.4]])

    mykernel_noop = TestKernel.NoopKernel()
    myfilter_noop = KSpaceFilter(cell1, ns1, mykernel_noop)

    def test_meshes_consistent_size(self):
        # make sure we get conistent mesh sizes
        self.mymesh1.compute_weights(self.points)
        mesh = self.mymesh1.points_to_mesh(self.weights)
        # nb - the third value is different because of the real-valued FT
        assert mesh.shape[1:3] == self.myfilter1._kvectors.shape[:-2]

    def test_meshes_inconsistent_size(self):
        # make sure we get consistent mesh sizes
        self.mymesh1.compute_weights(self.points)
        mesh = self.mymesh1.points_to_mesh(self.weights)
        match = "The real-space mesh is inconsistent with the k-space grid."
        with pytest.raises(ValueError, match=match):
            self.myfilter2.forward(mesh)

    def test_kernel_noop(self):
        # make sure that a filter of ones recovers the initial mesh
        self.mymesh1.compute_weights(self.points)
        mesh = self.mymesh1.points_to_mesh(self.weights)
        mesh_transformed = self.myfilter_noop.forward(mesh)

        torch.allclose(mesh, mesh_transformed, atol=1e-6, rtol=0)

    def test_filter_linear(self):
        # checks that the filter (as well as the mesh interpolator) are linear
        self.mymesh1.compute_weights(self.points)
        mesh1 = self.mymesh1.points_to_mesh(self.weights)

        mesh2 = torch.exp(mesh1)

        tmesh1 = self.myfilter1.forward(mesh1)
        tmesh2 = self.myfilter1.forward(mesh2)
        tmesh12 = self.myfilter1.forward(mesh1 + 0.3 * mesh2)

        torch.allclose(tmesh12, tmesh1 + 0.3 * tmesh2)


@pytest.mark.parametrize("cell_update", [None, torch.eye(3)])
@pytest.mark.parametrize("ns_mesh_update", [None, torch.tensor([3, 3, 3])])
def test_update(cell_update, ns_mesh_update):
    cell = 2 * torch.eye(3)
    ns_mesh = torch.tensor([2, 2, 2])

    kernel = TestKernel.DemoKernel(1.0)
    kernel_filter = KSpaceFilter(cell=cell, ns_mesh=ns_mesh, kernel=kernel)

    # update param of demo kernel and check if updates are consistent.
    kernel.param = 2.0

    kernel_filter.update(cell=cell_update, ns_mesh=ns_mesh_update)

    if cell_update is not None:
        assert torch.all(kernel_filter.cell == cell_update)

    if ns_mesh_update is not None:
        assert torch.all(kernel_filter.ns_mesh == ns_mesh_update)

    if cell_update is not None and ns_mesh_update is not None:
        kvectors = generate_kvectors_for_mesh(ns=ns_mesh_update, cell=cell_update)
        k_sq = torch.linalg.norm(kvectors, dim=3) ** 2

        torch.testing.assert_close(kernel_filter._k_sq, k_sq)

    torch.testing.assert_close(
        kernel_filter._kfilter, kernel.kernel_from_k_sq(kernel_filter._k_sq)
    )


def test_update_ns_wrong_shape():
    kernel_filter = KSpaceFilter(
        cell=torch.eye(3),
        ns_mesh=torch.tensor([2, 2, 2]),
        kernel=TestKernel.DemoKernel(1.0),
    )

    match = "shape \\[2\\] of `ns_mesh` has to be \\(3,\\)"
    with pytest.raises(ValueError, match=match):
        kernel_filter.update(ns_mesh=torch.tensor([2, 2]))


def test_update_cell_wrong_shape():
    kernel_filter = KSpaceFilter(
        cell=torch.eye(3),
        ns_mesh=torch.tensor([2, 2, 2]),
        kernel=TestKernel.DemoKernel(1.0),
    )

    match = "cell of shape \\[2, 3\\] should be of shape \\(3, 3\\)"
    with pytest.raises(ValueError, match=match):
        kernel_filter.update(cell=torch.tensor([[1.0, 0, 0], [0, 1, 0]]))


def test_update_devices_ns_cell():
    kernel_filter = KSpaceFilter(
        cell=torch.eye(3),
        ns_mesh=torch.tensor([2, 2, 2]),
        kernel=TestKernel.DemoKernel(1.0),
    )

    match = "`cell` and `ns_mesh` are on different devices, got meta and cpu"
    with pytest.raises(ValueError, match=match):
        kernel_filter.update(cell=torch.eye(3, device="meta"))


def test_fft_modes():
    ns = torch.tensor([2, 2, 2], device="cpu")
    cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], device="cpu")
    match = "Invalid option 'faster' for the `fft_norm` parameter."
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel(), fft_norm="faster")
    match = "Invalid option 'faster' for the `ifft_norm` parameter."
    with pytest.raises(ValueError, match=match):
        KSpaceFilter(cell, ns, KSpaceKernel(), ifft_norm="faster")
@pytest.mark.parametrize("ns_mesh", [(8, 8, 8), (9, 8, 7), (7, 6, 5), (10, 10, 10)])
@pytest.mark.parametrize("n_channels", [1, 2])
def test_filter_conv_matches_generic_autograd(ns_mesh, n_channels):
    """The fast self-adjoint ``apply_filter`` path agrees with generic FFT autograd.

    Checks forward values and gradients w.r.t. both the mesh and the (cell-dependent)
    filter against the plain ``irfftn(rfftn(x) * f)`` reference differentiated by
    PyTorch's generic FFT autograd.
    """
    from torchpme.lib.kspace_filter import _filter_conv

    nx, ny, nz = ns_mesh
    torch.manual_seed(nx * 100 + ny * 10 + nz + n_channels)
    x = torch.randn(n_channels, nx, ny, nz, dtype=torch.float64)
    f = torch.rand(nx, ny, nz // 2 + 1, dtype=torch.float64)
    g = torch.randn(n_channels, nx, ny, nz, dtype=torch.float64)

    def reference(x_, f_):
        h = torch.fft.rfftn(x_, norm="backward", dim=[1, 2, 3])
        return torch.fft.irfftn(h * f_, norm="forward", dim=[1, 2, 3], s=[nx, ny, nz])

    xf = x.clone().requires_grad_(True)
    ff = f.clone().requires_grad_(True)
    out_fast = _filter_conv(xf, ff, nx, ny, nz)
    out_fast.backward(g)

    xr = x.clone().requires_grad_(True)
    fr = f.clone().requires_grad_(True)
    out_ref = reference(xr, fr)
    out_ref.backward(g)

    torch.testing.assert_close(out_fast, out_ref, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(xf.grad, xr.grad, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(ff.grad, fr.grad, atol=1e-10, rtol=0.0)


def test_filter_conv_gradcheck():
    """`torch.autograd.gradcheck` on the custom k-space convolution operator."""
    from torchpme.lib.kspace_filter import _filter_conv

    nx, ny, nz = 6, 5, 4
    torch.manual_seed(0)
    x = torch.randn(2, nx, ny, nz, dtype=torch.float64, requires_grad=True)
    f = torch.rand(nx, ny, nz // 2 + 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x_, f_: _filter_conv(x_, f_, nx, ny, nz), (x, f), atol=1e-6, rtol=1e-4
    )
