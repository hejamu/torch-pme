from typing import Optional

import torch

from ..lib.kspace_filter import KSpaceFilter
from ..lib.kvectors import generate_kvectors_for_mesh_from_shape, get_ns_mesh
from ..lib.mesh_interpolator import MeshInterpolator
from ..potentials import Potential
from .calculator import Calculator


class PMECalculator(Calculator):
    r"""
    Potential using a particle mesh-based Ewald (PME).

    Scaling as :math:`\mathcal{O}(NlogN)` with respect to the number of particles
    :math:`N` used as a reference to test faster implementations.

    For getting reasonable values for the ``smaring`` of the potential class and  the
    ``mesh_spacing`` based on a given accuracy for a specific structure you should use
    :func:`torchpme.tuning.tune_pme`. This function will also find the optimal
    ``cutoff`` for the  **neighborlist**.

    .. hint::

        For a training exercise it is recommended only run a tuning procedure with
        :func:`torchpme.tuning.tune_pme` for the largest system in your dataset.

    :param potential: A :class:`torchpme.potentials.Potential` object that implements
        the evaluation of short and long-range potential terms. The ``smearing``
        parameter of the potential determines the split between real and k-space
        regions. For a :class:`torchpme.CoulombPotential` it corresponds to the
        smearing of the atom-centered Gaussian used to split the Coulomb potential into
        the short- and long-range parts. A reasonable value for most systems is to set
        it to ``1/5`` times the neighbor list cutoff.
    :param mesh_spacing: Value that determines the umber of Fourier-space grid points
        that will be used along each axis. If set to None, it will automatically be set
        to half of ``smearing``.
    :param interpolation_nodes: The number ``n`` of nodes used in the interpolation per
        coordinate axis. The total number of interpolation nodes in 3D will be ``n^3``.
        In general, for ``n`` nodes, the interpolation will be performed by piecewise
        polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic interpolation).
        Only the values ``3, 4, 5, 6, 7`` are supported.
    :param full_neighbor_list: If set to :obj:`True`, a "full" neighbor list
        is expected as input. This means that each atom pair appears twice. If
        set to :obj:`False`, a "half" neighbor list is expected.
    """

    # TorchScript requires class-level annotations for non-Tensor list attributes.
    _fixed_ns_mesh: list[int]

    def __init__(
        self,
        potential: Potential,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        full_neighbor_list: bool = False,
        ns_mesh: Optional[tuple[int, int, int]] = None,
    ):
        super().__init__(potential=potential, full_neighbor_list=full_neighbor_list)

        if potential.smearing is None:
            raise ValueError(
                "Must specify smearing to use a potential with PMECalculator"
            )
        if potential.smearing <= 0:
            raise ValueError(f"`smearing` is {potential.smearing} but must be positive")

        self.mesh_spacing: float = mesh_spacing

        # When set, ``_fixed_ns_mesh`` is a length-3 list of ints that bypasses the
        # per-forward ``get_ns_mesh`` CPU sync entirely. Dynamo specialises the list
        # values as constants → 0 graph breaks → one clean CUDA graph. The cell-derived
        # reciprocal quantities (kvectors, filter, weights) are still recomputed each
        # forward, so autograd on ``cell`` is fully preserved. When empty the adaptive
        # path is used (a single ``tolist`` sync per forward). Use ``list[int]`` rather
        # than ``tuple`` for TorchScript compatibility.
        self._fixed_ns_mesh: list[int] = list(ns_mesh) if ns_mesh is not None else []

        cell = torch.eye(
            3,
            device=self.potential.smearing.device,
            dtype=self.potential.smearing.dtype,
        )
        ns_mesh = torch.ones(3, dtype=int, device=cell.device)

        self.kspace_filter: KSpaceFilter = KSpaceFilter(
            cell=cell,
            ns_mesh=ns_mesh,
            kernel=self.potential,
            fft_norm="backward",
            ifft_norm="forward",
        )

        self.mesh_interpolator: MeshInterpolator = MeshInterpolator(
            cell=cell,
            ns_mesh=ns_mesh,
            interpolation_nodes=interpolation_nodes,
            method="Lagrange",  # convention for classic PME
        )
        self.interpolation_nodes: int = interpolation_nodes

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        periodic: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        kvectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Kernel function `G` and initialization of `MeshInterpolator` only depend
        # on `cell`. Caching may save up to 15% but issues with AD need to be resolved.
        #
        # NB: this is NOT that caching change. Every cell-derived quantity below is
        # recomputed each forward, exactly as before; we only compute them as local
        # variables instead of writing them onto `self`. Removing the per-forward state
        # mutation (and the `int(self.ns_mesh[i])` CPU syncs, now a single sync feeding a
        # static `ns_mesh` tuple) lets the whole forward be captured by `torch.compile`
        # without graph breaks and makes it compatible with `mode="reduce-overhead"`
        # (CUDA Graphs). AD on `cell` is preserved: each local stays a differentiable
        # function of `cell`.
        if node_mask is not None or kvectors is not None:
            raise NotImplementedError(
                "Batching not implemented for mesh-based calculators"
            )

        # Mesh shape as a Python int triple for the pure helpers.
        # Fixed path: ns_mesh is a compile-time constant → 0 graph breaks, enabling
        # one clean CUDA graph under mode="reduce-overhead".
        # Adaptive path: one tolist() sync per forward (3 data-dependent breaks).
        if len(self._fixed_ns_mesh) == 3:
            ns_mesh = (
                self._fixed_ns_mesh[0],
                self._fixed_ns_mesh[1],
                self._fixed_ns_mesh[2],
            )
            ns = torch.tensor(ns_mesh, device=cell.device, dtype=torch.long)
        else:
            ns = get_ns_mesh(cell, self.mesh_spacing)
            ns_list: list[int] = ns.tolist()
            ns_mesh = (ns_list[0], ns_list[1], ns_list[2])

        if cell.is_cuda:
            # use a routine that does not synchronize with the CPU
            inverse_cell = torch.linalg.inv_ex(cell)[0]
        else:
            inverse_cell = torch.linalg.inv(cell)

        # Reciprocal-space grid and filter (depend only on `cell`). The int-shaped
        # kvector generator uses Python ints for the FFT sizes, so no extra CPU sync /
        # graph break happens here.
        kvectors_mesh = generate_kvectors_for_mesh_from_shape(
            cell, inverse_cell, ns_mesh
        )
        k_sq = torch.linalg.norm(kvectors_mesh, dim=3) ** 2
        kfilter = self.potential.kernel_from_k_sq(k_sq)

        # Forward interpolation: particles -> mesh (no state written to `self`).
        (
            interpolation_weights,
            x_shifts,
            y_shifts,
            z_shifts,
            x_indices,
            y_indices,
            z_indices,
        ) = self.mesh_interpolator.compute_weights_pure(positions, inverse_cell, ns)
        rho_mesh = self.mesh_interpolator.points_to_mesh_pure(
            charges,
            interpolation_weights,
            x_shifts,
            y_shifts,
            z_shifts,
            x_indices,
            y_indices,
            z_indices,
            ns_mesh,
        )

        potential_mesh = self.kspace_filter.apply_filter(rho_mesh, kfilter, ns_mesh)

        ivolume = torch.abs(cell.det()).pow(-1)
        interpolated_potential = (
            self.mesh_interpolator.mesh_to_points_pure(
                potential_mesh,
                interpolation_weights,
                x_shifts,
                y_shifts,
                z_shifts,
                x_indices,
                y_indices,
                z_indices,
            )
            * ivolume
        )

        # Using the Coulomb potential as an example, this is the potential generated
        # at the origin by the fictituous Gaussian charge density in order to split
        # the potential into a SR and LR part. This contribution always should be
        # subtracted since it depends on the smearing parameter, which is purely a
        # convergence parameter.
        interpolated_potential -= charges * self.potential.self_contribution()

        # If the cell has a net charge (i.e. if sum(charges) != 0), the method
        # implicitly assumes that a homogeneous background charge of the opposite
        # sign is present to make the cell neutral. In this case, the potential has
        # to be adjusted to compensate for this. An extra factor of 2 is added to
        # compensate for the division by 2 later on
        charge_tot = torch.sum(charges, dim=0)
        prefac = self.potential.background_correction()
        interpolated_potential -= 2 * prefac * charge_tot * ivolume

        interpolated_potential += self.potential.pbc_correction(
            periodic, positions, cell, charges
        )

        # Compensate for double counting of pairs (i,j) and (j,i)
        return interpolated_potential / 2
