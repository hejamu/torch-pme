import torch
from torch.nn.utils.rnn import pad_sequence


def get_ns_mesh(cell: torch.Tensor, mesh_spacing: float):
    """
    Computes the mesh size given a target mesh spacing and cell
    getting the closest powers of 2 to help with FFT.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param mesh_spacing: float
    :param differentiable: boll

    :return: torch.tensor of length 3 containing the mesh size
    """
    basis_norms = torch.linalg.norm(cell, dim=1)
    ns_approx = basis_norms / mesh_spacing
    ns_actual_approx = 2 * ns_approx + 1  # actual number of mesh points
    # ns = [nx, ny, nz], closest power of 2 (helps for FT efficiency).
    return torch.pow(2, torch.ceil(torch.log2(ns_actual_approx)).long())


def _kvectors_core(
    inverse_cell: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
    for_ewald: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Shared core of all reciprocal-space grid generators.

    Builds the reciprocal-space vectors from ``inverse_cell`` and the Python-int FFT
    sizes ``(nx, ny, nz)``. ``for_ewald`` selects a full ``fftfreq`` on the z-axis
    (explicit Ewald sum) versus the half-spectrum ``rfftfreq`` (FFT-based mesh). Taking
    the sizes as concrete ints keeps :func:`torch.fft.fftfreq` free of a data-dependent
    tensor argument, so the FFT shapes are static under ``torch.compile``.
    """
    reciprocal_cell = 2 * torch.pi * inverse_cell.T
    bx = reciprocal_cell[0]
    by = reciprocal_cell[1]
    bz = reciprocal_cell[2]

    # The frequencies from `fftfreq` are of the form [0, 1/n, 2/n, ...]; multiplying by
    # n turns them into [0, 1, 2, ...], which then scale the reciprocal-space vectors.
    kxs = (bx * nx) * torch.fft.fftfreq(nx, device=device, dtype=dtype).unsqueeze(-1)
    kys = (by * ny) * torch.fft.fftfreq(ny, device=device, dtype=dtype).unsqueeze(-1)
    if for_ewald:
        kzs = (bz * nz) * torch.fft.fftfreq(nz, device=device, dtype=dtype).unsqueeze(
            -1
        )
    else:
        kzs = (bz * nz) * torch.fft.rfftfreq(nz, device=device, dtype=dtype).unsqueeze(
            -1
        )

    # cartesian product via broadcasting (avoids materialising intermediates), summed up
    return kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]


def _generate_kvectors(
    cell: torch.Tensor, ns: torch.Tensor, for_ewald: bool
) -> torch.Tensor:
    # Check that all provided parameters have the correct shapes and are consistent
    # with each other
    if cell.shape != (3, 3):
        raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")

    if ns.shape != (3,):
        raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

    if ns.device != cell.device:
        raise ValueError(
            f"`ns` and `cell` are not on the same device, got {ns.device} and "
            f"{cell.device}."
        )

    if cell.is_cuda:
        # use function that does not synchronize with the CPU
        inverse_cell = torch.linalg.inv_ex(cell)[0]
    else:
        inverse_cell = torch.linalg.inv(cell)

    # This (tensor ``ns``) path is the stateful / Ewald one; extracting the mesh size as
    # ints here is acceptable since it is not the ``torch.compile`` fast path.
    return _kvectors_core(
        inverse_cell,
        int(ns[0]),
        int(ns[1]),
        int(ns[2]),
        for_ewald,
        cell.device,
        cell.dtype,
    )


def generate_kvectors_for_mesh_from_shape(
    cell: torch.Tensor,
    inverse_cell: torch.Tensor,
    ns_mesh: tuple[int, int, int],
) -> torch.Tensor:
    """
    ``torch.compile``-friendly variant of :func:`generate_kvectors_for_mesh`.

    Identical result to ``generate_kvectors_for_mesh(cell, ns)`` (for the FFT/mesh
    convention) but takes the mesh shape as a Python ``(int, int, int)`` tuple and the
    precomputed ``inverse_cell``. Because the FFT sizes are concrete Python ints,
    :func:`torch.fft.fftfreq` is not called with a data-dependent tensor argument, so no
    ``torch.compile`` graph break / CPU sync happens inside this function. Differentiable
    w.r.t. ``cell`` through ``inverse_cell``.

    :param cell: torch.tensor of shape ``(3, 3)`` (used only for dtype/device).
    :param inverse_cell: torch.tensor of shape ``(3, 3)``, the inverse of ``cell``.
    :param ns_mesh: mesh shape ``(nx, ny, nz)`` as Python ints.
    :return: torch.tensor of shape ``(nx, ny, nz // 2 + 1, 3)``.
    """
    nx, ny, nz = ns_mesh
    return _kvectors_core(inverse_cell, nx, ny, nz, False, cell.device, cell.dtype)


def generate_kvectors_for_mesh(cell: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
    """
    Compute all reciprocal space vectors for Fourier space sums.

    This variant is used in combination with **mesh based calculators** using the fast
    fourier transform (FFT) algorithm.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param ns: torch.tensor of shape ``(3,)`` and dtype int
        ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
        z-direction, respectively. For faster performance during the Fast Fourier
        Transform (FFT) it is recommended to use values of nx, ny and nz that are
        powers of 2.


    :return: torch.tensor of shape ``(nx, ny, nz, 3)`` containing all reciprocal
        space vectors that will be used in the (FFT-based) mesh calculators.
        Note that ``k_vectors[0,0,0] = [0,0,0]`` always is the zero vector.

    .. seealso::

        :func:`generate_kvectors_for_ewald` for a function to be used for Ewald
        calculators.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=False)


def generate_kvectors_for_ewald(
    cell: torch.Tensor,
    ns: torch.Tensor,
) -> torch.Tensor:
    """
    Compute all reciprocal space vectors for Fourier space sums.

    This variant is used with the **Ewald calculator**, in which the sum over the
    reciprocal space vectors is performed explicitly rather than using the fast Fourier
    transform (FFT) algorithm.

    The main difference with :func:`generate_kvectors_for_mesh` is the shape of the
    output tensor (see documentation on return) and the fact that the full set of
    reciprocal space vectors is returned, rather than the FFT-optimized set that roughly
    contains only half of the vectors.

    :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
        vector of the unit cell
    :param ns: torch.tensor of shape ``(3,)`` and dtype int
        ``ns = [nx, ny, nz]`` contains the number of mesh points in the x-, y- and
        z-direction, respectively.

    :return: torch.tensor of shape ``(n, 3)`` containing all reciprocal
        space vectors that will be used in the Ewald calculator.
        Note that ``k_vectors[0] = [0,0,0]`` always is the zero vector.

    .. seealso::

        :func:`generate_kvectors_for_mesh` for a function to be used with mesh based
        calculators like PME.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=True).reshape(-1, 3)


def compute_batched_kvectors(
    lr_wavelength: float,
    cells: torch.Tensor,
) -> torch.Tensor:
    r"""
    Generate k-vectors for multiple systems in batches.

    :param lr_wavelength: Spatial resolution used for the long-range (reciprocal space)
        part of the Ewald sum. More concretely, all Fourier space vectors with a
        wavelength >= this value will be kept. If not set to a global value, it will be
        set to half the smearing parameter to ensure convergence of the
        long-range part to a relative precision of 1e-5.
    :param cell: torch.tensor of shape ``(B, 3, 3)``, where ``cell[i]`` is the i-th
        basis vector of the unit cell for system i in the batch of size B.

    """
    all_kvectors = []
    k_cutoff = 2 * torch.pi / lr_wavelength
    for cell in cells:
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()
        kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)
        all_kvectors.append(kvectors)
    # We do not return masks here; instead, we rely on the fact that for the Coulomb
    # potential, the k = 0 vector is ignored in the calculations and can therefore be
    # safely padded with zeros.
    return pad_sequence(all_kvectors, batch_first=True)
