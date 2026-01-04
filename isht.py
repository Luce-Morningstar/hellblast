import cupy as cp
import numpy as np
from .sht import _alm_norm, _compute_legendre_block

def alm2map_gpu(alm, thetas, nphi_per_ring, lmax, mmax=None, dtype=cp.float64):
    thetas = cp.asarray(thetas, dtype=cp.float64)
    nphi_per_ring = cp.asarray(nphi_per_ring, dtype=cp.int32)
    Nr = int(thetas.size)
    if mmax is None: mmax = lmax
    x = cp.cos(thetas)
    Plm_blocks = {}
    for m in range(0, mmax+1):
        Plm_blocks[m] = _compute_legendre_block(x, lmax, m)[:, m:(lmax+1)]
    rings = []
    for r in range(Nr):
        nphi = int(nphi_per_ring[r].get())
        F = cp.zeros((nphi,), dtype=cp.complex128)
        for m in range(0, min(mmax, nphi-1)+1):
            norms = cp.asarray([_alm_norm(l, m) for l in range(m, lmax+1)], dtype=cp.float64)
            Pl = Plm_blocks[m][r]
            coeff = cp.sum( norms * Pl * cp.asarray([alm[l*(l+1)+m] for l in range(m, lmax+1)], dtype=cp.complex128) )
            F[m] = coeff
            if m>0: F[nphi - m] = cp.conj(coeff)
        f = cp.fft.ifft(F) * nphi
        rings.append(cp.asrealarray(f).astype(dtype))
    return rings
