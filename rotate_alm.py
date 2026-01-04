import cupy as cp
import numpy as np

def rotate_alm_z(alm, alpha, lmax=None):
    """Rotate alm by a z-axis angle alpha (radians).
    a_{l m} -> a_{l m} * exp(-i m alpha)
    Works for m>=0 storage with conjugate symmetry assumed externally for real maps.
    """
    alm = cp.asarray(alm, dtype=cp.complex128)
    if lmax is None:
        lmax = int(cp.sqrt(alm.size).get()) - 1
    mphase = cp.exp(-1j * cp.arange(lmax+1, dtype=cp.float64) * alpha)
    out = alm.copy()
    for l in range(lmax+1):
        base = l*(l+1)
        # m = 0..l
        out[base:base+(l+1)] *= mphase[:(l+1)]
    return out
