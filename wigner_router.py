import cupy as cp
from .wigner_fast import wigner_d_full as _fast_full
from .wigner_full import _d_l_mmprime as _exact_single

def D_slice(alpha, beta, gamma, l, method='auto'):
    # For now: 'auto' -> fast (tiled explicit) for l<=2048; fallback to exact single if needed
    d = _fast_full(cp.asarray([beta]), l, batch=1)[0].astype(cp.complex128)
    m = cp.arange(-l, l+1, dtype=cp.float64)
    A = cp.exp(-1j * m[:,None] * alpha)
    G = cp.exp(-1j * m[None,:] * gamma)
    return A * d * G
