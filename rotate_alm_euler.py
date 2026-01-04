import cupy as cp
from .wigner_router import D_slice
from .wigner_full import _expand_alm_pm, _compress_alm_pm

def rotate_alm_euler_fast(alm, alpha, beta, gamma, lmax=None):
    alm = cp.asarray(alm, dtype=cp.complex128)
    if lmax is None:
        lmax = int(cp.sqrt(alm.size).get()) - 1
    rows = _expand_alm_pm(alm, lmax)
    out_rows = []
    for l in range(0, lmax+1):
        D = D_slice(alpha, beta, gamma, l, method='auto')
        out_rows.append(D @ rows[l])
    return _compress_alm_pm(out_rows)
