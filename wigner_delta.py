import cupy as cp
import numpy as np
from .wigner_full import _d_l_mmprime  # explicit closed-form builder

class DeltaCache:
    def __init__(self):
        self.cache = {}  # (l)-> Δ_l (complex128)
    def get(self, l:int):
        if l in self.cache:
            return self.cache[l]
        beta = cp.asarray([np.pi/2], dtype=cp.float64)
        d = _d_l_mmprime(beta[0], l)  # float64, shape (2l+1,2l+1)
        Delta = d.astype(cp.complex128)  # Δ is real, but we use complex for uniformity
        self.cache[l] = Delta
        return Delta

_DELTA = DeltaCache()

def wigner_d_via_delta(beta, l:int):
    """Compute full d^l_{m m'}(beta) using Δ(π/2) factorization:
        d^l(β) = Δ^l · diag(e^{-i n β}) · (Δ^l)^T
    Returns complex128 (mathematically real; small imag from fp error).
    """
    beta = cp.asarray(beta, dtype=cp.float64)
    Delta = _DELTA.get(int(l))                         # (2l+1,2l+1)
    m = cp.arange(-l, l+1, dtype=cp.float64)          # also used for n
    phase = cp.exp(-1j * m * beta)                    # (2l+1,)
    # Δ * diag(phase) * Δ^T
    A = Delta * phase[None, :]                        # broadcast columns by phase
    d = A @ Delta.T                                   # (2l+1,2l+1)
    return d

def build_wignerD_delta(alpha, beta, gamma, lmax):
    """Full Wigner D using Δ-factorization: D^l = diag(e^{-i m α}) · d^l(β) · diag(e^{-i m γ})."""
    Ds = []
    for l in range(0, lmax+1):
        m = cp.arange(-l, l+1, dtype=cp.float64)
        A = cp.exp(-1j * m * alpha)[:, None]
        G = cp.exp(-1j * m * gamma)[None, :]
        d = wigner_d_via_delta(beta, l)
        Ds.append( (A * d) * G )
    return Ds

from .wigner_full import _expand_alm_pm, _compress_alm_pm

def rotate_alm_euler_risbo(alm, alpha, beta, gamma, lmax=None):
    """Rotate alm with Euler (ZYZ) using Δ(π/2) factorization per l."""
    alm = cp.asarray(alm, dtype=cp.complex128)
    if lmax is None:
        lmax = int(cp.sqrt(alm.size).get()) - 1
    rows = _expand_alm_pm(alm, lmax)
    out_rows = []
    for l in range(0, lmax+1):
        D = build_wignerD_delta(alpha, beta, gamma, l)[0]
        out_rows.append(D @ rows[l])
    return _compress_alm_pm(out_rows)
