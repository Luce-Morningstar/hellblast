import cupy as cp
import numpy as np

# log-factorials for stability
_logfac_cache = {0:0.0}
def _logfac(n):
    if n in _logfac_cache: return _logfac_cache[n]
    val = float(cp.math.lgamma(n+1.0))
    _logfac_cache[n]=val
    return val

def _d_l_mmprime(beta, l):
    # Build full (2l+1)x(2l+1) d^l_{m m'}(beta), m,m'=-l..l
    ct = cp.cos(0.5*beta)
    st = cp.sin(0.5*beta)
    mvals = cp.arange(-l, l+1, dtype=cp.int32)
    d = cp.empty((2*l+1, 2*l+1), dtype=cp.float64)
    # vectorized over m,m' via for-loops (l is modest in practice)
    for im, m in enumerate(range(-l, l+1)):
        for ip, mp in enumerate(range(-l, l+1)):
            # sum over k
            kmin = max(0, m - mp)
            kmax = min(l+m, l - mp)
            # Precompute sqrt factorial prefactor in log domain
            logpref = 0.5*( _logfac(l+m) + _logfac(l-m) + _logfac(l+mp) + _logfac(l-mp) )
            s = 0.0
            for k in range(kmin, kmax+1):
                a = l + mp - k
                b = m - mp + 2*k
                c = l - m - k
                if a<0 or b<0 or c<0: 
                    continue
                logden = _logfac(l+m-k) + _logfac(k) + _logfac(mp-m+k) + _logfac(l-mp-k)
                # (-1)^{k + mp - m}
                sign = -1.0 if ((k + mp - m) & 1) else 1.0
                coeff = cp.exp(logpref - logden)
                term = sign * coeff * (ct**a) * (st**b)
                s += term
            d[im, ip] = s
    return d

def build_wignerD_slices(alpha, beta, gamma, lmax):
    # Returns list of D^l matrices as complex128, each shape (2l+1, 2l+1)
    ca = cp.exp(-1j * cp.arange(-lmax, lmax+1, dtype=cp.float64) * alpha)  # length 2lmax+1 superset
    cg = cp.exp(-1j * cp.arange(-lmax, lmax+1, dtype=cp.float64) * gamma)
    slices = []
    for l in range(0, lmax+1):
        d = _d_l_mmprime(beta, l)  # float64
        m = cp.arange(-l, l+1, dtype=cp.int32)
        # phase factors
        A = cp.exp(-1j * m[:,None] * alpha)   # (2l+1,1)
        G = cp.exp(-1j * m[None,:] * gamma)   # (1,2l+1)
        D = A * d.astype(cp.complex128) * G
        slices.append(D)
    return slices

def _expand_alm_pm(alm, lmax):
    # Convert half-stored alm(m>=0) to full vector per l with m=-l..l using symmetry for real maps.
    # Here we make no assumption; we reconstruct a full (2l+1) with negative m filled by conjugation.
    full = []
    for l in range(0, lmax+1):
        row = cp.empty((2*l+1,), dtype=cp.complex128)
        # positive m (including 0)
        pos = cp.asarray([alm[l*(l+1)+m] for m in range(0, l+1)], dtype=cp.complex128)
        row[l:] = pos  # indices m=0..l
        # negative m via Condon-Shortley: a_{l,-m} = (-1)^m * conj(a_{l,m})
        neg = cp.asarray([ ((-1)**m)*cp.conj(alm[l*(l+1)+m]) for m in range(1, l+1) ], dtype=cp.complex128)
        row[:l] = neg[::-1]  # m=-l..-1
        full.append(row)
    return full  # list length lmax+1

def _compress_alm_pm(full_rows):
    # Convert per-l full m vector back to half-storage (m>=0)
    out = []
    for l, row in enumerate(full_rows):
        out.extend(row[l:])  # m=0..l
    return cp.asarray(out, dtype=cp.complex128)

def rotate_alm_euler(alm, alpha, beta, gamma, lmax=None):
    # Rotate alm by Euler angles (ZYZ convention): D^l(α,β,γ)
    alm = cp.asarray(alm, dtype=cp.complex128)
    if lmax is None:
        lmax = int(cp.sqrt(alm.size).get()) - 1
    # expand to full m for each l
    rows = _expand_alm_pm(alm, lmax)
    # build D slices
    Ds = build_wignerD_slices(alpha, beta, gamma, lmax)
    # apply per l: a'_l = D^l a_l
    out_rows = []
    for l in range(0, lmax+1):
        a = rows[l]                  # (2l+1,)
        D = Ds[l]                    # (2l+1, 2l+1)
        ap = D @ a                   # GEMM-like multiply
        out_rows.append(ap)
    # compress back to m>=0 storage
    return _compress_alm_pm(out_rows)
