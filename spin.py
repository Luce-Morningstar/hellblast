import cupy as cp
import numpy as np
from .sht import map2alm_gpu as map2alm_scalar
from .isht import alm2map_gpu as alm2map_scalar

_kern_cache = {}
def _get_fun(src, fun):
    key = (src, fun)
    if key in _kern_cache: return _kern_cache[key]
    import os
    here = os.path.dirname(__file__)
    with open(os.path.join(here, src), "r", encoding="utf-8") as f:
        code = f.read()
    mod = cp.RawModule(code=code, options=("--std=c++11",), name_expressions=(fun,))
    fn = mod.get_function(fun)
    _kern_cache[key] = fn
    return fn

def _wigner_d_ms(theta, lmax, m, s):
    theta = cp.asarray(theta, dtype=cp.float64)
    N = int(theta.size)
    out = cp.zeros((N, lmax+1), dtype=cp.float64)
    fn = _get_fun("wigner_d_ms.cu", "wigner_d_ms_sum")
    threads=256; blocks=(N+threads-1)//threads or 1
    fn((blocks,), (threads,), (theta, N, int(lmax), int(m), int(s), out))
    return out  # [N, lmax+1], valid for l>=max(|m|,|s|)

def _spin_norm(l):
    # Using _sY_lm(θ,φ) = sqrt((2l+1)/(4π)) d^l_{m,s}(θ) e^{imφ}
    return np.sqrt((2*l + 1)/(4.0*np.pi))

def map2alm_spin_gpu(Q_rings, U_rings, thetas, nphi_per_ring, weights_dz, lmax, mmax=None):
    """Compute spin-2 harmonics from Q/U rings on GPU.
    Returns a2_lm and a-2_lm (flattened l-major, m>=0).
    Caution: experimental normalization/sign; verify against a reference for your convention.
    """
    thetas = cp.asarray(thetas, dtype=cp.float64)
    nphi_per_ring = cp.asarray(nphi_per_ring, dtype=cp.int32)
    w = cp.asarray(weights_dz, dtype=cp.float64)
    Nr = int(thetas.size)
    if mmax is None: mmax = lmax

    P_rings = [cp.asarray(Q_rings[r], dtype=cp.float64) + 1j*cp.asarray(U_rings[r], dtype=cp.float64) for r in range(Nr)]
    M_rings = [cp.asarray(Q_rings[r], dtype=cp.float64) - 1j*cp.asarray(U_rings[r], dtype=cp.float64) for r in range(Nr)]

    # FFT along phi per ring
    PFm_list = []; MFm_list = []; mmax_eff = 0
    for r in range(Nr):
        nphi = int(nphi_per_ring[r].get())
        PF = cp.fft.fft(P_rings[r]) / nphi
        MF = cp.fft.fft(M_rings[r]) / nphi
        PFm_list.append(PF); MFm_list.append(MF)
        mmax_eff = max(mmax_eff, min(mmax, nphi//2))
    x = cp.cos(thetas)
    a2 = cp.zeros(((lmax+1)*(lmax+1),), dtype=cp.complex128)
    am2 = cp.zeros_like(a2)
    for m in range(0, mmax_eff+1):
        d_m_p2 = _wigner_d_ms(thetas, lmax, m, +2)  # [Nr, lmax+1]
        d_m_m2 = _wigner_d_ms(thetas, lmax, m, -2)
        norms = cp.asarray([_spin_norm(l) for l in range(0, lmax+1)], dtype=cp.float64)
        for l in range(max(m,2), lmax+1):
            # accumulate over rings
            dcol_p2 = d_m_p2[:, l]  # [Nr]
            dcol_m2 = d_m_m2[:, l]
            PFm = cp.asarray([PFm_list[r][m] if m < int(nphi_per_ring[r].get()) else 0.0+0.0j for r in range(Nr)])
            MFm = cp.asarray([MFm_list[r][m] if m < int(nphi_per_ring[r].get()) else 0.0+0.0j for r in range(Nr)])
            a2_val = cp.sum( w * dcol_p2 * PFm ) * norms[l] * (2*np.pi)
            am2_val = cp.sum( w * dcol_m2 * MFm ) * norms[l] * (2*np.pi)
            a2[l*(l+1)+m] = a2_val
            am2[l*(l+1)+m] = am2_val
    return a2, am2

def eb_from_qu_gpu(Q_rings, U_rings, thetas, nphi_per_ring, weights_dz, lmax, mmax=None):
    a2, am2 = map2alm_spin_gpu(Q_rings, U_rings, thetas, nphi_per_ring, weights_dz, lmax, mmax)
    # E = -(a2 + a-2)/2 ; B = (i/2) (a2 - a-2)
    E = -(a2 + am2) * 0.5
    B = (1j)*(a2 - am2) * 0.5
    return E, B

def qu_from_eb_gpu(Ealm, Balm, thetas, nphi_per_ring, lmax, mmax=None):
    # Build a2 and a-2 from E,B then synthesize P and M via iFFTs
    if mmax is None: mmax = lmax
    Nr = int(thetas.size)
    a2 = -(Ealm - 1j*Balm)
    am2 = -(Ealm + 1j*Balm)
    # Precompute d^l_{m,±2}(theta)
    d_p2 = {}; d_m2 = {}
    for m in range(0, mmax+1):
        d_p2[m] = _wigner_d_ms(thetas, lmax, m, +2)
        d_m2[m] = _wigner_d_ms(thetas, lmax, m, -2)
    P_rings = []; M_rings = []
    for r in range(Nr):
        nphi = int(nphi_per_ring[r].get())
        PF = cp.zeros((nphi,), dtype=cp.complex128)
        MF = cp.zeros_like(PF)
        for m in range(0, min(mmax, nphi-1)+1):
            accP = 0.0 + 0.0j
            accM = 0.0 + 0.0j
            for l in range(max(m,2), lmax+1):
                Nl = _spin_norm(l)
                accP += Nl * d_p2[m][r, l] * a2[l*(l+1)+m]
                accM += Nl * d_m2[m][r, l] * am2[l*(l+1)+m]
            PF[m] = accP
            if m>0: PF[nphi - m] = cp.conj(accP)
            MF[m] = accM
            if m>0: MF[nphi - m] = cp.conj(accM)
        P = cp.fft.ifft(PF) * nphi
        M = cp.fft.ifft(MF) * nphi
        Q = 0.5*(P + M)
        U = (-0.5j)*(P - M)
        P_rings.append(cp.asrealarray(Q))
        M_rings.append(cp.asrealarray(U))
    Q_out = [r.astype(cp.float64) for r in P_rings]
    U_out = [r.astype(cp.float64) for r in M_rings]
    return Q_out, U_out
