import cupy as cp
import numpy as np
from .spin import _get_fun, _wigner_d_ms, _spin_norm  # reuse helpers from spin.py

def map2alm_spin1_gpu(X_rings, Y_rings, thetas, nphi_per_ring, weights_dz, lmax, mmax=None):
    thetas = cp.asarray(thetas, dtype=cp.float64)
    nphi_per_ring = cp.asarray(nphi_per_ring, dtype=cp.int32)
    w = cp.asarray(weights_dz, dtype=cp.float64)
    Nr = int(thetas.size)
    if mmax is None: mmax = lmax

    P_rings = [cp.asarray(X_rings[r], dtype=cp.float64) + 1j*cp.asarray(Y_rings[r], dtype=cp.float64) for r in range(Nr)]
    M_rings = [cp.asarray(X_rings[r], dtype=cp.float64) - 1j*cp.asarray(Y_rings[r], dtype=cp.float64) for r in range(Nr)]

    PFm_list = []; MFm_list = []; mmax_eff = 0
    for r in range(Nr):
        nphi = int(nphi_per_ring[r].get())
        PF = cp.fft.fft(P_rings[r]) / nphi
        MF = cp.fft.fft(M_rings[r]) / nphi
        PFm_list.append(PF); MFm_list.append(MF)
        mmax_eff = max(mmax_eff, min(mmax, nphi//2))

    a1 = cp.zeros(((lmax+1)*(lmax+1),), dtype=cp.complex128)
    am1 = cp.zeros_like(a1)
    for m in range(0, mmax_eff+1):
        d_m_p1 = _wigner_d_ms(thetas, lmax, m, +1)  # [Nr, lmax+1]
        d_m_m1 = _wigner_d_ms(thetas, lmax, m, -1)
        for l in range(max(m,1), lmax+1):
            Nl = _spin_norm(l)
            dcol_p1 = d_m_p1[:, l]  # [Nr]
            dcol_m1 = d_m_m1[:, l]
            PFm = cp.asarray([PFm_list[r][m] if m < int(nphi_per_ring[r].get()) else 0.0+0.0j for r in range(Nr)])
            MFm = cp.asarray([MFm_list[r][m] if m < int(nphi_per_ring[r].get()) else 0.0+0.0j for r in range(Nr)])
            val_p1 = cp.sum( w * dcol_p1 * PFm ) * Nl * (2*np.pi)
            val_m1 = cp.sum( w * dcol_m1 * MFm ) * Nl * (2*np.pi)
            a1[l*(l+1)+m]  = val_p1
            am1[l*(l+1)+m] = val_m1
    return a1, am1

def alm2map_spin1_gpu(a1, am1, thetas, nphi_per_ring, lmax, mmax=None):
    if mmax is None: mmax = lmax
    Nr = int(thetas.size)
    d_p1 = {}; d_m1 = {}
    for m in range(0, mmax+1):
        d_p1[m] = _wigner_d_ms(thetas, lmax, m, +1)
        d_m1[m] = _wigner_d_ms(thetas, lmax, m, -1)
    X_rings = []; Y_rings = []
    for r in range(Nr):
        nphi = int(nphi_per_ring[r].get())
        PF = cp.zeros((nphi,), dtype=cp.complex128)
        MF = cp.zeros_like(PF)
        for m in range(0, min(mmax, nphi-1)+1):
            accP = 0.0 + 0.0j
            accM = 0.0 + 0.0j
            for l in range(max(m,1), lmax+1):
                Nl = _spin_norm(l)
                accP += Nl * d_p1[m][r, l] * a1[l*(l+1)+m]
                accM += Nl * d_m1[m][r, l] * am1[l*(l+1)+m]
            PF[m] = accP
            if m>0: PF[nphi - m] = cp.conj(accP)
            MF[m] = accM
            if m>0: MF[nphi - m] = cp.conj(accM)
        P = cp.fft.ifft(PF) * nphi
        M = cp.fft.ifft(MF) * nphi
        X = 0.5*(P + M)
        Y = (-0.5j)*(P - M)
        X_rings.append(cp.asrealarray(X).astype(cp.float64))
        Y_rings.append(cp.asrealarray(Y).astype(cp.float64))
    return X_rings, Y_rings
