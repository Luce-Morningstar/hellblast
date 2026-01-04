import cupy as cp
import numpy as np

def _alm_norm(l, m):
    # Condon-Shortley phase not included since we use standard HEALPy normalisation
    from math import factorial, sqrt, pi
    # 4π normalization for spherical harmonics
    return np.sqrt((2*l + 1)/(4*np.pi) * float(np.math.factorial(l-m))/float(np.math.factorial(l+m)))

def _compute_legendre_block(x, lmax, m):
    # Compute P_l^m(x) for l=m..lmax for all x in vector x, return [len(x), lmax+1] with columns 0..lmax (unused below m filled with 0)
    x = cp.asarray(x, dtype=cp.float64)
    nx = x.size
    P = cp.zeros((nx, lmax+1), dtype=cp.float64)
    # Starting with P_m^m
    mm = m
    # P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    if mm>0:
        fact = 1.0
        for k in range(1, 2*mm, 2):
            fact *= k
        Pmm = ((-1)**mm) * fact * cp.power(1.0 - x*x, 0.5*mm)
    else:
        Pmm = cp.ones_like(x)
    P[:, mm] = Pmm
    if mm==lmax:
        return P
    # P_{m+1}^m = x (2m+1) P_m^m
    Pmp1m = x * (2*mm + 1) * Pmm
    P[:, mm+1] = Pmp1m
    for l in range(mm+2, lmax+1):
        Plm = ( (2*l-1)*x*P[:, l-1] - (l+mm-1)*P[:, l-2] )/(l-mm)
        P[:, l] = Plm
    return P

def map2alm_gpu(rings, thetas, nphi_per_ring, weights_dz, lmax, mmax=None):
    """
    rings: list of CuPy arrays, one per ring (phi-major, length nphi_r)
    thetas: CuPy array [Nr]
    nphi_per_ring: CuPy int32 array [Nr]
    weights_dz: CuPy float64 array [Nr]  (Δz per ring; sum to 2)
    Returns alm flattened (l-major), complex128
    """
    thetas = cp.asarray(thetas, dtype=cp.float64)
    nphi_per_ring = cp.asarray(nphi_per_ring, dtype=cp.int32)
    w = cp.asarray(weights_dz, dtype=cp.float64)
    if mmax is None: mmax = lmax
    Nr = int(thetas.size)
    # FFT along phi for each ring
    Fm = []  # list of arrays of length mmax+1 with Fourier coeffs at each ring
    for r in range(Nr):
        f = cp.asarray(rings[r], dtype=cp.float64)
        nphi = int(nphi_per_ring[r].get())
        F = cp.fft.fft(f)/nphi  # unitary consistent with Δφ=2π/nphi
        Fm.append([F[m] for m in range(min(mmax, nphi-1)+1)])
    # build alm
    alm = cp.zeros(((lmax+1)*(lmax+1),), dtype=cp.complex128)
    x = cp.cos(thetas)
    for m in range(0, mmax+1):
        Pl = _compute_legendre_block(x, lmax, m)  # [Nr, lmax+1]
        for l in range(m, lmax+1):
            # integral over z using weights; over phi already done -> 2π*Fm
            val = cp.sum( (2*np.pi)*cp.asarray([Fm[r][m] if m < len(Fm[r]) else 0.0+0.0j for r in range(Nr)]) * Pl[:, l] * w )
            alm[l*(l+1)+m] = _alm_norm(l, m) * val
    return alm
