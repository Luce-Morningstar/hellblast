import cupy as cp
import numpy as np
import os

_kernel_cache = {}
def _load_raw_kernel(path, name, options=()):
    key = (path, name, options)
    if key in _kernel_cache: return _kernel_cache[key]
    with open(os.path.join(os.path.dirname(__file__), path), "r") as f:
        src = f.read()
    mod = cp.RawModule(code=src, options=options, name_expressions=(name,))
    fun = mod.get_function(name)
    _kernel_cache[key] = fun
    return fun

def _compute_legendre_block(x_cos, lmax, m):
    x_cos = cp.asarray(x_cos, dtype=cp.float64)
    Ntheta = x_cos.size
    Plm = cp.zeros((Ntheta, lmax+1), dtype=cp.float64)
    kern = _load_raw_kernel("legendre.cu", "plm_recurrence", options=("--std=c++11",))
    threads = 256
    blocks = (Ntheta + threads - 1) // threads
    kern((blocks,), (threads,), (x_cos, Ntheta, lmax, m, Plm))
    return Plm

def _alm_norm(l, m):
    from math import factorial, pi, sqrt
    return np.sqrt((2*l+1)/(4*np.pi) * float(factorial(l-m))/float(factorial(l+m)))

def map2alm_gpu(rings, thetas, nphi_per_ring, weights_dz, lmax, mmax=None):
    thetas = cp.asarray(thetas, dtype=cp.float64)
    nphi_per_ring = cp.asarray(nphi_per_ring, dtype=cp.int32)
    w = cp.asarray(weights_dz, dtype=cp.float64)
    if mmax is None: mmax = lmax
    Nr = int(thetas.size)
    Fm_list = []
    mmax_eff = 0
    for r in range(Nr):
        f_r = cp.asarray(rings[r], dtype=cp.float64)
        nphi = int(nphi_per_ring[r].get())
        F_r = cp.fft.fft(f_r) / nphi
        Fm_list.append(F_r)
        mmax_eff = max(mmax_eff, min(mmax, nphi//2))
    x = cp.cos(thetas)
    alm = cp.zeros(((lmax+1)*(lmax+1),), dtype=cp.complex128)
    for m in range(0, mmax_eff+1):
        Plm = _compute_legendre_block(x, lmax, m)  # [Nr, lmax+1]
        norms = cp.asarray([_alm_norm(l, m) for l in range(m, lmax+1)], dtype=cp.float64)
        Plm_block = Plm[:, m:(lmax+1)]
        wcol = w[:, None]
        Fm_vec = cp.asarray([Fm_list[r][m] if m < int(nphi_per_ring[r].get()) else 0.0+0.0j for r in range(Nr)],
                            dtype=cp.complex128)[:, None]
        accum = cp.sum(wcol * Plm_block * Fm_vec, axis=0)
        accum *= norms
        for idx_l, l in enumerate(range(m, lmax+1)):
            alm[l*(l+1) + m] = accum[idx_l]
    return alm
