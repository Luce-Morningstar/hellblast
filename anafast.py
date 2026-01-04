import cupy as cp
from .sht import map2alm_gpu
from .alm_ops import alm2cl

def anafast_ring(rings, thetas, nphi_per_ring, weights_dz, lmax):
    alm = map2alm_gpu(rings, thetas, nphi_per_ring, weights_dz, lmax)
    return alm2cl(alm, lmax=lmax)
