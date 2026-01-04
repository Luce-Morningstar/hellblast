import cupy as cp
import numpy as np
from .ring_tables import HealpixRingTables
from .nested_gpu import ang2pix_nest_gpu

_cache = {}

def _build_cache_for_nside(nside:int):
    # Build on device: for each HEALPix pixel center in RING geometry, compute its NESTED id,
    # then store its unit vector (x,y,z) indexed by ipnest. Everything stays on GPU.
    tabs = HealpixRingTables(nside)
    Nr = int(tabs.Nr)
    npix = int(tabs.npix)
    theta_all = cp.empty((npix,), dtype=cp.float64)
    phi_all   = cp.empty((npix,), dtype=cp.float64)
    # fill θ,φ for every pixel center in RING
    for r in range(Nr):
        npr = int(tabs.nphi[r].get())
        base = int(tabs.ring_start[r].get())
        dphi = 2*np.pi / npr
        pc0  = tabs.phi_center0[r]
        # centers at pc0 + j*dphi
        j = cp.arange(npr, dtype=cp.float64)
        phi_all[base:base+npr] = pc0 + j * dphi
        theta_all[base:base+npr] = tabs.theta_centers[r]
    # map to NESTED ids using pure-GPU ang2pix_nest
    ipnest = ang2pix_nest_gpu(theta_all, phi_all, nside)
    # construct unit vectors
    st = cp.sin(theta_all); ct = cp.cos(theta_all)
    x = st * cp.cos(phi_all); y = st * cp.sin(phi_all); z = ct
    # allocate table ordered by ipnest
    vx = cp.empty((npix,), dtype=cp.float64)
    vy = cp.empty_like(vx)
    vz = cp.empty_like(vx)
    vx[ipnest] = x
    vy[ipnest] = y
    vz[ipnest] = z
    _cache[nside] = (vx, vy, vz)
    return vx, vy, vz

def pix2vec_nest_gpu(nside:int, ipnest):
    ipnest = cp.asarray(ipnest, dtype=cp.int64)
    if nside not in _cache:
        _build_cache_for_nside(int(nside))
    vx, vy, vz = _cache[int(nside)]
    return vx[ipnest], vy[ipnest], vz[ipnest]
