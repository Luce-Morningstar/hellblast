import cupy as cp
import numpy as np
from .ring_tables import HealpixRingTables
from .nested_gpu import ang2pix_nest_gpu
from .nested_pix2ang import pix2ang_nest_gpu
from .coords_ring import RingMapperGPU

def ring2nest_full(m_ring, nside):
    m_ring = cp.asarray(m_ring)
    nside = int(nside)
    tabs = HealpixRingTables(nside)
    npix = int(tabs.npix)
    th = cp.empty((npix,), dtype=cp.float64); ph = cp.empty_like(th)
    for r in range(int(tabs.Nr)):
        npr = int(tabs.nphi[r].get()); base = int(tabs.ring_start[r].get())
        dphi = 2*cp.pi / npr; pc0 = tabs.phi_center0[r]
        j = cp.arange(npr, dtype=cp.float64)
        ph[base:base+npr] = pc0 + j*dphi
        th[base:base+npr] = tabs.theta_centers[r]
    ipnest = ang2pix_nest_gpu(nside, th, ph)
    out = cp.empty_like(m_ring)
    out[ipnest] = m_ring
    return out

def nest2ring_full(m_nest, nside):
    m_nest = cp.asarray(m_nest)
    nside = int(nside)
    npix = 12*nside*nside
    th, ph = pix2ang_nest_gpu(nside, cp.arange(npix, dtype=cp.int64))
    mapper = RingMapperGPU(nside)
    ipring = mapper.ang2pix(th, ph)
    out = cp.empty_like(m_nest)
    out[ipring] = m_nest
    return out
