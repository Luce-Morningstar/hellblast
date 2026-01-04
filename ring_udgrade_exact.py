import cupy as cp
import numpy as np
from .ring_tables import HealpixRingTables
from .coords_ring import RingMapperGPU

def _ring_theta_phi_all(tabs: HealpixRingTables):
    npix = int(tabs.npix)
    th = cp.empty((npix,), dtype=cp.float64)
    ph = cp.empty_like(th)
    for r in range(int(tabs.Nr)):
        npr = int(tabs.nphi[r].get()); base = int(tabs.ring_start[r].get())
        dphi = 2*np.pi / npr; pc0 = tabs.phi_center0[r]
        j = cp.arange(npr, dtype=cp.float64)
        ph[base:base+npr] = pc0 + j*dphi
        th[base:base+npr] = tabs.theta_centers[r]
    return th, ph

def degrade_ring_exact_gpu(m_src_flat, nside_src:int, nside_dst:int, conserve='mean'):
    """Exact RING degrade (GPU): average child pixels into parent pixels without changing order.
    Assumes nside_src is a multiple of nside_dst by a power of two.
    Uses angular mapping of *source pixel centers* to destination pixel IDs in RING order.
    """
    m_src_flat = cp.asarray(m_src_flat, dtype=cp.float64)
    nside_src = int(nside_src); nside_dst = int(nside_dst)
    assert (nside_src % nside_dst)==0, "nside_src must be multiple of nside_dst"
    factor = nside_src // nside_dst
    assert (factor & (factor-1))==0, "factor must be a power of two"
    tabs_src = HealpixRingTables(nside_src)
    tabs_dst = HealpixRingTables(nside_dst)
    mapper_dst = RingMapperGPU(nside_dst)
    th_src, ph_src = _ring_theta_phi_all(tabs_src)
    ip_dst = mapper_dst.ang2pix(th_src, ph_src)  # parent id in RING@dst
    npix_dst = 12*nside_dst*nside_dst
    sums = cp.bincount(ip_dst, weights=m_src_flat, minlength=npix_dst)
    if conserve=='sum':
        return sums
    cnts = cp.bincount(ip_dst, minlength=npix_dst)
    cnts = cp.maximum(cnts, 1)
    return sums / cnts

def upgrade_ring_exact_gpu(m_src_flat, nside_src:int, nside_dst:int, conserve='mean'):
    """Exact RING upgrade (GPU): replicate parent value to all children (area-preserving).
    Assumes nside_dst is a multiple of nside_src by a power of two.
    """
    m_src_flat = cp.asarray(m_src_flat, dtype=cp.float64)
    nside_src = int(nside_src); nside_dst = int(nside_dst)
    assert (nside_dst % nside_src)==0, "nside_dst must be multiple of nside_src"
    factor = nside_dst // nside_src
    assert (factor & (factor-1))==0, "factor must be a power of two"
    tabs_dst = HealpixRingTables(nside_dst)
    mapper_src = RingMapperGPU(nside_src)
    # For each destination pixel, find its parent in the source grid by mapping its center angle
    th_dst, ph_dst = _ring_theta_phi_all(tabs_dst)
    ip_parent = mapper_src.ang2pix(th_dst, ph_dst)  # parent in RING@src
    out = m_src_flat[ip_parent]
    if conserve=='sum':
        # split parent sum evenly among its (factor^2) children
        out = out / float(factor*factor)
    return out
