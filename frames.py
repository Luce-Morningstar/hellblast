import cupy as cp
import numpy as np

# Hipparcos/IAU 1958 constants (J2000) — eq -> gal rotation matrix
R_EQ2GAL = cp.asarray([[ -0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                       [  0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
                       [ -0.8676661490190047, -0.1980763734312015,  0.4559837761750669]], dtype=cp.float64)
R_GAL2EQ = R_EQ2GAL.T

def ang_to_vec(theta, phi):
    st = cp.sin(theta); ct = cp.cos(theta)
    x = st*cp.cos(phi); y = st*cp.sin(phi); z = ct
    return x, y, z

def vec_to_ang(x, y, z):
    r = cp.sqrt(x*x + y*y + z*z)
    x = x/r; y = y/r; z = z/r
    theta = cp.arccos(cp.clip(z, -1.0, 1.0))
    phi = cp.mod(cp.arctan2(y, x), 2*cp.pi)
    return theta, phi

def convert_angles(theta, phi, src='E', dst='G'):
    src = src.upper()[0]; dst = dst.upper()[0]
    if src == dst: return cp.asarray(theta), cp.asarray(phi)
    x,y,z = ang_to_vec(theta, phi)
    v = cp.stack([x,y,z], axis=0)  # [3,N]
    if src=='E' and dst=='G':
        R = R_EQ2GAL
    elif src=='G' and dst=='E':
        R = R_GAL2EQ
    else:
        raise ValueError("src/dst must be 'E' or 'G'")
    v2 = R @ v
    return vec_to_ang(v2[0], v2[1], v2[2])

def map_convert_frame(m, nside, src='E', dst='G', order='RING', interpolation='bilinear'):
    """Reproject a full-sky map between frames by inverse mapping from target pixels to source sky."""
    m = cp.asarray(m, dtype=cp.float64)
    from .ring_tables import HealpixRingTables
    if order.upper() != 'RING':
        raise NotImplementedError("Only RING order implemented for map_convert_frame")
    tabs = HealpixRingTables(int(nside))
    npix = int(tabs.npix)
    th = cp.empty((npix,), dtype=cp.float64); ph = cp.empty_like(th)
    for r in range(int(tabs.Nr)):
        npr = int(tabs.nphi[r].get()); base = int(tabs.ring_start[r].get())
        dphi = 2*cp.pi / npr; pc0 = tabs.phi_center0[r]
        j = cp.arange(npr, dtype=cp.float64)
        ph[base:base+npr] = pc0 + j*dphi
        th[base:base+npr] = tabs.theta_centers[r]
    # For each target pixel center in dst frame, find the source angle in src frame
    th_src, ph_src = convert_angles(th, ph, src=dst, dst=src)  # inverse mapping
    from .coords_ring import RingMapperGPU
    mapper = RingMapperGPU(int(nside))
    ip_src = mapper.ang2pix(th_src, ph_src)
    return m[ip_src]
