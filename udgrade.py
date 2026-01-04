import cupy as cp
import numpy as np
from .nested import part1by1_64, morton2d_64, encode_nested, Ring2NestLUT
from .ring_tables import HealpixRingTables

# --------- bit un-interleave (inverse of part1by1_64) ---------
def compact1by1_64(x):
    x = x & 0x5555555555555555
    x = (x ^ (x >> 1)) & 0x3333333333333333
    x = (x ^ (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    x = (x ^ (x >> 4)) & 0x00FF00FF00FF00FF
    x = (x ^ (x >> 8)) & 0x0000FFFF0000FFFF
    x = (x ^ (x >> 16)) & 0x00000000FFFFFFFF
    return x

def decode_nested_to_face_ixiy(ipix_nested, nside):
    # ipix_nested: cp.ndarray int64
    ip = cp.asarray(ipix_nested, dtype=cp.int64)
    nside = int(nside)
    pix_per_face = nside*nside
    face = (ip // pix_per_face).astype(cp.int64)
    idx_in_face = (ip % pix_per_face).astype(cp.uint64)
    # deinterleave morton to ix,iy
    ix = compact1by1_64(idx_in_face).astype(cp.int64)
    iy = compact1by1_64(idx_in_face >> 1).astype(cp.int64)
    return face, ix, iy

def encode_face_ixiy_to_nested(face, ix, iy, nside):
    face = cp.asarray(face, dtype=cp.int64)
    ix = cp.asarray(ix, dtype=cp.int64)
    iy = cp.asarray(iy, dtype=cp.int64)
    nside = int(nside)
    pix_per_face = nside*nside
    mort = ( (part1by1_64(iy.astype(cp.uint64)) << 1) | part1by1_64(ix.astype(cp.uint64)) ).astype(cp.int64)
    return face*pix_per_face + mort

# --------- NESTED degrade (factor must be a power of 2) ---------
def degrade_nested_gpu(map_nested, nside_src, nside_dst, conserve='mean'):
    map_nested = cp.asarray(map_nested)
    nside_src = int(nside_src); nside_dst = int(nside_dst)
    assert (nside_src % nside_dst)==0, "nside_src must be a multiple of nside_dst"
    factor = nside_src // nside_dst
    assert (factor & (factor-1))==0, "factor must be power of 2"
    npix_src = 12*nside_src*nside_src
    npix_dst = 12*nside_dst*nside_dst
    ip_src = cp.arange(npix_src, dtype=cp.int64)
    # decode child ix,iy and shift down by log2(factor) to get parent coords
    k = int(np.log2(factor))
    face, ix, iy = decode_nested_to_face_ixiy(ip_src, nside_src)
    pface = face
    pix = (ix >> k)
    piy = (iy >> k)
    ip_parent = encode_face_ixiy_to_nested(pface, pix, piy, nside_dst)
    # sum and count per parent
    s = cp.bincount(ip_parent, weights=map_nested, minlength=npix_dst)
    if conserve=='sum':
        return s
    c = cp.bincount(ip_parent, minlength=npix_dst)
    c = cp.maximum(c, 1)  # avoid div by zero
    return s / c

# --------- NESTED upgrade (spread parent to children, factor power of 2) ---------
def upgrade_nested_gpu(map_nested, nside_src, nside_dst, conserve='mean'):
    map_nested = cp.asarray(map_nested)
    nside_src = int(nside_src); nside_dst = int(nside_dst)
    assert (nside_dst % nside_src)==0, "nside_dst must be a multiple of nside_src"
    factor = nside_dst // nside_src
    assert (factor & (factor-1))==0, "factor must be power of 2"
    npix_dst = 12*nside_dst*nside_dst
    ip_dst = cp.arange(npix_dst, dtype=cp.int64)
    k = int(np.log2(factor))
    # For each child, compute its parent
    face, ix, iy = decode_nested_to_face_ixiy(ip_dst, nside_dst)
    pface = face
    pix = (ix >> k)
    piy = (iy >> k)
    ip_parent = encode_face_ixiy_to_nested(pface, pix, piy, nside_src)
    # gather
    out = map_nested[ip_parent]
    if conserve=='sum':
        # distribute parent's sum across children equally
        out = out / (factor*factor)
    return out

# --------- RING wrappers via LUT ---------
def pack_rings_to_flat(rings, tables: HealpixRingTables):
    # rings: list of cp arrays per ring
    npix = int(tables.npix)
    out = cp.empty((npix,), dtype=rings[0].dtype)
    for r in range(int(tables.Nr)):
        npr = int(tables.nphi[r].get())
        base = int(tables.ring_start[r].get())
        out[base:base+npr] = rings[r]
    return out

def unpack_flat_to_rings(flat, tables: HealpixRingTables):
    rings = []
    for r in range(int(tables.Nr)):
        npr = int(tables.nphi[r].get())
        base = int(tables.ring_start[r].get())
        rings.append(flat[base:base+npr])
    return rings

def ud_grade_ring_gpu(map_flat_ring, nside_src, nside_dst, lut: Ring2NestLUT, conserve='mean'):
    # Convert ring->nest, apply nested udgrade, convert back
    map_flat_ring = cp.asarray(map_flat_ring)
    ring2nest = lut.ring2nest
    nest2ring = lut.nest2ring
    map_nested = map_flat_ring[ring2nest]
    if nside_src > nside_dst:
        out_nested = degrade_nested_gpu(map_nested, nside_src, nside_dst, conserve=conserve)
    elif nside_dst > nside_src:
        out_nested = upgrade_nested_gpu(map_nested, nside_src, nside_dst, conserve=conserve)
    else:
        out_nested = map_nested
    out_ring = cp.empty_like(out_nested)
    out_ring[nest2ring] = out_nested
    return out_ring

# --------- Bilinear interpolation on RING (theta,phi) -> value ---------
def get_interp_val_ring(theta, phi, map_flat_ring, tables: HealpixRingTables):
    # Bilinear across nearest two rings and two neighboring phi samples
    theta = cp.asarray(theta, dtype=cp.float64)
    phi = cp.asarray(phi, dtype=cp.float64)
    m = cp.asarray(map_flat_ring)
    # locate rings via z_edges
    z = cp.cos(theta)
    z_edges = tables.z_edges
    Nr = z_edges.size - 1
    r = cp.searchsorted(-z_edges, -z, side='right') - 1
    r = cp.clip(r, 0, Nr-1)
    # neighbor ring (towards south)
    r2 = cp.clip(r+1, 0, Nr-1)
    t1 = tables.theta_centers[r]
    t2 = tables.theta_centers[r2]
    # weights along theta
    wt = cp.where(t2!=t1, (theta - t1)/(t2 - t1 + 1e-18), 0.0)
    # per-ring phi grid
    def sample_on_ring(rr, wphi):
        npr = tables.nphi[rr]
        dphi = 2*np.pi / npr
        pc0 = tables.phi_center0[rr]
        # fractional index relative to center-0
        jf = (phi - pc0) / dphi
        j0 = cp.floor(jf).astype(cp.int64)
        j1 = (j0 + 1) % npr
        base = tables.ring_start[rr]
        f0 = m[base + (j0 % npr)]
        f1 = m[base + j1]
        w = jf - cp.floor(jf)
        return (1.0 - w)*f0 + w*f1
    v1 = sample_on_ring(r, phi)
    v2 = sample_on_ring(r2, phi)
    return (1.0 - wt)*v1 + wt*v2
