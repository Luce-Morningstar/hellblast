import cupy as cp, numpy as np
from .ring_tables import HealpixRingTables
from .coords_ring import RingMapperGPU
from .queries_ring import RingQueryGPU
from .queries_extra import RingQueryExtraGPU
from .isht import alm2map_gpu as alm2map_scalar
from .sht import map2alm_gpu as map2alm_scalar
from .spin import map2alm_spin_gpu as map2alm_spin2, qu_from_eb_gpu as qu_from_eb2, eb_from_qu_gpu as eb_from_qu2
from .spin1 import map2alm_spin1_gpu, alm2map_spin1_gpu
from .udgrade import unpack_flat_to_rings, pack_rings_to_flat, ud_grade_ring_gpu, degrade_nested_gpu, upgrade_nested_gpu
from .nested_gpu import ang2pix_nest_gpu
from .nested_pix2ang import pix2ang_nest_gpu, pix2vec_nest_analytic_gpu
from .alm_ops import almxfl as _almxfl, alm2cl as _alm2cl, synalm as _synalm
from .smoothing import smooth_alm_gaussian as _smooth_alm_gaussian, gaussian_beam_fl
from .anafast import anafast_ring as _anafast_ring
from .rotate_alm_euler import rotate_alm_euler_fast as _rotate_alm_euler_fast
from .wigner_delta import rotate_alm_euler_risbo as _rotate_alm_euler_risbo
from .io_fits import read_map_fits, write_map_fits, read_alm_fits, write_alm_fits
from .io_fits_multi import read_tqu_fits, write_tqu_fits
from .viz import mollview as _mollview
from .beams import smooth_map as _smooth_map, pixel_window_fl, smooth_alm_with_beam
from .synfast_tqu import synfast_tqu as _synfast_tqu

def _nside_from_npix(npix:int)->int:
    nside2 = npix // 12
    nside = int(np.sqrt(nside2))
    if 12*nside*nside != npix:
        raise ValueError("npix is not 12*nside^2")
    return nside

def _rings_from_flat(flat_map, tabs: HealpixRingTables):
    rings = []
    for r in range(int(tabs.Nr)):
        npr = int(tabs.nphi[r].get())
        base = int(tabs.ring_start[r].get())
        rings.append(cp.asarray(flat_map[base:base+npr], dtype=cp.float64))
    return rings

# ---------- map<->alm (scalar) ----------
def map2alm(m, lmax=None, mmax=None, nest=False):
    m = cp.asarray(m, dtype=cp.float64)
    npix = int(m.size)
    nside = _nside_from_npix(npix)
    tabs = HealpixRingTables(nside)
    if lmax is None: lmax = 3*nside
    rings = _rings_from_flat(m, tabs)
    alm = map2alm_scalar(rings, tabs.theta_centers, tabs.nphi, tabs.weights_dz(), lmax, mmax=mmax)
    return alm

def alm2map(alm, nside, lmax=None, nest=False):
    tabs = HealpixRingTables(int(nside))
    if lmax is None: lmax = int(np.sqrt(cp.asarray(alm).size).get()) - 1
    rings = alm2map_scalar(alm, tabs.theta_centers, tabs.nphi, lmax)
    return pack_rings_to_flat(rings, tabs)

# ---------- spin transforms ----------
def map2alm_spin(Q, U, spin=2, lmax=None, mmax=None, nest=False):
    Q = cp.asarray(Q, dtype=cp.float64)
    U = cp.asarray(U, dtype=cp.float64)
    npix = int(Q.size); nside = _nside_from_npix(npix)
    if lmax is None: lmax = 3*nside
    tabs = HealpixRingTables(nside)
    Qr = _rings_from_flat(Q, tabs); Ur = _rings_from_flat(U, tabs)
    if spin==2:
        return eb_from_qu2(Qr, Ur, tabs.theta_centers, tabs.nphi, tabs.weights_dz(), lmax, mmax)
    elif spin==1:
        return map2alm_spin1_gpu(Qr, Ur, tabs.theta_centers, tabs.nphi, tabs.weights_dz(), lmax, mmax)
    else:
        raise ValueError("spin must be 1 or 2")

def alm2map_spin(Aplus, Aminus, spin=2, nside=None, lmax=None, nest=False):
    if nside is None:
        lmax = int(np.sqrt(cp.asarray(Aplus).size).get()) - 1 if lmax is None else lmax
        # choose minimal nside so lmax ≈ 3 nside
        nside = max(1, int(np.ceil(lmax/3)))
    tabs = HealpixRingTables(int(nside))
    if lmax is None: lmax = int(np.sqrt(cp.asarray(Aplus).size).get()) - 1
    if spin==2:
        Qr, Ur = qu_from_eb2(Aplus, Aminus, tabs.theta_centers, tabs.nphi, lmax)
    elif spin==1:
        Qr, Ur = alm2map_spin1_gpu(Aplus, Aminus, tabs.theta_centers, tabs.nphi, lmax)
    else:
        raise ValueError("spin must be 1 or 2")
    return pack_rings_to_flat(Qr, tabs), pack_rings_to_flat(Ur, tabs)

# ---------- polarization helpers ----------
def iqu2eb(Q, U, lmax=None, nest=False):
    Q = cp.asarray(Q, dtype=cp.float64); U = cp.asarray(U, dtype=cp.float64)
    npix = int(Q.size); nside = _nside_from_npix(npix)
    if lmax is None: lmax = 3*nside
    tabs = HealpixRingTables(nside)
    Qr = _rings_from_flat(Q, tabs); Ur = _rings_from_flat(U, tabs)
    E,B = eb_from_qu2(Qr, Ur, tabs.theta_centers, tabs.nphi, tabs.weights_dz(), lmax)
    return E, B

def eb2iqu(E, B, nside=None, lmax=None, nest=False):
    if nside is None:
        lmax = int(cp.sqrt(cp.asarray(E).size).get()) - 1 if lmax is None else lmax
        nside = max(1, int(np.ceil(lmax/3)))
    tabs = HealpixRingTables(int(nside))
    Qr, Ur = qu_from_eb2(E, B, tabs.theta_centers, tabs.nphi, lmax if lmax is not None else int(cp.sqrt(cp.asarray(E).size).get())-1)
    return pack_rings_to_flat(Qr, tabs), pack_rings_to_flat(Ur, tabs)

# ---------- interpolation helper ----------
def get_interp_val(nside, theta, phi, m, order='bilinear'):
    from .udgrade import pack_rings_to_flat
    from .ring_tables import HealpixRingTables
    from .udgrade import unpack_flat_to_rings
    from .udgrade import get_interp_val_ring
    tabs = HealpixRingTables(int(nside))
    return get_interp_val_ring(theta, phi, m, tabs)

    # ---------- alm utilities ----------
def almxfl(alm, fl):
    return _almxfl(alm, fl)
def alm2cl(alm, lmax=None):
    return _alm2cl(alm, lmax=lmax)
def synalm(cl, seed=None):
    return _synalm(cl, seed=seed)

# ---------- rotations ----------
def rotate_alm(alm, alpha, beta, gamma, method="delta"):
    if method=="delta":
        return _rotate_alm_euler_risbo(alm, alpha, beta, gamma)
    elif method=="fast":
        return _rotate_alm_euler_fast(alm, alpha, beta, gamma)
    else:
        raise ValueError("method must be 'delta' or 'fast'")

# ---------- geometry ----------
def ang2pix(nside, theta, phi, nest=False):
    if nest:
        return ang2pix_nest_gpu(int(nside), theta, phi)
    mapper = RingMapperGPU(int(nside))
    return mapper.ang2pix(theta, phi)

def pix2ang(nside, ipix, nest=False):
    if nest:
        return pix2ang_nest_gpu(int(nside), ipix)
    mapper = RingMapperGPU(int(nside))
    return mapper.pix2ang(ipix)

def pix2vec(nside, ipix, nest=False):
    if nest:
        return pix2vec_nest_analytic_gpu(int(nside), ipix)
    mapper = RingMapperGPU(int(nside))
    th, ph = mapper.pix2ang(ipix)
    x = cp.sin(th)*cp.cos(ph); y = cp.sin(th)*cp.sin(ph); z = cp.cos(th)
    return x, y, z

# ---------- ud_grade ----------
def ud_grade(m, nside_in=None, nside_out=None, order_in="RING", order_out="RING", conserve="mean"):
    m = cp.asarray(m)
    npix = int(m.size)
    if nside_in is None:
        nside_in = _nside_from_npix(npix)
    if nside_out is None:
        raise ValueError("nside_out required")
    nside_in = int(nside_in); nside_out = int(nside_out)
    if order_in.upper()=="RING" and order_out.upper()=="RING" and ( (max(nside_in,nside_out) % min(nside_in,nside_out))==0 ):
        from .ring_udgrade_exact import degrade_ring_exact_gpu, upgrade_ring_exact_gpu
        if nside_in > nside_out:
            return degrade_ring_exact_gpu(m, nside_in, nside_out, conserve=conserve)
        elif nside_out > nside_in:
            return upgrade_ring_exact_gpu(m, nside_in, nside_out, conserve=conserve)
        else:
            return m
    # fallback: previous path via NESTED conversion
    tabs_in = HealpixRingTables(int(nside_in))
    from .nested_gpu import ang2pix_nest_gpu
    th_all = cp.empty((npix,), dtype=cp.float64); ph_all = cp.empty((npix,), dtype=cp.float64)
    for r in range(int(tabs_in.Nr)):
        npr = int(tabs_in.nphi[r].get()); base = int(tabs_in.ring_start[r].get())
        dphi = 2*np.pi / npr; pc0 = tabs_in.phi_center0[r]
        j = cp.arange(npr, dtype=cp.float64)
        ph_all[base:base+npr] = pc0 + j*dphi
        th_all[base:base+npr] = tabs_in.theta_centers[r]
    ipnest = ang2pix_nest_gpu(int(nside_in), th_all, ph_all)
    from .udgrade import degrade_nested_gpu, upgrade_nested_gpu, pack_rings_to_flat
    m_nested = m[ipnest]
    if nside_out < nside_in:
        m2 = degrade_nested_gpu(m_nested, int(nside_in), int(nside_out), conserve=conserve)
    elif nside_out > nside_in:
        m2 = upgrade_nested_gpu(m_nested, int(nside_in), int(nside_out), conserve=conserve)
    else:
        m2 = m_nested
    tabs_out = HealpixRingTables(int(nside_out))
    npix_out = 12*int(nside_out)*int(nside_out)
    th2 = cp.empty((npix_out,), dtype=cp.float64); ph2 = cp.empty_like(th2)
    for r in range(int(tabs_out.Nr)):
        npr = int(tabs_out.nphi[r].get()); base = int(tabs_out.ring_start[r].get())
        dphi = 2*np.pi / npr; pc0 = tabs_out.phi_center0[r]
        j = cp.arange(npr, dtype=cp.float64)
        ph2[base:base+npr] = pc0 + j*dphi
        th2[base:base+npr] = tabs_out.theta_centers[r]
    ipnest_out = ang2pix_nest_gpu(int(nside_out), th2, ph2)
    out_ring = cp.empty_like(m2)
    idx = cp.argsort(ipnest_out)
    out_ring[idx] = m2
    return out_ring



# ---------- FITS helpers ----------
def read_map(path, hdu=1, to_gpu=True, dtype=np.float64):
    return read_map_fits(path, hdu=hdu, to_gpu=to_gpu, dtype=dtype)
def write_map(path, m, header=None, overwrite=True):
    return write_map_fits(path, m, header=header, overwrite=overwrite)
def read_alm(path, hdu=1, to_gpu=True, dtype=np.complex128):
    return read_alm_fits(path, hdu=hdu, to_gpu=to_gpu, dtype=dtype)
def write_alm(path, alm, header=None, overwrite=True):
    return write_alm_fits(path, alm, header=header, overwrite=overwrite)

# ---------- visualization ----------
def mollview(m, nside=None, xsize=1200, ysize=None, nest=False, title=None, cmap='viridis',
             vmin=None, vmax=None, return_image=False, save=None, interpolation='bilinear'):
    return _mollview(m, nside=nside, xsize=xsize, ysize=ysize, nest=nest, title=title, cmap=cmap,
                     vmin=vmin, vmax=vmax, return_image=return_image, save=save, interpolation=interpolation)


# ---------- extra FITS helpers (TQU) ----------
def read_tqu(path, to_gpu=True):
    return read_tqu_fits(path, to_gpu=to_gpu)
def write_tqu(path, T=None, Q=None, U=None, nside=None, ordering="RING", coordsys="G", overwrite=True, extra_header=None):
    return write_tqu_fits(path, T=T, Q=Q, U=U, nside=nside, ordering=ordering, coordsys=coordsys, overwrite=overwrite, extra_header=extra_header)

# ---------- smoothing pipeline ----------
def smooth_map(m, nside, lmax=None, fwhm=None, bl=None, use_pixel_window=True, nest=False):
    return _smooth_map(m, nside, lmax=lmax, fwhm=fwhm, bl=bl, use_pixel_window=use_pixel_window, nest=nest)

# ---------- synfast ----------
def synfast(cls, nside, lmax=None, fwhm=None, use_pixel_window=True, seed=None,
            sigmaT=None, sigmaQ=None, sigmaU=None, return_alm=False):
    return _synfast_tqu(cls, nside, lmax=lmax, fwhm=fwhm, use_pixel_window=use_pixel_window, seed=seed,
                        sigmaT=sigmaT, sigmaQ=sigmaQ, sigmaU=sigmaU, return_alm=return_alm)


# ---------- frame conversions ----------
from .frames import convert_angles as convert_angles_frame, map_convert_frame as map_convert_frame

# ---------- order conversions ----------
from .order_convert import ring2nest_full as ring2nest, nest2ring_full as nest2ring

# ---------- extra views ----------
from .viz import orthview as _orthview, gnomview as _gnomview
def orthview(m, nside=None, xsize=800, fov=np.pi/2, lon0=0.0, lat0=0.0, nest=False, title=None, cmap='viridis',
             vmin=None, vmax=None, return_image=False, save=None, interpolation='bilinear'):
    return _orthview(m, nside=nside, xsize=xsize, fov=fov, lon0=lon0, lat0=lat0, nest=nest, title=title, cmap=cmap,
                     vmin=vmin, vmax=vmax, return_image=return_image, save=save, interpolation=interpolation)
def gnomview(m, nside=None, xsize=800, fov=np.deg2rad(60.0), lon0=0.0, lat0=0.0, nest=False, title=None, cmap='viridis',
             vmin=None, vmax=None, return_image=False, save=None, interpolation='bilinear'):
    return _gnomview(m, nside=nside, xsize=xsize, fov=fov, lon0=lon0, lat0=lat0, nest=nest, title=title, cmap=cmap,
                     vmin=vmin, vmax=vmax, return_image=return_image, save=save, interpolation=interpolation)
