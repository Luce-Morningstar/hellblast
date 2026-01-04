import cupy as cp
import numpy as np
from .smoothing import gaussian_beam_fl
from .alm_ops import almxfl

def pixel_window_fl(nside, lmax, pol=False):
    """Return pixel window p_l for HEALPix at given nside.
    If healpy is available, uses healpy.sphtfunc.pixwin(...). Otherwise returns ones.
    For pol=True, if healpy is present, returns (pT, pP); else both ones.
    """
    try:
        import healpy as hp  # CPU, optional
        w = hp.sphtfunc.pixwin(nside, lmax=lmax, pol=pol)
        if pol:
            pT, pP = w[0], w[1]
            return cp.asarray(pT), cp.asarray(pP)
        else:
            return cp.asarray(w), None
    except Exception:
        if pol:
            return cp.ones((lmax+1,), dtype=cp.float64), cp.ones((lmax+1,), dtype=cp.float64)
        return cp.ones((lmax+1,), dtype=cp.float64), None

def smooth_alm_with_beam(alm, lmax, fwhm=None, bl=None, pixwin=None):
    """Apply combined filter F_l = bl(l)*pixwin(l) to alm.
    - If bl is None and fwhm is provided, build Gaussian b_l.
    - pixwin: if None, do not apply; otherwise a vector of length lmax+1.
    """
    fl = cp.ones((lmax+1,), dtype=cp.float64)
    if bl is not None:
        fl = fl * cp.asarray(bl, dtype=cp.float64)
    elif fwhm is not None:
        fl = fl * gaussian_beam_fl(lmax, fwhm)
    if pixwin is not None:
        fl = fl * cp.asarray(pixwin, dtype=cp.float64)
    return almxfl(alm, fl)

def smooth_map(m, nside, lmax=None, fwhm=None, bl=None, use_pixel_window=True, nest=False):
    from .hp_compat import map2alm, alm2map
    if lmax is None: lmax = 3*int(nside)
    # build pixwin if requested
    pw = None
    if use_pixel_window:
        pw, _ = pixel_window_fl(int(nside), lmax, pol=False)
    alm = map2alm(m, lmax=lmax, nest=nest)
    alm_sm = smooth_alm_with_beam(alm, lmax, fwhm=fwhm, bl=bl, pixwin=pw)
    return alm2map(alm_sm, nside=int(nside), lmax=lmax, nest=nest)
