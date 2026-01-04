import cupy as cp
import numpy as np
from .alm_ops import almxfl

def gaussian_beam_fl(lmax, fwhm_rad):
    # b_l = exp(-0.5 * l(l+1) * sigma^2), sigma = fwhm / sqrt(8 ln 2)
    sigma = fwhm_rad / np.sqrt(8.0*np.log(2.0))
    ell = cp.arange(lmax+1, dtype=cp.float64)
    return cp.exp(-0.5 * ell*(ell+1.0) * sigma*sigma)

def smooth_alm_gaussian(alm, fwhm_rad):
    lmax = int(cp.sqrt(cp.asarray(alm).size).get()) - 1
    bl = gaussian_beam_fl(lmax, fwhm_rad)
    return almxfl(alm, bl)
