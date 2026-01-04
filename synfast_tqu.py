import cupy as cp
import numpy as np
from .alm_ops import almxfl
from .smoothing import gaussian_beam_fl
from .beams import pixel_window_fl
from .isht import alm2map_gpu as alm2map_scalar
from .spin import qu_from_eb_gpu

def _synalm_scalar_from_cl(cl):
    # Gaussian alm for scalar field
    rs = cp.random
    cl = cp.asarray(cl, dtype=cp.float64)
    lmax = int(cl.size) - 1
    alm = cp.zeros(((lmax+1)*(lmax+1),), dtype=cp.complex128)
    for l in range(lmax+1):
        C = float(cl[l])
        if C <= 0.0:
            continue
        a0 = rs.normal(0.0, np.sqrt(C))
        alm[l*(l+1)+0] = a0 + 0.0j
        if l>0:
            std = np.sqrt(C/2.0)
            re = rs.normal(0.0, std, size=l)
            im = rs.normal(0.0, std, size=l)
            alm[l*(l+1)+1:l*(l+1)+l+1] = re + 1j*im
    return alm

def _synalm_TE_block(cltt, clee, clbb, clte):
    # Build correlated T/E; B is independent. cl* are vectors length lmax+1
    lmax = int(cltt.size) - 1
    aT = cp.zeros(((lmax+1)*(lmax+1),), dtype=cp.complex128)
    aE = cp.zeros_like(aT)
    aB = cp.zeros_like(aT)
    rs = cp.random
    for l in range(lmax+1):
        Ctt = float(cltt[l]); Cee = float(clee[l]); Cbb = float(clbb[l]); Cte = float(clte[l])
        # Cholesky of [[Ctt, Cte],[Cte, Cee]]
        if Ctt<=0 or Cee<=0 or (Ctt*Cee - Cte*Cte)<=0:
            L11 = np.sqrt(max(Ctt, 0.0)); L21 = 0.0; L22 = np.sqrt(max(Cee, 0.0))
        else:
            L11 = np.sqrt(Ctt)
            L21 = Cte / L11
            L22 = np.sqrt(Cee - L21*L21)
        # m=0 real
        z1 = rs.normal(0.0, 1.0); z2 = rs.normal(0.0, 1.0)
        t0 = L11*z1
        e0 = L21*z1 + L22*z2
        aT[l*(l+1)+0] = t0 + 0.0j
        aE[l*(l+1)+0] = e0 + 0.0j
        # B m=0
        b0 = rs.normal(0.0, np.sqrt(max(Cbb,0.0)))
        aB[l*(l+1)+0] = b0 + 0.0j
        # m>0 complex
        if l>0:
            stdB = np.sqrt(max(Cbb,0.0)/2.0)
            reB = rs.normal(0.0, stdB, size=l)
            imB = rs.normal(0.0, stdB, size=l)
            aB[l*(l+1)+1:l*(l+1)+l+1] = reB + 1j*imB
            # T/E correlated
            std = 1.0
            z1r = rs.normal(0.0, 1.0, size=l); z1i = rs.normal(0.0, 1.0, size=l)
            z2r = rs.normal(0.0, 1.0, size=l); z2i = rs.normal(0.0, 1.0, size=l)
            t = (L11*z1r + 1j*L11*z1i)
            e = (L21*z1r + L22*z2r) + 1j*(L21*z1i + L22*z2i)
            aT[l*(l+1)+1:l*(l+1)+l+1] = t
            aE[l*(l+1)+1:l*(l+1)+l+1] = e
    return aT, aE, aB

def synfast_tqu(cls, nside, lmax=None, fwhm=None, use_pixel_window=True, seed=None,
                sigmaT=None, sigmaQ=None, sigmaU=None, return_alm=False):
    """Generate T, Q, U maps from spectra on GPU.
    cls: if pol, provide dict with keys 'tt','ee','bb','te'; if scalar-only, provide {'tt':...}.
    nside: output resolution
    fwhm: optional Gaussian beam FWHM (radians) to apply
    use_pixel_window: multiply by pixel window p_l(nside)
    noise: per-pixel white noise stddevs sigmaT, sigmaQ, sigmaU (floats); if None, no noise
    return_alm: if True, also return (aT, aE, aB)
    """
    if seed is not None:
        cp.random.seed(int(seed))
    if lmax is None:
        # infer from provided spectra length
        anycl = next(iter(cls.values()))
        lmax = int(cp.asarray(anycl).size) - 1
    lmax = int(lmax)
    # spectra to device
    get = lambda k: cp.asarray(cls.get(k, None)) if cls.get(k, None) is not None else None
    Ctt, Cee, Cbb, Cte = get('tt'), get('ee'), get('bb'), get('te')
    if Cee is None and Cbb is None and Cte is None:
        # scalar-only T
        aT = _synalm_scalar_from_cl(Ctt)
        aE = cp.zeros_like(aT); aB = cp.zeros_like(aT)
    else:
        if Ctt is None or Cee is None or Cbb is None or Cte is None:
            raise ValueError("For polarized synfast provide tt, ee, bb, te spectra.")
        aT, aE, aB = _synalm_TE_block(Ctt, Cee, Cbb, Cte)
    # apply beam/pixwin
    bl = None
    if fwhm is not None:
        bl = gaussian_beam_fl(lmax, fwhm)
    if use_pixel_window:
        pT, pP = pixel_window_fl(int(nside), lmax, pol=True)
    else:
        pT, pP = None, None
    def filt(alm, p):
        if bl is None and p is None: return alm
        fl = cp.ones((lmax+1,), dtype=cp.float64)
        if bl is not None: fl *= bl
        if p is not None: fl *= p
        return almxfl(alm, fl)
    aT = filt(aT, pT); aE = filt(aE, pP); aB = filt(aB, pP)
    # synthesize maps
    from .ring_tables import HealpixRingTables
    tabs = HealpixRingTables(int(nside))
    T_rings = alm2map_scalar(aT, tabs.theta_centers, tabs.nphi, lmax)
    Q_rings, U_rings = qu_from_eb_gpu(aE, aB, tabs.theta_centers, tabs.nphi, lmax)
    from .udgrade import pack_rings_to_flat
    T = pack_rings_to_flat(T_rings, tabs)
    Q = pack_rings_to_flat(Q_rings, tabs)
    U = pack_rings_to_flat(U_rings, tabs)
    # add white noise if requested
    rs = cp.random
    if sigmaT is not None and sigmaT>0: T = T + rs.normal(0.0, float(sigmaT), size=T.shape)
    if sigmaQ is not None and sigmaQ>0: Q = Q + rs.normal(0.0, float(sigmaQ), size=Q.shape)
    if sigmaU is not None and sigmaU>0: U = U + rs.normal(0.0, float(sigmaU), size=U.shape)
    if return_alm:
        return T, Q, U, aT, aE, aB
    return T, Q, U
