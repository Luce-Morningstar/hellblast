import cupy as cp
import numpy as np

def alm_index(l, m):
    return l*(l+1) + m  # m>=0

def almxfl(alm, fl):
    alm = cp.asarray(alm, dtype=cp.complex128)
    fl = cp.asarray(fl, dtype=cp.float64)
    lmax = int(cp.sqrt(alm.size).get()) - 1
    out = cp.empty_like(alm)
    for l in range(lmax+1):
        scale = fl[l]
        start = l*(l+1)
        end = start + (l+1)
        out[start:end] = alm[start:end] * scale
    return out

def alm2cl(alm, lmax=None):
    alm = cp.asarray(alm, dtype=cp.complex128)
    if lmax is None:
        lmax = int(cp.sqrt(alm.size).get()) - 1
    cl = cp.zeros((lmax+1,), dtype=cp.float64)
    for l in range(lmax+1):
        base = l*(l+1)
        al = alm[base:base+(l+1)]
        # al[0] is m=0, al[1..l] are m=1..l
        m0 = cp.abs(al[0])**2
        mp = cp.abs(al[1:])**2
        tot = m0 + 2.0*cp.sum(mp)  # conjugate symmetry doubles m>0
        cl[l] = cp.real(tot / (2*l + 1))
    return cl

def synalm(cl, seed=None):
    # Generate Gaussian isotropic alm up to len(cl)-1 with variance Cl
    rs = cp.random.RandomState(seed) if seed is not None else cp.random
    cl = cp.asarray(cl, dtype=cp.float64)
    lmax = int(cl.size) - 1
    alm = cp.zeros(((lmax+1)*(lmax+1),), dtype=cp.complex128)
    for l in range(lmax+1):
        C = float(cl[l])
        # m=0 real Gaussian with var C
        if C <= 0.0:
            alm[l*(l+1)+0] = 0.0 + 0.0j
            continue
        a0 = rs.normal(0.0, np.sqrt(C))
        alm[l*(l+1)+0] = a0 + 0.0j
        # m>0 complex with <|alm|^2>=C, so Re and Im ~ N(0, C/2)
        if l>0:
            std = np.sqrt(C/2.0)
            re = rs.normal(0.0, std, size=l)
            im = rs.normal(0.0, std, size=l)
            alm[l*(l+1)+1 : l*(l+1)+(l+1)] = re + 1j*im
    return alm
