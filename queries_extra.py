import cupy as cp
import numpy as np
from .queries_ring import RingQueryGPU

class RingQueryExtraGPU(RingQueryGPU):
    def query_annulus(self, theta0, phi0, rmin, rmax, batch=1<<20):
        assert rmin <= rmax
        Npix = int(self.map.tables.npix)
        ip = cp.arange(Npix, dtype=cp.int64)
        c0, s0 = np.cos(theta0), np.sin(theta0)
        cos_rmin = np.cos(rmin)
        cos_rmax = np.cos(rmax)
        out = []
        for s in range(0, Npix, batch):
            e = min(Npix, s+batch)
            t, p = self.map.pix2ang(ip[s:e])
            cosd = cp.cos(t)*c0 + cp.sin(t)*s0*cp.cos(p - phi0)
            sel = (cosd <= cos_rmin) & (cosd >= cos_rmax)
            out.append(ip[s:e][sel])
        return cp.concatenate(out, axis=0) if out else cp.empty((0,), dtype=cp.int64)

    def query_strip(self, theta_min, theta_max, batch=1<<20):
        assert 0.0 <= theta_min <= np.pi and 0.0 <= theta_max <= np.pi and theta_min <= theta_max
        Npix = int(self.map.tables.npix)
        ip = cp.arange(Npix, dtype=cp.int64)
        out = []
        for s in range(0, Npix, batch):
            e = min(Npix, s+batch)
            t, p = self.map.pix2ang(ip[s:e])
            sel = (t >= theta_min) & (t <= theta_max)
            out.append(ip[s:e][sel])
        return cp.concatenate(out, axis=0) if out else cp.empty((0,), dtype=cp.int64)


import numpy as np
import cupy as cp

class RingQueryExtraGPU(RingQueryGPU):
    # existing methods are inherited; add box queries
    def query_box(self, theta_min, theta_max, phi_min, phi_max, inclusive=True, batch=1<<20):
        Npix = int(self.map.tables.npix)
        ip = cp.arange(Npix, dtype=cp.int64)
        # normalize phi to [0, 2pi)
        phi_min = (phi_min + 2*np.pi) % (2*np.pi)
        phi_max = (phi_max + 2*np.pi) % (2*np.pi)
        wrap = phi_max < phi_min
        out = []
        for s in range(0, Npix, batch):
            e = min(Npix, s+batch)
            t, p = self.map.pix2ang(ip[s:e])
            in_theta = (t >= theta_min) & (t <= theta_max) if inclusive else (t > theta_min) & (t < theta_max)
            if not wrap:
                in_phi = (p >= phi_min) & (p <= phi_max) if inclusive else (p > phi_min) & (p < phi_max)
            else:
                in_phi = ((p >= phi_min) | (p <= phi_max)) if inclusive else ((p > phi_min) | (p < phi_max))
            sel = in_theta & in_phi
            out.append(ip[s:e][sel])
        return cp.concatenate(out, axis=0) if out else cp.empty((0,), dtype=cp.int64)

    def query_lonlat_rect(self, lon_min, lon_max, lat_min, lat_max, inclusive=True, batch=1<<20):
        # lon in radians (phi), lat in radians (-pi/2..pi/2)
        theta_min = (np.pi/2 - lat_max); theta_max = (np.pi/2 - lat_min)
        return self.query_box(theta_min, theta_max, lon_min, lon_max, inclusive=inclusive, batch=batch)
