import cupy as cp
import numpy as np
from .coords_ring import RingMapperGPU

class RingQueryGPU:
    def __init__(self, nside:int):
        self.map = RingMapperGPU(nside)

    def get_all_neighbours(self, ipix):
        ipix = cp.asarray(ipix, dtype=cp.int64)
        # ring index and j
        rs = cp.searchsorted(self.map.tables.ring_start, ipix, side='right') - 1
        j = ipix - self.map.tables.ring_start[rs]
        npr = self.map.tables.nphi[rs]
        base = self.map.tables.ring_start[rs]
        west = base + (j - 1) % npr
        east = base + (j + 1) % npr
        # map to adjacent rings via phi projection
        th, ph = self.map.pix2ang(ipix)
        def proj_to_ring(phi, r):
            npr2 = self.map.tables.nphi[r]
            dphi = 2*np.pi / npr2
            pc0 = self.map.tables.phi_center0[r]
            j2 = cp.mod(cp.asarray(cp.rint(((phi - pc0) % (2*np.pi)) / dphi), dtype=cp.int64), npr2)
            return self.map.tables.ring_start[r] + j2
        rN = cp.maximum(rs - 1, 0)
        rS = cp.minimum(rs + 1, self.map.tables.Nr - 1)
        north = proj_to_ring(ph, rN)
        south = proj_to_ring(ph, rS)
        nprN = self.map.tables.nphi[rN]
        dphiN = 2*np.pi / nprN
        pc0N = self.map.tables.phi_center0[rN]
        jN = cp.mod(cp.asarray(cp.rint(((ph - pc0N) % (2*np.pi)) / dphiN), dtype=cp.int64), nprN)
        nw = self.map.tables.ring_start[rN] + (jN - 1) % nprN
        ne = self.map.tables.ring_start[rN] + (jN + 1) % nprN
        nprS = self.map.tables.nphi[rS]
        dphiS = 2*np.pi / nprS
        pc0S = self.map.tables.phi_center0[rS]
        jS = cp.mod(cp.asarray(cp.rint(((ph - pc0S) % (2*np.pi)) / dphiS), dtype=cp.int64), nprS)
        sw = self.map.tables.ring_start[rS] + (jS - 1) % nprS
        se = self.map.tables.ring_start[rS] + (jS + 1) % nprS
        return cp.stack([sw, west, nw, north, ne, east, se, south], axis=0)

    def query_disc(self, theta0, phi0, radius, batch=1<<20):
        Npix = int(self.map.tables.npix)
        ip = cp.arange(Npix, dtype=cp.int64)
        c0, s0 = np.cos(theta0), np.sin(theta0)
        sel = []
        for start in range(0, Npix, batch):
            end = min(Npix, start+batch)
            t, p = self.map.pix2ang(ip[start:end])
            cosd = cp.cos(t)*c0 + cp.sin(t)*s0*cp.cos(p - phi0)
            sel.append(ip[start:end][cosd >= np.cos(radius)])
        return cp.concatenate(sel, axis=0) if sel else cp.empty((0,), dtype=cp.int64)

    def query_polygon(self, thetas, phis):
        thetas = cp.asarray(thetas, dtype=cp.float64)
        phis = cp.asarray(phis, dtype=cp.float64)
        vx = cp.sin(thetas)*cp.cos(phis)
        vy = cp.sin(thetas)*cp.sin(phis)
        vz = cp.cos(thetas)
        vx2 = cp.roll(vx, -1); vy2 = cp.roll(vy, -1); vz2 = cp.roll(vz, -1)
        nx = vy*vz2 - vz*vy2
        ny = vz*vx2 - vx*vz2
        nz = vx*vy2 - vy*vx2
        norm = cp.sqrt(nx*nx+ny*ny+nz*nz) + 1e-18
        nx/=norm; ny/=norm; nz/=norm
        Npix = int(self.map.tables.npix)
        ip = cp.arange(Npix, dtype=cp.int64)
        outs = []
        batch = 1<<20
        for s in range(0, Npix, batch):
            e = min(Npix, s+batch)
            t, p = self.map.pix2ang(ip[s:e])
            x = cp.sin(t)*cp.cos(p)
            y = cp.sin(t)*cp.sin(p)
            z = cp.cos(t)
            inside = cp.ones((e-s,), dtype=cp.bool_)
            for k in range(nx.size):
                inside = inside & (nx[k]*x + ny[k]*y + nz[k]*z >= 0.0)
            outs.append(ip[s:e][inside])
        return cp.concatenate(outs, axis=0) if outs else cp.empty((0,), dtype=cp.int64)
