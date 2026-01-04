import cupy as cp
import numpy as np
import os
from .ring_tables import HealpixRingTables

_kern_cache = {}
def _get_fun(src, fun):
    key = (src, fun)
    if key in _kern_cache: return _kern_cache[key]
    here = os.path.dirname(__file__)
    with open(os.path.join(here, src), "r", encoding="utf-8") as f:
        code = f.read()
    mod = cp.RawModule(code=code, options=("--std=c++11",), name_expressions=(fun,))
    fn = mod.get_function(fun)
    _kern_cache[key] = fn
    return fn

def _locate_rings_from_z(z, z_edges):
    z = cp.asarray(z, dtype=cp.float64)
    z_edges = cp.asarray(z_edges, dtype=cp.float64)
    Nr = z_edges.size - 1
    idx = cp.searchsorted(-z_edges, -z, side='right') - 1
    idx = cp.clip(idx, 0, Nr-1)
    return idx.astype(cp.int32)

class RingMapperGPU:
    def __init__(self, nside:int):
        self.tables = HealpixRingTables(nside)
        self._ang2pix = _get_fun("coords_ring.cu", "ang2pix_ring_with_rings")
        self._pix2ang = _get_fun("coords_ring.cu", "pix2ang_ring_batch")

    def ang2pix(self, theta, phi):
        theta = cp.asarray(theta, dtype=cp.float64)
        phi = cp.asarray(phi, dtype=cp.float64)
        ring_idx = _locate_rings_from_z(cp.cos(theta), self.tables.z_edges)
        N = int(theta.size)
        out = cp.empty((N,), dtype=cp.int64)
        threads=256
        blocks=(N+threads-1)//threads or 1
        self._ang2pix((blocks,), (threads,), (theta, phi, ring_idx, N,
            self.tables.ring_start, self.tables.nphi, self.tables.phi_center0, out))
        return out

    def pix2ang(self, ipix):
        ipix = cp.asarray(ipix, dtype=cp.int64)
        N = int(ipix.size)
        out_theta = cp.empty((N,), dtype=cp.float64)
        out_phi = cp.empty((N,), dtype=cp.float64)
        threads=256
        blocks=(N+threads-1)//threads or 1
        self._pix2ang((blocks,), (threads,), (ipix, N,
            self.tables.theta_centers, self.tables.ring_start, self.tables.nphi, self.tables.phi_center0,
            out_theta, out_phi))
        return out_theta, out_phi

    def weights_dz(self):
        return self.tables.weights_dz()


def vec2pix_ring_gpu(x, y, z, mapper: 'RingMapperGPU'):
    import cupy as cp, numpy as np
    x = cp.asarray(x, dtype=cp.float64)
    y = cp.asarray(y, dtype=cp.float64)
    z = cp.asarray(z, dtype=cp.float64)
    theta = cp.arccos(cp.clip(z, -1.0, 1.0))
    phi = cp.arctan2(y, x)
    phi = cp.where(phi < 0, phi + 2*np.pi, phi)
    return mapper.ang2pix(theta, phi)
