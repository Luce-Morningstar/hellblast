import cupy as cp
import numpy as np
import os

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

class HealpixRingTables:
    def __init__(self, nside:int):
        assert nside >= 1 and (nside & (nside-1))==0, "nside must be power of 2"
        self.nside = int(nside)
        Nr = 4*self.nside - 1
        self.z_centers = cp.empty((Nr,), dtype=cp.float64)
        self.theta_centers = cp.empty((Nr,), dtype=cp.float64)
        self.nphi = cp.empty((Nr,), dtype=cp.int32)
        self.phi_center0 = cp.empty((Nr,), dtype=cp.float64)
        self.ring_start = cp.empty((Nr+1,), dtype=cp.int64)
        self.z_edges = cp.empty((Nr+1,), dtype=cp.float64)
        fn = _get_fun("ring_tables_healpix.cu", "build_ring_tables")
        threads=256
        blocks=(Nr+threads-1)//threads or 1
        fn((blocks,), (threads,), (self.nside,
                                  self.z_centers, self.theta_centers,
                                  self.nphi, self.phi_center0,
                                  self.ring_start, self.z_edges))
        self.Nr = Nr
        self.npix = int(self.ring_start[-1].get())
        assert self.npix == 12*self.nside*self.nside, f"npix mismatch {self.npix} vs 12 nside^2"

    def weights_dz(self):
        return self.z_edges[:-1] - self.z_edges[1:]

    def dump_host(self):
        return {
            "nside": self.nside,
            "Nr": self.Nr,
            "npix": self.npix,
            "z_centers": self.z_centers.get(),
            "theta_centers": self.theta_centers.get(),
            "nphi": self.nphi.get(),
            "phi_center0": self.phi_center0.get(),
            "ring_start": self.ring_start.get(),
            "z_edges": self.z_edges.get(),
        }
