import cupy as cp
import numpy as np

def face_offset(face:int, nside:int)->int:
    return face * (nside*nside)

def part1by1_64(x):
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8))  & 0x00FF00FF00FF00FF
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2))  & 0x3333333333333333
    x = (x | (x << 1))  & 0x5555555555555555
    return x

def morton2d_64(ix:int, iy:int)->int:
    return (part1by1_64(iy) << 1) | part1by1_64(ix)

def encode_nested(face:int, ix:int, iy:int, nside:int)->int:
    return face_offset(face, nside) + morton2d_64(ix, iy)

class Ring2NestLUT:
    def __init__(self, nside:int, ring2nest_host=None):
        self.nside = int(nside)
        self.npix = 12*self.nside*self.nside
        if ring2nest_host is None:
            try:
                import healpy as hp
                ring = np.arange(self.npix, dtype=np.int64)
                nest = hp.ring2nest(self.nside, ring)
                ring2nest_host = nest.astype(np.int64)
            except Exception as e:
                raise RuntimeError("Provide ring2nest_host or install healpy to auto-build LUT") from e
        self.ring2nest = cp.asarray(ring2nest_host, dtype=cp.int64)
        inv = np.empty((self.npix,), dtype=np.int64)
        inv[ring2nest_host] = np.arange(self.npix, dtype=np.int64)
        self.nest2ring = cp.asarray(inv, dtype=cp.int64)

    def ring_to_nest(self, ipix_ring):
        ipix_ring = cp.asarray(ipix_ring, dtype=cp.int64)
        return self.ring2nest[ipix_ring]

    def nest_to_ring(self, ipix_nest):
        ipix_nest = cp.asarray(ipix_nest, dtype=cp.int64)
        return self.nest2ring[ipix_nest]
