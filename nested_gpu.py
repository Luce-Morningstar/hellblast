import cupy as cp
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

def ang2pix_nest_gpu(theta, phi, nside:int):
    theta = cp.asarray(theta, dtype=cp.float64)
    phi = cp.asarray(phi, dtype=cp.float64)
    N = int(theta.size)
    out = cp.empty((N,), dtype=cp.uint64)
    fn = _get_fun("nested_gpu.cu", "vec2pix_nest_batch")
    threads=256; blocks=(N+threads-1)//threads or 1
    fn((blocks,), (threads,), (theta, phi, N, int(nside), out))
    return out.astype(cp.int64)
