import cupy as cp, os, numpy as np

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

def pix2ang_nest_gpu(nside:int, ipnest):
    ipnest = cp.asarray(ipnest, dtype=cp.int64)
    N = int(ipnest.size)
    out_theta = cp.empty((N,), dtype=cp.float64)
    out_phi = cp.empty((N,), dtype=cp.float64)
    fn = _get_fun("nested_pix2ang.cu", "pix2ang_nest_batch")
    threads = 256
    blocks = (N + threads - 1)//threads or 1
    fn((blocks,), (threads,), (ipnest.astype(cp.uint64), N, int(nside), out_theta, out_phi))
    return out_theta, out_phi

def pix2vec_nest_analytic_gpu(nside:int, ipnest):
    th, ph = pix2ang_nest_gpu(nside, ipnest)
    x = cp.sin(th)*cp.cos(ph); y = cp.sin(th)*cp.sin(ph); z = cp.cos(th)
    return x, y, z
