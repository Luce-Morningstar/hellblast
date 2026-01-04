import cupy as cp, os

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

def wigner_d_full(beta, l, batch=1):
    beta = cp.asarray(beta, dtype=cp.float64).ravel()
    B = int(beta.size)
    ldim = 2*l + 1
    out = cp.empty((B, ldim, ldim), dtype=cp.float64)
    fn = _get_fun("wigner_d_fast.cu", "wigner_d_full_tiled")
    threads = 64
    blocks = (ldim, ldim, B)
    sh = 0
    fn(blocks, (threads,), (beta, B, int(l), out), shared_mem=sh)
    return out
