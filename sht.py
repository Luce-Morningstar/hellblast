try:
    from .sht_raw import map2alm_gpu  # CUDA RawKernel path
except Exception:
    from .sht_fallback import map2alm_gpu  # pure-CuPy fallback
