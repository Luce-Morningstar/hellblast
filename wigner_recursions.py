import cupy as cp
from .wigner_full import _d_l_mmprime

def d_l_mmprime(beta, l, method='auto'):
    # For now, delegate to explicit-sum builder for robustness.
    # TODO: Implement Trapani-Navaza or Risbo recurrences with chunked normalization and shared-memory tiles.
    return _d_l_mmprime(beta, l)
