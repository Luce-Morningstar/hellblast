import cupy as cp
import numpy as np
from .ring_tables import HealpixRingTables

def rotate_map_z(rings, tables: HealpixRingTables, delta_phi):
    """Rotate a ringed map about z by delta_phi (radians), GPU-only.
    rings: list[cp.ndarray] per ring (phi-major)
    tables: HealpixRingTables (to get per-ring nphi and phi0)
    delta_phi: scalar float
    Returns a new list of rings (same shapes)
    """
    out = []
    for r in range(int(tables.Nr)):
        f = cp.asarray(rings[r])
        npr = int(tables.nphi[r].get())
        # Fractional index shift: use FFT phase for exact circular shift
        F = cp.fft.fft(f)
        k = cp.fft.fftfreq(npr) * (2*np.pi)  # angular frequency per bin
        # shift by delta_phi corresponds to phase e^{-i m delta_phi} in Fourier domain, where m = bin index
        # For evenly spaced phi, bin index m corresponds to integer harmonics. Use k relation: exp(-i * m * delta)
        m = cp.arange(npr, dtype=cp.float64)
        phase = cp.exp(-1j * m * delta_phi)
        f_rot = cp.fft.ifft(F * phase).astype(f.dtype)
        out.append(cp.asrealarray(f_rot))
    return out
