"""
Hellblast - GPU-accelerated HEALPix spherical harmonics library

A complete CuPy/CUDA replacement for healpy, providing:
- Spherical harmonic transforms (map2alm, alm2map)
- Power spectrum estimation (anafast)
- Spin-weighted harmonics for polarization (E/B modes)
- Wigner-d matrices and alm rotations
- Full HEALPix geometry (RING and NESTED)
- FITS I/O for maps and alm

Usage:
    import hellblast.hp_compat as hp  # Drop-in healpy replacement
    
    # Or use individual modules:
    from hellblast import map2alm, alm2map, anafast
"""

__version__ = "0.1.0"
__author__ = "Luce Morningstar"

# Core SHT
try:
    from .sht import map2alm_gpu
except ImportError:
    pass

try:
    from .isht import alm2map_gpu
except ImportError:
    pass

# Power spectrum
try:
    from .anafast import anafast_ring
except ImportError:
    pass

# alm operations
try:
    from .alm_ops import almxfl, alm2cl, synalm
except ImportError:
    pass

# Ring geometry
try:
    from .ring_tables import HealpixRingTables
except ImportError:
    pass

try:
    from .coords_ring import RingMapperGPU
except ImportError:
    pass

# Nested geometry  
try:
    from .nested_gpu import ang2pix_nest_gpu
except ImportError:
    pass

try:
    from .nested_pix2ang import pix2ang_nest_gpu
except ImportError:
    pass

# Udgrade
try:
    from .udgrade import (
        degrade_nested_gpu, 
        upgrade_nested_gpu,
        pack_rings_to_flat,
        unpack_flat_to_rings
    )
except ImportError:
    pass

# Smoothing
try:
    from .smoothing import smooth_alm_gaussian, gaussian_beam_fl
except ImportError:
    pass

try:
    from .beams import smooth_map, pixel_window_fl
except ImportError:
    pass

# Rotations
try:
    from .rotate_alm_euler import rotate_alm_euler_fast
except ImportError:
    pass

try:
    from .wigner_delta import rotate_alm_euler_risbo
except ImportError:
    pass

# Spin harmonics
try:
    from .spin import map2alm_spin_gpu, eb_from_qu_gpu, qu_from_eb_gpu
except ImportError:
    pass

# Visualization
try:
    from .viz import mollview, orthview, gnomview
except ImportError:
    pass

# FITS I/O
try:
    from .io_fits import read_map_fits, write_map_fits, read_alm_fits, write_alm_fits
except ImportError:
    pass

# Frame conversions
try:
    from .frames import convert_angles, map_convert_frame
except ImportError:
    pass

# Convenience aliases matching healpy API
map2alm = map2alm_gpu if 'map2alm_gpu' in dir() else None
alm2map = alm2map_gpu if 'alm2map_gpu' in dir() else None
anafast = anafast_ring if 'anafast_ring' in dir() else None

__all__ = [
    # Core
    'map2alm_gpu', 'alm2map_gpu', 'anafast_ring',
    'map2alm', 'alm2map', 'anafast',
    
    # alm ops
    'almxfl', 'alm2cl', 'synalm',
    
    # Geometry
    'HealpixRingTables', 'RingMapperGPU',
    'ang2pix_nest_gpu', 'pix2ang_nest_gpu',
    
    # Udgrade
    'degrade_nested_gpu', 'upgrade_nested_gpu',
    'pack_rings_to_flat', 'unpack_flat_to_rings',
    
    # Smoothing
    'smooth_alm_gaussian', 'gaussian_beam_fl',
    'smooth_map', 'pixel_window_fl',
    
    # Rotations
    'rotate_alm_euler_fast', 'rotate_alm_euler_risbo',
    
    # Spin
    'map2alm_spin_gpu', 'eb_from_qu_gpu', 'qu_from_eb_gpu',
    
    # Viz
    'mollview', 'orthview', 'gnomview',
    
    # I/O
    'read_map_fits', 'write_map_fits', 
    'read_alm_fits', 'write_alm_fits',
    
    # Frames
    'convert_angles', 'map_convert_frame',
]
