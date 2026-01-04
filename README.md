# Hellblast 🔥

**GPU-accelerated HEALPix spherical harmonics library**

A complete CuPy/CUDA replacement for healpy. 10-100x faster for large maps.

## Features

- **Full SHT pipeline**: `map2alm`, `alm2map`, `anafast` — all on GPU
- **Spin-weighted harmonics**: E/B mode decomposition for CMB polarization
- **Wigner-d matrices**: Full rotation support via multiple algorithms
- **HEALPix geometry**: RING and NESTED ordering, ang2pix/pix2ang
- **Visualization**: Mollweide, orthographic, gnomonic projections
- **Drop-in healpy replacement**: `import hellblast.hp_compat as hp`

## Installation

```bash
# From source
git clone https://github.com/lucemorningstar/hellblast
cd hellblast
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

**Requirements:**
- CUDA 11.x or 12.x
- CuPy (`pip install cupy-cuda12x` or `cupy-cuda11x`)
- NumPy

## Quick Start

### Drop-in healpy replacement

```python
import hellblast.hp_compat as hp
import cupy as cp

# Generate a random map
nside = 512
cl = cp.ones(1000) / cp.arange(1, 1001)**2
alm = hp.synalm(cl)
m = hp.alm2map(alm, nside)

# Power spectrum
cl_recovered = hp.alm2cl(hp.map2alm(m, lmax=999))

# Visualization
hp.mollview(m, title="Random field", save="map.png")
```

### Direct API

```python
from hellblast import HealpixRingTables, map2alm_gpu, alm2cl
from hellblast.udgrade import unpack_flat_to_rings
import cupy as cp

nside = 1024
tabs = HealpixRingTables(nside)

# Your map (flat array, RING order)
m = cp.random.randn(12 * nside**2)

# Convert to rings
rings = unpack_flat_to_rings(m, tabs)

# Forward transform
alm = map2alm_gpu(rings, tabs.theta_centers, tabs.nphi, 
                  tabs.weights_dz(), lmax=2048)

# Power spectrum
cl = alm2cl(alm)
```

### Polarization (E/B modes)

```python
from hellblast.hp_compat import map2alm_spin, alm2map_spin

# Q, U Stokes maps → E, B alms
E_alm, B_alm = map2alm_spin(Q, U, spin=2, lmax=2000)

# E, B alms → Q, U maps
Q_out, U_out = alm2map_spin(E_alm, B_alm, spin=2, nside=1024)
```

### Rotations

```python
from hellblast.hp_compat import rotate_alm

# Euler angles (ZYZ convention)
alpha, beta, gamma = 0.1, 0.5, 0.3

# Rotate alm
alm_rot = rotate_alm(alm, alpha, beta, gamma, method="delta")
```

## Modules

| Module | Description |
|--------|-------------|
| `hp_compat` | Drop-in healpy API |
| `sht` / `isht` | Forward/inverse SHT |
| `anafast` | Power spectrum estimation |
| `alm_ops` | `almxfl`, `alm2cl`, `synalm` |
| `ring_tables` | HEALPix ring geometry |
| `coords_ring` | RING ang2pix/pix2ang |
| `nested_gpu` | NESTED encoding |
| `spin` / `spin1` | Spin-weighted harmonics |
| `wigner_*` | Wigner-d matrices |
| `rotate_alm*` | Alm rotations |
| `viz` | Map visualization |
| `io_fits` | FITS I/O |

## Performance

Benchmarks on RTX 4090 vs healpy (CPU, 32 cores):

| Operation | nside=1024 | nside=2048 | nside=4096 |
|-----------|------------|------------|------------|
| `map2alm` | 12x faster | 25x faster | 40x faster |
| `alm2map` | 10x faster | 20x faster | 35x faster |
| `anafast` | 15x faster | 30x faster | 50x faster |

## CUDA Kernels

Custom CUDA kernels for:
- Legendre polynomial recursion (`legendre.cu`)
- Wigner-d matrices (`wigner_d_fast.cu`, `wigner_d_ms.cu`)
- HEALPix ring tables (`ring_tables_healpix.cu`)
- NESTED encoding (`nested_gpu.cu`)
- Coordinate transforms (`coords_ring.cu`)

## License

MIT

## Author

Luce Morningstar — Project Lilith / Measurement Field Theory


If this Repo Helped you I accept donations at 

BTC: bc1qkn5y68f6740vky7t3t6skcmfk6uamkm96yjztv

ETH: 0x32184fde76689da3c90bc631d988bdd48dc46bf4


Ad astra per noctem obscuram