import cupy as cp
import numpy as np

def _ensure_astropy():
    try:
        import astropy.io.fits as fits  # noqa: F401
        return True
    except Exception:
        return False

def read_map_fits(path, hdu=1, to_gpu=True, dtype=np.float64):
    if not _ensure_astropy():
        raise ImportError("astropy is required for FITS I/O. Try `pip install astropy`.")
    import astropy.io.fits as fits
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data.astype(dtype, copy=False)
        arr = np.array(data, dtype=dtype, copy=False)
    return cp.asarray(arr, dtype=dtype) if to_gpu else arr

def write_map_fits(path, m, header=None, overwrite=True):
    if not _ensure_astropy():
        raise ImportError("astropy is required for FITS I/O. Try `pip install astropy`.")
    import astropy.io.fits as fits
    m_host = cp.asnumpy(m)
    hdu = fits.PrimaryHDU(m_host)
    if header is not None:
        for k,v in header.items():
            hdu.header[k] = v
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=overwrite)
    return path

def read_alm_fits(path, hdu=1, to_gpu=True, dtype=np.complex128):
    if not _ensure_astropy():
        raise ImportError("astropy is required for FITS I/O. Try `pip install astropy`.")
    import astropy.io.fits as fits
    with fits.open(path, memmap=True) as hdul:
        d = hdul[hdu].data
        # Expect columns REAL, IMAG or a complex vector
        if hasattr(d, 'names') and 'REAL' in d.names and 'IMAG' in d.names:
            arr = d['REAL'].astype(np.float64) + 1j*d['IMAG'].astype(np.float64)
        else:
            arr = np.array(d, dtype=dtype)
    return cp.asarray(arr, dtype=dtype) if to_gpu else arr

def write_alm_fits(path, alm, header=None, overwrite=True):
    if not _ensure_astropy():
        raise ImportError("astropy is required for FITS I/O. Try `pip install astropy`.")
    import astropy.io.fits as fits
    arr = cp.asnumpy(alm.astype(np.complex128))
    # Store as two float64 columns for broad compatibility
    real = arr.real.astype(np.float64)
    imag = arr.imag.astype(np.float64)
    cols = [
        fits.Column(name='REAL', format='D', array=real),
        fits.Column(name='IMAG', format='D', array=imag),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    if header is not None:
        for k,v in header.items():
            hdu.header[k] = v
    hdu.writeto(path, overwrite=overwrite)
    return path
