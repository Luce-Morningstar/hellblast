import cupy as cp
import numpy as np

KEY_KEEP = [
    "NSIDE","ORDERING","COORDSYS","EXTNAME","OBJECT","DATE","CREATOR","HISTORY","COMMENT",
    "TTYPE1","TTYPE2","TTYPE3","BUNIT","TELESCOP","INSTRUME","OBS_MODE"
]

def _ensure_astropy():
    try:
        import astropy.io.fits as fits  # noqa: F401
        return True
    except Exception:
        return False

def _copy_header(dst, src):
    for k in src.keys():
        if k in KEY_KEEP or (k and k.startswith("HIERARCH")):
            try:
                dst[k] = src[k]
            except Exception:
                pass

def write_tqu_fits(path, T=None, Q=None, U=None, nside=None, ordering="RING",
                   coordsys="G", overwrite=True, extra_header=None):
    if not _ensure_astropy():
        raise ImportError("astropy is required for FITS I/O. Try pip install astropy.")
    import astropy.io.fits as fits
    hdus = [fits.PrimaryHDU()]
    for arr, name in [(T,"T"),(Q,"Q"),(U,"U")]:
        if arr is None: continue
        arr_host = cp.asnumpy(cp.asarray(arr))
        hdu = fits.ImageHDU(arr_host, name=name)
        if nside is not None: hdu.header["NSIDE"] = int(nside)
        hdu.header["ORDERING"] = str(ordering).upper()
        hdu.header["COORDSYS"] = coordsys
        if extra_header:
            for k,v in extra_header.items(): hdu.header[k] = v
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path

def read_tqu_fits(path, to_gpu=True):
    if not _ensure_astropy():
        raise ImportError("astropy is required for FITS I/O. Try pip install astropy.")
    import astropy.io.fits as fits
    with fits.open(path, memmap=True) as hdul:
        T = Q = U = None
        meta = {}
        for hdu in hdul[1:]:
            name = (hdu.name or "").upper()
            data = np.array(hdu.data, copy=False)
            if name == "T": T = data
            elif name == "Q": Q = data
            elif name == "U": U = data
            # propagate some header keys
            for k in ["NSIDE","ORDERING","COORDSYS"]:
                if k in hdu.header: meta[k] = hdu.header[k]
    if to_gpu:
        T = None if T is None else cp.asarray(T)
        Q = None if Q is None else cp.asarray(Q)
        U = None if U is None else cp.asarray(U)
    return (T,Q,U), meta
