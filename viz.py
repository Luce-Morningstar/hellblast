import cupy as cp
import numpy as np

def _moll_inverse(X, Y):
    # X in [-2*sqrt2, 2*sqrt2], Y in [-sqrt2, sqrt2]
    r2 = np.sqrt(2.0)
    th = cp.arcsin(cp.clip(Y / r2, -1.0, 1.0))      # theta in Mollweide param
    lat = cp.arcsin(cp.clip((2*th + cp.sin(2*th)) / np.pi, -1.0, 1.0))   # geographic latitude
    lon = (np.pi * X) / (2.0 * r2 * cp.cos(th).clip(1e-12))             # longitude in [-pi, pi]
    return lat, lon

def mollview(m, nside=None, xsize=1200, ysize=None, nest=False, title=None, cmap='viridis',
             vmin=None, vmax=None, return_image=False, save=None, interpolation='bilinear'):
    """Render a RING or NESTED map to a Mollweide image.
    - m: 1D map (GPU or CPU). If CPU, it will be copied to GPU.
    - nside: inferred from len(m) if None.
    - xsize: image width in pixels; ysize inferred to keep aspect if None.
    - nest: if True, treat input as NESTED; otherwise RING.
    - interpolation: 'nearest' or 'bilinear' (GPU bilinear for RING; nearest for NESTED by default).
    """
    import matplotlib.pyplot as plt  # for saving/colormap only
    m = cp.asarray(m)
    npix = int(m.size)
    if nside is None:
        nside2 = npix//12
        nside = int(np.sqrt(nside2))
    if ysize is None:
        # Mollweide aspect: width:height = 2:1
        ysize = xsize // 2

    # Build image grid (center-of-pixel sampling)
    xs = cp.linspace(-2*np.sqrt(2)+1e-9, 2*np.sqrt(2)-1e-9, xsize, dtype=cp.float64)
    ys = cp.linspace(-np.sqrt(2)+1e-9, np.sqrt(2)-1e-9, ysize, dtype=cp.float64)
    X, Y = cp.meshgrid(xs, ys)  # [H,W]
    lat, lon = _moll_inverse(X, Y)  # radians
    theta = (np.pi/2) - lat
    phi = (lon + 2*np.pi) % (2*np.pi)

    # Sample the map
    if not nest:
        from .ring_tables import HealpixRingTables
        from .udgrade import get_interp_val_ring
        tabs = HealpixRingTables(int(nside))
        if interpolation == 'bilinear':
            vals = get_interp_val_ring(theta.ravel(), phi.ravel(), m, tabs)
        else:
            from .coords_ring import RingMapperGPU
            mapper = RingMapperGPU(int(nside))
            ip = mapper.ang2pix(theta.ravel(), phi.ravel())
            vals = m[ip]
    else:
        from .nested_gpu import ang2pix_nest_gpu
        ip = ang2pix_nest_gpu(int(nside), theta.ravel(), phi.ravel())
        vals = m[ip]

    img = vals.reshape(ysize, xsize)

    # colorize and save/return
    img_host = cp.asnumpy(img)
    if vmin is None: vmin = np.nanpercentile(img_host, 0.5)
    if vmax is None: vmax = np.nanpercentile(img_host, 99.5)
    fig, ax = plt.subplots(figsize=(xsize/150, ysize/150), dpi=150)
    ax.imshow(img_host, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=[-2*np.sqrt(2), 2*np.sqrt(2), -np.sqrt(2), np.sqrt(2)])
    ax.axis('off')
    if title: ax.set_title(title)
    if save:
        fig.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    if return_image:
        return img_host
    return None


def _lonlat_to_vec(lon, lat):
    x = cp.cos(lat)*cp.cos(lon); y = cp.cos(lat)*cp.sin(lon); z = cp.sin(lat)
    return x,y,z

def _rotate_to_center(lon, lat, lon0, lat0):
    # rotate so that (lon0,lat0) maps to (0,0) (center)
    # build local coords by rotating around z then y
    # do via vectors for stability
    x,y,z = _lonlat_to_vec(lon, lat)
    x0,y0,z0 = _lonlat_to_vec(lon0, lat0)
    # Orthonormal basis at center
    ez = cp.stack([x0,y0,z0], axis=0)           # pointing to center
    ex = cp.stack([-y0, x0, cp.zeros_like(x0)], axis=0)
    ex = ex / cp.sqrt(cp.sum(ex**2, axis=0)).clip(1e-12)
    ey = cp.cross(ez.T, ex.T).T
    v = cp.stack([x,y,z], axis=0)
    # coordinates in local basis
    vx = cp.sum(v*ex, axis=0); vy = cp.sum(v*ey, axis=0); vz = cp.sum(v*ez, axis=0)
    return vx, vy, vz

def orthview(m, nside=None, xsize=800, fov=np.pi/2, lon0=0.0, lat0=0.0, nest=False,
             title=None, cmap='viridis', vmin=None, vmax=None, return_image=False, save=None, interpolation='bilinear'):
    """Orthographic projection centered at (lon0,lat0), showing visible hemisphere (fov<=pi)."""
    import matplotlib.pyplot as plt
    m = cp.asarray(m)
    if nside is None: nside = int(cp.sqrt(m.size//12))
    H = W = xsize
    u = cp.linspace(-1,1,W); v = cp.linspace(-1,1,H)
    X,Y = cp.meshgrid(u,v)
    mask = X*X + Y*Y <= 1.0
    vx = X; vy = Y; vz = cp.sqrt(cp.maximum(0.0, 1.0 - X*X - Y*Y))
    # rotate back to sphere around center
    lon = cp.arctan2(vy, vx); lat = cp.arcsin(vz)
    lon_c = cp.asarray(lon0); lat_c = cp.asarray(lat0)
    # Convert local coords to global lon/lat
    # Build global vector from local basis at center
    x0,y0,z0 = _lonlat_to_vec(lon_c, lat_c)
    ez = cp.stack([x0,y0,z0], axis=0)
    ex = cp.stack([-y0, x0, cp.zeros_like(x0)], axis=0)
    ex = ex / cp.sqrt(cp.sum(ex**2)).clip(1e-12)
    ey = cp.cross(ez.T, ex.T).T
    vloc = cp.stack([cp.cos(lat)*cp.cos(lon), cp.cos(lat)*cp.sin(lon), cp.sin(lat)], axis=0)
    vglob = ex[:,None,None]*vloc[0] + ey[:,None,None]*vloc[1] + ez[:,None,None]*vloc[2]
    xg, yg, zg = vglob[0], vglob[1], vglob[2]
    th = cp.arccos(cp.clip(zg, -1.0, 1.0)); ph = cp.mod(cp.arctan2(yg, xg), 2*cp.pi)
    # sample
    if not nest:
        from .ring_tables import HealpixRingTables
        from .udgrade import get_interp_val_ring
        tabs = HealpixRingTables(int(nside))
        vals = get_interp_val_ring(th.ravel(), ph.ravel(), m, tabs)
    else:
        from .nested_gpu import ang2pix_nest_gpu
        ip = ang2pix_nest_gpu(int(nside), th.ravel(), ph.ravel())
        vals = m[ip]
    img = cp.full((H,W), np.nan, dtype=cp.float64)
    img = img.reshape(-1)
    img[:] = vals
    img = img.reshape(H,W)
    img_host = cp.asnumpy(img)
    import numpy as np
    if vmin is None: vmin = np.nanpercentile(img_host, 0.5)
    if vmax is None: vmax = np.nanpercentile(img_host, 99.5)
    fig, ax = plt.subplots(figsize=(W/150, H/150), dpi=150)
    ax.imshow(img_host, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off'); 
    if title: ax.set_title(title)
    if save: fig.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    if return_image: return img_host
    return None

def gnomview(m, nside=None, xsize=800, fov=np.deg2rad(60.0), lon0=0.0, lat0=0.0, nest=False,
             title=None, cmap='viridis', vmin=None, vmax=None, return_image=False, save=None, interpolation='bilinear'):
    """Gnomonic (tangent-plane) projection centered at lon0,lat0 with given field-of-view (radians)."""
    import matplotlib.pyplot as plt
    m = cp.asarray(m)
    if nside is None: nside = int(cp.sqrt(m.size//12))
    H = W = xsize
    # plane coords in radians: X, Y in [-tan(fov/2), tan(fov/2)]
    t = cp.tan(fov/2)
    X = cp.linspace(-t, t, W); Y = cp.linspace(-t, t, H)
    XX, YY = cp.meshgrid(X, Y)
    rho = cp.sqrt(XX*XX + YY*YY)
    c = cp.arctan(rho)
    lonc = cp.asarray(lon0); latc = cp.asarray(lat0)
    # Inverse gnomonic to lon/lat
    sinlat = cp.cos(c)*cp.sin(latc) + (YY*cp.sin(c)*cp.cos(latc))/cp.where(rho==0, 1.0, rho)
    lat = cp.arcsin(cp.clip(sinlat, -1.0, 1.0))
    lon = lonc + cp.arctan2(XX*cp.sin(c), rho*cp.cos(latc)*cp.cos(c) - YY*cp.sin(latc)*cp.sin(c))
    lon = cp.mod(lon, 2*cp.pi)
    theta = (cp.pi/2) - lat
    phi = lon
    # sample
    if not nest:
        from .ring_tables import HealpixRingTables
        from .udgrade import get_interp_val_ring
        tabs = HealpixRingTables(int(nside))
        vals = get_interp_val_ring(theta.ravel(), phi.ravel(), m, tabs)
    else:
        from .nested_gpu import ang2pix_nest_gpu
        ip = ang2pix_nest_gpu(int(nside), theta.ravel(), phi.ravel())
        vals = m[ip]
    img = vals.reshape(H,W)
    img_host = cp.asnumpy(img)
    import numpy as np
    if vmin is None: vmin = np.nanpercentile(img_host, 0.5)
    if vmax is None: vmax = np.nanpercentile(img_host, 99.5)
    fig, ax = plt.subplots(figsize=(W/150, H/150), dpi=150)
    ax.imshow(img_host, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    if title: ax.set_title(title)
    if save: fig.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    if return_image: return img_host
    return None
