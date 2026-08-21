#!/usr/bin/env python3
"""Generate LRIS-2 long-slit spectral trace files.

PROVISIONAL GEOMETRY. Everything here is a straight, untilted, linearly
dispersed trace, which is what was specified as a starting point. Real gratings
are not linear and real traces are neither straight nor untilted. Rerun this
script with better numbers when they exist.

Assumptions, all of them stated rather than measured:
  - dispersion runs along CCD COLUMNS, i.e. wavelength maps to y
  - the spectrum fills all 4080 unbinned pixels of the dispersion axis
  - the central wavelength lands at the detector centre for a centred slit
  - dispersion is linear in Angstrom per pixel
  - no tilt and no curvature: x depends only on slit position, y only on wavelength

Central wavelengths are taken as the midpoint of each arm's coverage in
LRIS2_ETC/Instrument.py (R400 5500-9500 A, B600 3100-5700 A). Those are the
least arbitrary values available, but they are NOT a measured grating setting.
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table

PIX_MM      = 0.015     # mm per unbinned pixel
NPIX_DISP   = 4080      # pixels along the dispersion axis
PLATE_SCALE = 10.0      # arcsec per mm at the detector
SLIT_HALF   = 12.0      # arcsec, half of the 24 arcsec slit

N_XI, N_WAVE = 20, 100

GRATINGS = {
    #          A/pix  central wavelength (A)
    "R400": (1.13, 7500.0),
    "B600": (0.62, 4400.0),
}


def make_trace_table(ang_per_pix, cwave_ang):
    half_mm  = 0.5 * NPIX_DISP * PIX_MM              # +/- 30.6 mm
    ang_p_mm = ang_per_pix / PIX_MM
    y   = np.linspace(-half_mm, half_mm, N_WAVE)     # mm, dispersion axis
    lam = (cwave_ang + y * ang_p_mm) / 1e4           # micron
    xi  = np.linspace(-SLIT_HALF, SLIT_HALF, N_XI)   # arcsec along the slit
    x   = xi / PLATE_SCALE                           # mm, spatial axis

    XI, LAM = np.meshgrid(xi, lam, indexing="ij")
    X,  Y   = np.meshgrid(x,  y,   indexing="ij")
    tbl = Table(data=[LAM.ravel(), XI.ravel(), X.ravel(), Y.ravel()],
                names=["wavelength", "xi", "x", "y"])
    for col, unit in zip(tbl.columns.values(), ["um", "arcsec", "mm", "mm"]):
        col.unit = unit
    return tbl, lam


def build(name, ang_per_pix, cwave_ang, outdir=".."):
    tbl, lam = make_trace_table(ang_per_pix, cwave_ang)
    ext = fits.table_to_hdu(tbl)
    ext.header["EXTNAME"] = name

    toc = Table(data=[[name], [2], [0], [0]],
                names=["description", "extension_id",
                       "aperture_id", "image_plane_id"])
    toc_hdu = fits.table_to_hdu(toc)
    toc_hdu.header["EXTNAME"] = "TOC"

    pri = fits.PrimaryHDU()
    pri.header["ECAT"]  = 1
    pri.header["EDATA"] = 2
    pri.header.update({
        "AUTHOR":   "BPH",
        "DESCRIPT": f"LRIS-2 {name} long-slit trace, PROVISIONAL linear geometry",
        "SOURCE":   "dispersion from LRIS2_ETC Instrument.py",
        "DATE-CRE": "2026-08-21",
        "ANGPPIX":  ang_per_pix,
        "CWAVE":    cwave_ang,
        "SLITLEN":  2 * SLIT_HALF,
    })
    path = f"{outdir}/traces/LSS_{name}_TRACE.fits"
    fits.HDUList([pri, toc_hdu, ext]).writeto(path, overwrite=True)
    print(f"{path}: {len(tbl)} rows, {lam.min()*1e4:.1f}-{lam.max()*1e4:.1f} A "
          f"({ang_per_pix} A/pix, centre {cwave_ang:.0f} A)")


if __name__ == "__main__":
    for gname, (app, cw) in GRATINGS.items():
        build(gname, app, cw)
