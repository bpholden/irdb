#!/usr/bin/env python3
"""Generate LRIS-2 long-slit spectral trace files from the vendor spreadsheet.

Source: Dropbox/lris2/dispersive_elements/LRIS2 Spectral Coverage_v2.xlsx

The wavelength solution is taken DIRECTLY from the tabulated WAVE / Y POS
columns. Nothing is fitted, so the non-linear dispersion is carried exactly as
supplied rather than being approximated by a polynomial. Earlier versions of
this script fitted a Hermite cubic to a handful of endpoint numbers; that is no
longer necessary and the table supersedes it.

COORDINATE ROTATION. The spreadsheet (X,Y) are focal-plane positions and are
ROTATED with respect to the simulation axes. The mapping is:

    spreadsheet X  (+/-150" imaged, +/-90" usable)  <->  simulation y  <->  DISPERSION
    spreadsheet Y  (+/-300")                        <->  simulation x  <->  SPATIAL

So (-90", -300") and (+90", +300") are diagonally opposite corners. The
ApertureMask in LRIS-2.yaml is 600 x 300 arcsec, i.e. simulation x +/-300 and
y +/-150, and that +/-150 is what the +150"/-150" blocks correspond to. The
spectrograph does NOT work beyond +/-90" in spreadsheet X, so a slit may only
be positioned within simulation y = +/-90" even though the imaging field
extends to +/-150".

This is why the slit geometry is what it is: LongSlit0.7Arcsec.dat runs
-12 to +12 in x (the 24 arcsec LENGTH, spatial) and -0.35 to +0.35 in y (the
0.7 arcsec WIDTH, along dispersion, which is what sets the resolution).
Verified: the slit illuminates exactly 159 detector columns, and
24" / 0.1505"/pix = 159.

Only the (X,Y) = (0", 0") block is used, i.e. the field centre. The spreadsheet
also tabulates four off-centre field positions. Use the +90"/-90" blocks
(columns S and Y) for any future off-axis work; the +150"/-150" blocks
(columns G and M) are NOT to be used. Off-axis behaviour is not modelled here
at all: x depends only on slit position and y only on wavelength, so the trace
is straight and untilted.

GRATING COMPLEMENT. Only B600 and R400 can be modelled, because they are the
only two with efficiency curves. The B1200, B1300, R700 and R750 sheets in the
spreadsheet are OBSOLETE - those gratings were replaced by B1250-1 + B1250-2
and R725-1 + R725-2 - and no efficiency curves exist for the replacements yet.
Do not add the obsolete sheets. Add the new ones when their curves arrive.

Detector formats differ between the arms:
  red  STA5100, 4080 pixels, +/- 30.60 mm
  blue not yet finalised, assumed 4096 pixels, +/- 30.72 mm
The spreadsheet quotes nominal CCD edges of +/- 30.72 mm for both, so the red
arm loses a sliver at each end relative to the nominal.
"""
import numpy as np
import openpyxl
from astropy.io import fits
from astropy.table import Table

XLSX = ("/Users/holden/Dropbox/lris2/dispersive_elements/"
        "LRIS2 Spectral Coverage_v2.xlsx")

PIX_MM      = 0.015     # mm per unbinned pixel
PLATE_SCALE = 10.0      # arcsec per mm at the detector
SLIT_HALF   = 12.0      # arcsec, half of the 24 arcsec slit
N_XI        = 20

GRATINGS = {
    #        sheet             detector half-height (mm), npix
    "R400": ("RED_400lpmm",   30.60, 4080),
    "B600": ("BLUE_600lpmm",  30.72, 4096),
}


def read_solution(sheet):
    """Read WAVE (nm), Y POS (mm), DISPERSION (nm/pix) for the field centre."""
    ws = openpyxl.load_workbook(XLSX, data_only=True)[sheet]
    rows = []
    for r in range(3, ws.max_row + 1):
        w, y, d = (ws.cell(r, c).value for c in (1, 2, 3))
        if w is None or y is None:
            continue
        rows.append((float(w), float(y), float(d)))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def build(name, sheet, half_mm, npix, outdir=".."):
    wave, ypos, disp = read_solution(sheet)

    # sanity: the tabulated dispersion must agree with the derivative of Y POS
    disp_from_y = PIX_MM / np.gradient(ypos, wave)
    worst = np.nanmax(np.abs((disp_from_y - disp) / disp))
    assert worst < 0.05, f"{name}: dispersion column disagrees with Y POS by {worst:.1%}"

    on = (ypos >= -half_mm) & (ypos <= half_mm)
    wave, ypos, disp = wave[on], ypos[on], disp[on]

    xi = np.linspace(-SLIT_HALF, SLIT_HALF, N_XI)
    XI, LAM = np.meshgrid(xi, wave / 1e3, indexing="ij")     # nm -> micron
    X,  Y   = np.meshgrid(xi / PLATE_SCALE, ypos, indexing="ij")
    tbl = Table(data=[LAM.ravel(), XI.ravel(), X.ravel(), Y.ravel()],
                names=["wavelength", "xi", "x", "y"])
    for col, unit in zip(tbl.columns.values(), ["um", "arcsec", "mm", "mm"]):
        col.unit = unit

    ext = fits.table_to_hdu(tbl)
    ext.header["EXTNAME"] = name
    toc = Table(data=[[name], [2], [0], [0]],
                names=["description", "extension_id",
                       "aperture_id", "image_plane_id"])
    toc_hdu = fits.table_to_hdu(toc)
    toc_hdu.header["EXTNAME"] = "TOC"

    pri = fits.PrimaryHDU()
    pri.header["ECAT"], pri.header["EDATA"] = 1, 2
    pri.header.update({
        "AUTHOR":   "BPH",
        "DESCRIPT": f"LRIS-2 {name} long-slit trace from the vendor spreadsheet",
        "SOURCE":   f"LRIS2 Spectral Coverage_v2.xlsx, sheet {sheet}, (X,Y)=(0,0)",
        "DATE-CRE": "2026-08-21",
        "LAMLO":    wave.min(),
        "LAMHI":    wave.max(),
        "DISPLO":   disp.min(),
        "DISPHI":   disp.max(),
        "NPIXDET":  npix,
        "SOLUTION": "tabulated",
    })
    path = f"{outdir}/traces/LSS_{name}_TRACE.fits"
    fits.HDUList([pri, toc_hdu, ext]).writeto(path, overwrite=True)
    used = (ypos.max() - ypos.min()) / PIX_MM
    print(f"{path}: {wave.min():.0f}-{wave.max():.0f} nm, "
          f"dispersion {disp.min():.4f}-{disp.max():.4f} nm/pix, "
          f"y {ypos.min():+.3f} to {ypos.max():+.3f} mm, "
          f"{used:.0f} of {npix} px ({100*used/npix:.1f}%)")


if __name__ == "__main__":
    for gname, (sheet, half_mm, npix) in GRATINGS.items():
        build(gname, sheet, half_mm, npix)
