"""Tests for the LRIS-2 instrument package

Each mode observes a V=15 star for 60 s. The expected values are regression
numbers measured on 2026-10-20 with the COO ScopeSim fork (including the
zodiacal-light fix on branch fix-zodiacal-ecliptic-coords) and the COO PALACE
airglow model. The imaging sky corresponds to a dark Maunakea sky of about
21.0 mag/arcsec2 in r and 22.2 in g. The long-slit values were updated on
2026-10-21, when the grating curves became grating-only (DE x PRISMS). All
star and sky values rose by about 1.29 on 2026-10-21, when the per-arm common
path from the ETC replaced the flat 0.70 camera and 0.998 collimator. They
then moved by 2 percent or less when the Keck mirrors and ADC were taken from
the ETC and the blue QE became the STA5100 prediction.

The Maunakea sky effects query SkyCalc over the network and need the COO
ScopeSim fork, so the tests are skipped when that fork is not installed.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import scopesim as sim
from scopesim.source.source_templates import star

PATH_HERE = Path(__file__).parent
sim.rc.__config__["!SIM.file.local_packages_path"] = str(PATH_HERE.parent.parent)

pytestmark = [
    pytest.mark.webtest,
    pytest.mark.slow,
    pytest.mark.skipif(not hasattr(sim.effects, "PalaceAirglowEmission"),
                       reason="needs the COO ScopeSim fork for the Maunakea sky"),
]

BIAS = 1800             # from LRIS-2_DET.yaml
DIT = 60                # [s] from default.yaml
BINNED_PIXEL = 0.30     # [arcsec] 0.15 arcsec pixels, 2x2 binned in imaging

# pylint: disable=missing-class-docstring,missing-function-docstring


@lru_cache(maxsize=None)
def observe_star(mode):
    """Bias-subtracted detector image of a V=15 star at the field centre"""
    cmds = sim.UserCommands(use_instrument="LRIS-2", set_modes=[mode],
                            properties={"!SIM.random.seed": 9001})
    lris2 = sim.OpticalTrain(cmds)
    lris2.observe(star(flux=15))
    return lris2.readout()[0][1].data.astype(float) - BIAS


@pytest.mark.parametrize("mode, shape", [("IMG_R", (2040, 2040)),
                                         ("IMG_B", (2048, 2048)),
                                         ("LSS_R", (4080, 4080)),
                                         ("LSS_B", (4096, 4096))])
def test_mode_reads_out_with_detector_shape(mode, shape):
    assert observe_star(mode).shape == shape


class TestImaging:
    @staticmethod
    def star_and_sky(mode):
        """Star [e-/s] in a 6 arcsec radius and sky [e-/s/arcsec2] around it"""
        img = observe_star(mode)
        yc, xc = np.unravel_index(img.argmax(), img.shape)
        r = np.hypot(*(np.indices(img.shape) - np.array([yc, xc])[:, None, None]))
        sky = np.median(img[(r > 30) & (r < 60)])
        star_counts = (img[r < 20] - sky).sum()
        return star_counts / DIT, sky / DIT / BINNED_PIXEL**2

    @pytest.mark.parametrize("mode, rate", [("IMG_R", 4.25e5), ("IMG_B", 6.52e5)])
    def test_star_count_rate(self, mode, rate):
        star_rate, _ = self.star_and_sky(mode)
        assert star_rate == pytest.approx(rate, rel=0.1)

    @pytest.mark.parametrize("mode, rate", [("IMG_R", 1776), ("IMG_B", 827)])
    def test_dark_sky_count_rate(self, mode, rate):
        _, sky_rate = self.star_and_sky(mode)
        assert sky_rate == pytest.approx(rate, rel=0.25)

    @pytest.mark.parametrize("mode", ["IMG_R", "IMG_B"])
    def test_aperture_lights_half_the_detector(self, mode):
        # The 10 x 5 arcmin entrance aperture covers 2000 of the ~4080
        # unbinned rows, so about half of the rows see sky.
        lit_rows = np.median(observe_star(mode), axis=1) > 100
        assert lit_rows.mean() == pytest.approx(0.5, abs=0.02)


class TestLongSlit:
    @staticmethod
    def trace_spectrum(mode):
        """Sky-subtracted star spectrum [e-] summed over the trace, per row"""
        img = observe_star(mode)
        xc = int(np.argmax(np.median(img, axis=0)))
        sky_cols = np.r_[xc - 60:xc - 30, xc + 30:xc + 60]
        sky = np.median(img[:, sky_cols], axis=1)
        return (img[:, xc - 15:xc + 16] - sky[:, None]).sum(axis=1)

    @pytest.mark.parametrize("mode, total", [("LSS_R", 3.37e7), ("LSS_B", 3.47e7)])
    def test_star_counts_through_slit(self, mode, total):
        assert self.trace_spectrum(mode).sum() == pytest.approx(total, rel=0.2)

    @pytest.mark.parametrize("mode, min_rows", [("LSS_R", 3500), ("LSS_B", 3000)])
    def test_spectrum_is_dispersed_along_rows(self, mode, min_rows):
        spec = self.trace_spectrum(mode)
        lit_rows = spec > 0.05 * np.percentile(spec, 99)
        assert lit_rows.sum() > min_rows
