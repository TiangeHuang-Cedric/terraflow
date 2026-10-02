from __future__ import annotations
from typing import TYPE_CHECKING
import math


from terraflow.test_case.abstract import AbstractBaroclinicModelTestCase


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet


class DCMIP31TestCase(AbstractBaroclinicModelTestCase):
    """DCMIP 3-1 nonhydrostatic gravity wave test case."""
    X    : float
    u0   : float
    teq  : float
    peq  : float
    d    : float
    lonc : float
    latc : float
    dpt  : float
    lz   : float
    N    : float
    N2   : float
    t0   : float

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "dcmip31"
        self.name = "DCMIP 3-1"
        self.description = (
            "DCMIP 3-1 baroclinic nonhydrostatic gravity wave test case "
            "on a reduced-radius, non-rotating sphere."
        )
        self.references = (
            "Ullrich, P. A. et al. DCMIP2012 Test Case Document.",
        )
        self.keywords = (
            "baroclinic",
            "nonhydrostatic",
            "gravity wave",
            "reduced radius",
            "DCMIP",
        )
        self.X    = 125.0
        self.u0   = 20.0
        self.teq  = 300.0
        self.peq  = 1.0e5
        self.d    = 5000.0
        self.lonc = 2.0 * math.pi / 3.0
        self.latc = 0.0
        self.dpt  = 1.0
        self.lz   = 20000.0
        self.N    = 0.01
        self.N2   = self.N ** 2
        super().__init__(para)
        if not para.has_modified():
            self.set_params()
        elif para._modified_by != self.alias_name:
            raise ValueError(
                f"para already modified by {para._modified_by}, "
                f"cannot reuse for {self.alias_name}!"
            )
        self.t0   = (
            self.para.gravity.unwrap() ** 2
            / self.N2
            / self.para.cpd.unwrap()
        )

    def set_params(self) -> None:
        self.para.omega.d      = 0.0
        self.para.radius.d     = self.para.radius.unwrap() / self.X
        self.para._modified    = True
        self.para._modified_by = self.alias_name
