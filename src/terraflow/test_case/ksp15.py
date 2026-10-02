from __future__ import annotations
from typing import TYPE_CHECKING
import math


from terraflow.test_case.abstract import AbstractBaroclinicModelTestCase


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet


class KSP15TestCase(AbstractBaroclinicModelTestCase):
    """Base class for the KSP15 reduced-radius mountain-wave tests."""
    teq : float
    peq : float
    ueq : float
    X   : float
    h0  : float
    d0  : float
    xi0 : float
    lonc: float
    latc: float

    def __init__(self, para: PlanetParameterSet):
        self.teq  = 300.0
        self.peq  = 1.0e5
        self.ueq  = 20.0
        self.X    = 166.7
        self.h0   = 0.0
        self.d0   = 5000.0
        self.xi0  = 4000.0
        self.lonc = math.pi
        self.latc = 0.0
        super().__init__(para)
        if not para.has_modified():
            self.set_params()
        elif para._modified_by != self.alias_name:
            raise ValueError(
                f"para already modified by {para._modified_by}, "
                f"cannot reuse for {self.alias_name}!"
            )

    def set_params(self) -> None:
        self.para.omega.d      = 0.0
        self.para.radius.d     = self.para.radius.unwrap() / self.X
        self.para._modified    = True
        self.para._modified_by = self.alias_name


class KSP1501TestCase(KSP15TestCase):

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "ksp15_01"
        self.name = "KSP15-01"
        self.description = (
            "Klemp, Skamarock, and Park 2015 reduced-radius hydrostatic "
            "mountain-wave test with a zonal ridge."
        )
        self.references = (
            "Klemp, J. B., Skamarock, W. C. & Park, S.-H. Idealized global nonhydrostatic "
            "atmospheric test cases on a reduced-radius sphere. Journal of Advances in "
            "Modeling Earth Systems 7, 1155-1177 (2015).",
        )
        self.keywords = ("baroclinic", "nonhydrostatic", "mountain wave", "KSP15")
        super().__init__(para)
        self.h0 = 25.0


class KSP1502TestCase(KSP15TestCase):

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "ksp15_02"
        self.name = "KSP15-02"
        self.description = (
            "Klemp, Skamarock, and Park 2015 reduced-radius hydrostatic "
            "mountain-wave test with an isolated mountain."
        )
        self.references = (
            "Klemp, J. B., Skamarock, W. C. & Park, S.-H. Idealized global nonhydrostatic "
            "atmospheric test cases on a reduced-radius sphere. Journal of Advances in "
            "Modeling Earth Systems 7, 1155-1177 (2015).",
        )
        self.keywords = ("baroclinic", "nonhydrostatic", "mountain wave", "KSP15")
        super().__init__(para)
        self.h0 = 250.0
