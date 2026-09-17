from __future__ import annotations
from typing import TYPE_CHECKING
import math


from terraflow.test_case.abstract import AbstractAdvectionModel2DTestCase


if TYPE_CHECKING:
    from terraflow.util.planet import PlanetParameterSet


class DeformationalFlowTestCase(AbstractAdvectionModel2DTestCase):
    """Deformational flow test case for advection dynamics."""
    case_n: int
    period: float
    lon1  : float
    lat1  : float 
    lon2  : float 
    lat2  : float 

    def __init__(self, case_n: int, para: PlanetParameterSet):
        self.case_n = case_n
        self.alias_name = f"adv_dc{case_n}"
        self.name = f"Deformational Flow Case {case_n}"
        self.description = f"Deformational flow test case {case_n} for advection dynamics."
        self.references = (
            "Blossey, P. N. & Durran, D. R. "
            "Selective monotonicity preservation in scalar advection. "
            "J. Comput. Phys. 227, 5160–5183 (2008).",
        )
        self.keywords = ("advection", "deformational flow", "atmosphere")
        self.period = 12 * 86400.0
        if case_n == 1:
            self.lon1 = math.pi
            self.lat1 = math.pi / 3.0
            self.lon2 = math.pi
            self.lat2 = -math.pi / 3.0
        elif case_n == 3:
            self.lon1 = math.pi * 3.0 / 4.0
            self.lat1 = 0.0
            self.lon2 = math.pi * 5.0 / 4.0
            self.lat2 = 0.0
        elif case_n == 2 or case_n == 4:
            self.lon1 = math.pi * 5.0 / 6.0
            self.lat1 = 0.0
            self.lon2 = math.pi * 7.0 / 6.0
            self.lat2 = 0.0
        else:
            raise ValueError(
                f"Invalid case number: {case_n}. Supported cases are from 1 to 4.")
        super().__init__(para)


class DeformationalFlowCase1(DeformationalFlowTestCase):
    """Deformational flow test case 1 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(case_n=1, para=para)


class DeformationalFlowCase2(DeformationalFlowTestCase):
    """Deformational flow test case 2 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(case_n=2, para=para)


class DeformationalFlowCase3(DeformationalFlowTestCase):
    """Deformational flow test case 3 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(case_n=3, para=para)


class DeformationalFlowCase4(DeformationalFlowTestCase):
    """Deformational flow test case 4 for advection dynamics."""

    def __init__(self, para: PlanetParameterSet):
        super().__init__(case_n=4, para=para)