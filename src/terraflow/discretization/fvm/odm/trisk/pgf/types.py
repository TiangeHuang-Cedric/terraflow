from typing import Self
from terraflow.discretization.abstract import AbstractEquationTermAuxiliaryDataset
from terraflow.discretization.fvm.odm.trisk.pgf.ptb import RefProfileData


class PressureGradientForceDataset(AbstractEquationTermAuxiliaryDataset):
    pro: RefProfileData

    @classmethod
    def empty(cls) -> Self:
        return cls(
            pro=RefProfileData.empty()
        )
