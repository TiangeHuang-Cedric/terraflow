from typing import Self
from terraflow.discretization.abstract import AbstractEquationTermAuxiliaryDataset
from terraflow.discretization.fvm.scvt.trisk.common.damp.smag import SmagorinskyDampData


class DampData(AbstractEquationTermAuxiliaryDataset):
    smag: SmagorinskyDampData

    @classmethod
    def empty(cls) -> Self:
        return cls(
            smag=SmagorinskyDampData.empty()
        )
