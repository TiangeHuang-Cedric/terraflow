from terraflow.discretization.abstract import (
    AbstractEquationTermAuxiliaryDataset
)
from terraflow.discretization.fvm.scvt.trisk.common.coriolis.trisk import TriSKCoriolisForceAtEdgeData


class CoriolisForceAtEdgeData(AbstractEquationTermAuxiliaryDataset):
    trisk: TriSKCoriolisForceAtEdgeData

    @classmethod
    def empty(cls) -> "CoriolisForceAtEdgeData":
        return cls(
            trisk=TriSKCoriolisForceAtEdgeData.empty()
        )
