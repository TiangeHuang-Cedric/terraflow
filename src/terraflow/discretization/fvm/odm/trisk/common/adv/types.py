from typing import Self

from terraflow.util import typing
from terraflow.discretization.abstract import AbstractEquationTermAuxiliaryDataset
from terraflow.discretization.fvm.odm.trisk.common.adv.flux_reconstruct import FluxReconstructAuxiliaryData

Float = typing.Float
Array = typing.Array


class AdvectionAuxiliaryDataset(AbstractEquationTermAuxiliaryDataset):
    # Variables needed in WRF flux reconstruct scheme
    edp2nd: FluxReconstructAuxiliaryData
    edp4th: FluxReconstructAuxiliaryData
    edd2nd: FluxReconstructAuxiliaryData
    edd4th: FluxReconstructAuxiliaryData

    @classmethod
    def empty(cls) -> Self:
        return cls(
            edp2nd=FluxReconstructAuxiliaryData.empty(),
            edp4th=FluxReconstructAuxiliaryData.empty(),
            edd2nd=FluxReconstructAuxiliaryData.empty(),
            edd4th=FluxReconstructAuxiliaryData.empty(),
        )
