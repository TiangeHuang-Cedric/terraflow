from typing import Self

from terraflow.util import typing
from terraflow.discretization.abstract import AbstractEquationTermAuxiliaryDataset
from terraflow.discretization.fvm.scvt.trisk.common.adv.flux_reconstruct import FluxReconstructAuxiliaryData
from terraflow.discretization.fvm.scvt.trisk.common.adv.fct import FluxCorrectedTransportData

Float = typing.Float
Array = typing.Array


class AdvectionAuxiliaryDataset(AbstractEquationTermAuxiliaryDataset):
    # Variables needed in WRF flux reconstruct scheme
    edp2nd: FluxReconstructAuxiliaryData
    edp4th: FluxReconstructAuxiliaryData
    edd2nd: FluxReconstructAuxiliaryData
    edd4th: FluxReconstructAuxiliaryData

    # Variables needed in FCT limiter
    fct: FluxCorrectedTransportData

    @classmethod
    def empty(cls) -> Self:
        return cls(
            edp2nd=FluxReconstructAuxiliaryData.empty(),
            edp4th=FluxReconstructAuxiliaryData.empty(),
            edd2nd=FluxReconstructAuxiliaryData.empty(),
            edd4th=FluxReconstructAuxiliaryData.empty(),
            fct=FluxCorrectedTransportData.empty(),
        )
