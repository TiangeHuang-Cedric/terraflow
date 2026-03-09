from typing import TypeVar
import jax.numpy as jnp
from loguru import logger

from terraflow.util import typing
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.mesh.abstract import AbstractMesh
from terraflow.layout.abstract import AbstractLayout
from terraflow.layout.abstract import VertLoc
from terraflow.layout.abstract import CoordinateInfo, CoordinateSystem
from terraflow.vert_coord.hybrid_coord import HybridVertCoordinate

Array = typing.Array

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)


class HybridVertCoordLayout(AbstractLayout):
    def __init__(self):
        logger.success("Initialized HybridVertCoordLayout.")

    @staticmethod
    def alias_name() -> str:
        return "hybridVert"

    def shape(
        self,
        coord: CoordinateSystem[
            HM,
            VM,
            HLO,
            "HybridVertCoordLayout"
        ],
        loc: str,
    ) -> tuple[int, ...]:
        coord_info = self.get_layout(coord, loc).vertical_coord
        if coord_info is None:
            raise ValueError(
                f"Vertical coordinate information is missing for location: {loc}"
            )
        return coord_info.shape

    def get_layout(
        self,
        coord: CoordinateSystem[
            HM,
            VM,
            HLO,
            "HybridVertCoordLayout"
        ],
        loc: str,
        *args,
    ) -> CoordinateInfo:
        if isinstance(coord.vert, HybridVertCoordinate):
            if loc == VertLoc.Cell:
                vert_point = coord.vert.get_full_lev()
            elif loc == VertLoc.Lev:
                vert_point = coord.vert.get_half_lev()
            elif loc == VertLoc.Sfc:
                vert_point = jnp.array([])
            else:
                raise ValueError(
                    f"Unsupported vertical location for hybrid vertical coordinate layout: {loc}"
                )
        else:
            raise ValueError(
                f"Unsupported vertical coordinate type: {type(coord.vert)}"
            )
        return CoordinateInfo(
            vertical_coord=vert_point,
        )
