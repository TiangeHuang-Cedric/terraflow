from typing import TypeVar
import jax.numpy as jnp
from loguru import logger

from terraflow.util import typing
from terraflow.vert_coord.abstract import AbstractVertCoordinate
from terraflow.mesh.abstract import AbstractMesh
from terraflow.layout.abstract import AbstractLayout
from terraflow.layout.abstract import CoordinateInfo, CoordinateSystem
from terraflow.vert_coord.onelayer_coord import OneLayerVertCoordinate

Array = typing.Array

HM = TypeVar("HM", bound=AbstractMesh)
VM = TypeVar("VM", bound=AbstractVertCoordinate)
HLO = TypeVar("HLO", bound=AbstractLayout)


class OneLayerVertCoordLayout(AbstractLayout):
    def __init__(self):
        logger.success("Initialized OneLayerVertCoordLayout.")

    @staticmethod
    def alias_name() -> str:
        return "1LayerVert"

    def shape(
        self,
        coord: CoordinateSystem[
            HM,
            VM,
            HLO,
            "OneLayerVertCoordLayout"
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
            "OneLayerVertCoordLayout"
        ],
        loc: str,
        *args,
    ) -> CoordinateInfo:
        if isinstance(coord.vert, OneLayerVertCoordinate):
            vert_point = jnp.array([])
        else:
            raise ValueError(
                f"Unsupported vertical coordinate type: {type(coord.vert)}"
            )
        return CoordinateInfo(
            vertical_coord=vert_point,
        )
