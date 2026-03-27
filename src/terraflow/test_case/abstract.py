from abc import ABC
from typing import Dict, Union

from terraflow.planet import PlanetParameterSet


class AbstractTestCase(ABC):
    """Abstract class for dynamics test cases.

    This class uses Equinox Module which makes it compatible with JAX transformations.
    Static fields (para, name, description, etc.) are marked to exclude them from pytree,
    as they don't change during computation and don't need to be traced by JAX.
    """
    _para: PlanetParameterSet
    name: str
    description: str
    references: tuple[str, ...]
    keywords: tuple[str, ...]

    @property
    def para(self) -> PlanetParameterSet:
        return self._para

    def get_info(self) -> Dict[str, Union[str, tuple[str, ...]]]:
        return {
            "name": self.name,
            "description": self.description,
            "references": self.references,
            "keywords": self.keywords
        }
