from abc import ABC
from typing import Dict, Union


from terraflow.util.planet import PlanetParameterSet

class AbstractTestCase(ABC):
    alias_name : str
    para       : PlanetParameterSet
    alias_name : str            
    name       : str             
    description: str            
    references : tuple[str, ...]
    keywords   : tuple[str, ...]
    
    def __init__(
        self,
        para: PlanetParameterSet
    ):
        self.para = para

    def get_info(self) -> Dict[str, Union[str, tuple[str, ...]]]:
        return {
            "name": self.name,
            "alias_name": self.alias_name,
            "description": self.description,
            "references": self.references,
            "keywords": self.keywords
        }


class AbstractShallowWaterModelTestCase(AbstractTestCase):
    """Base class for shallow water dynamics test cases."""
    pass


class AbstractAdvectionModelTestCase(AbstractTestCase):
    pass



class AbstractAdvectionModel2DTestCase(AbstractAdvectionModelTestCase):
    """Base class for 2D advection dynamics test cases."""
    pass


class AbstractAdvectionModel3DTestCase(AbstractAdvectionModelTestCase):
    """Base class for 3D advection dynamics test cases."""
    pass


class AbstractBaroclinicModelTestCase(AbstractTestCase):
    """Base class for baroclinic dynamics test cases."""
    pass