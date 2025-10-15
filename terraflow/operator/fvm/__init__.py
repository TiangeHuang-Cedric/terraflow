from terraflow.operator import Operator

class FiniteVolumeStaggeredCMethod(Operator):
    pass


class Divergence(FiniteVolumeStaggeredCMethod):
    pass


class Gradient(FiniteVolumeStaggeredCMethod):
    pass


class Curl(FiniteVolumeStaggeredCMethod):
    pass


class FluxReconstruction(FiniteVolumeStaggeredCMethod):
    pass


class Interpolation(FiniteVolumeStaggeredCMethod):
    pass