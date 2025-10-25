from terraflow.equation.base import EquationSet, EquationForm


class ShallowWaterEquSet(EquationSet):
    continuity_equ: EquationForm
    momentum_equ: EquationForm


class ShallowWaterEquForm(EquationForm):
    pass


class ContinuityEquForm(EquationForm):
    pass


class MomentumEquForm(EquationForm):
    pass
