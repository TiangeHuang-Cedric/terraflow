from typing import Callable

import terraflow.equation.atmosphere.shallow_water_equation.base as base
import terraflow.equation.atmosphere.shallow_water_equation.vector_invariant_form as form


class VectorInvariantForm(base.ShallowWaterEquForm):
    pass


class NonlinearCoriolisTerm:
    pass


class GradientKineticEnergyTerm:
    pass


class GradientGeopotentialTerm:
    pass


class DivergenceThickness:
    pass


class ContinuityEqu(VectorInvariantForm):
    tendency: Callable
    div_thickness: DivergenceThickness


class MomentumEqu(VectorInvariantForm):
    tendency: Callable
    nonlinear_coriolis_term: NonlinearCoriolisTerm
    grad_kinetic_energy_term: GradientKineticEnergyTerm
    grad_geopotential_term: GradientGeopotentialTerm
    pass
