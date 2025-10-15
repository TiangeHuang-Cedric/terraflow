class VectorInvariantForm(EquationForm):
    pass


class ContinuityEqu(VectorInvariantForm):
    dhdt
    thickness_divergence


class MomentumEqu(VectorInvariantForm):
    dudt
    nonlinear_coriolis
    
    pass