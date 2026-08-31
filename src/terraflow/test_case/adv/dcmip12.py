from terraflow.util.planet import PlanetParameterSet
from terraflow.test_case.abstract import AbstractAdvectionModel3DTestCase


class DCMIP12TestCase(AbstractAdvectionModel3DTestCase):
    """DCMIP 1-2 3D advection test case."""
    T0  : float 
    p0  : float 
    K0  : float 
    u0  : float 
    w0  : float 
    z1  : float 
    z2  : float 
    z0  : float 
    ztop: float 
    tau : float 

    def __init__(self, para: PlanetParameterSet):
        self.alias_name = "adv_dcmip12"
        self.name = "DCMIP 1-2 3D Advection"
        self.description = "DCMIP 1-2 3D advection test case."
        self.references = (
            "Ullrich, P. A. et al. Dynamical Core Model Intercomparison Project (DCMIP) Test Case Document.",
        )
        self.keywords = ("advection", "3D", "DCMIP", "atmosphere")
        self.T0   = 300.0
        self.p0   = 1.0e5
        self.K0   = 5
        self.u0   = 40.0
        self.w0   = 0.15
        self.z1   = 2000.0
        self.z2   = 5000.0
        self.z0   = 0.5 * (self.z1 + self.z2)
        self.ztop = 12000.0
        self.tau  = 86400.0
        self.rho0 = self.p0 / (para.rd.unwrap() * self.T0)
        super().__init__(para)