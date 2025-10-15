# TerraFlow

TerraFlow is a Python toolkit for experimenting with finite-volume discretisations of atmosphere and ocean equations on orthogonal dual meshes. The code base focuses on building geometric operators on GRiST meshes and wiring them into reusable, JAX-compatible components.

## Implemented Features
- Mesh ingestion from GRiST outputs via `OrthogonalDualMesh.from_grist`, assembling paired primal and dual grids with auxiliary edge metadata.
- Spherical-geometry helpers (`terraflow.util.math`) that compute kite areas, spherical triangle areas, rotated neighbour lists, and corrected edge normals.
- Finite-volume staggered C-grid operators (`DivergenceAtPrimeCell`, `GradientAtEdge`, `CurlAtDualCell`) that act on edge-centred or cell-centred fields.
- Flux reconstruction utilities on edges, currently including the second-order central-difference scheme.
- Interpolation operators for primal/dual transfers and TRiSK-style edge weighting, including automatic tabulation of TRiSK weights.
- Device and I/O utilities that configure JAX backends (`CPU`, `GPU`, `TPU`) and load/save NetCDF datasets used by the mesh layer.

## Project Layout
```
terraflow/
  mesh/orthogonal_dual_mesh/          # GRiST ingestion and spherical geometry helpers
  operator/fvm/orthogonal_dual_mesh/  # Divergence/gradient/curl, interpolation, flux reconstruction
  util/                               # JAX device selection, IO, math utilities, typing aliases
examples/notebook/read_from_GRIST.ipynb  # Interactive walkthrough for loading a mesh
data/GRIST_INPUT/                     # Sample GRiST reference data (NetCDF)
```

## Installation
TerraFlow targets Python 3.11+ and depends on JAX, xarray, and related scientific libraries.

```bash
# Clone this repository, then inside the project root:
pip install -e .
# or use Poetry
poetry install
```

The repository ships with GRiST reference data under `data/GRIST_INPUT/`. Replace any paths in the examples below with the dataset you want to work with.

## Quick Start
```python
import jax.numpy as jnp

from terraflow.mesh.orthogonal_dual_mesh import OrthogonalDualMesh
from terraflow.operator.fvm.orthogonal_dual_mesh.base import (
    DivergenceAtPrimeCell,
    GradientAtEdge,
)
from terraflow.operator.fvm.orthogonal_dual_mesh.interp import InterpTriSKWeightedAdd
from terraflow.util.device import CPU

# Build a mesh from GRiST NetCDF files (replace with actual file names)
mesh = OrthogonalDualMesh.from_grist(
    "data/GRIST_INPUT/QU-G6/mesh_qug6_0d.nc",
    "data/GRIST_INPUT/QU-G6/mesh_qug6_2d.nc",
    dev=CPU(),
)

# Divergence of an edge-centred field on the primal cells
flux = jnp.zeros(mesh.shape["n_edge"])
divergence = DivergenceAtPrimeCell()
div_at_cell = divergence(flux, radius=6.371e6, plg=mesh.plg, edp=mesh.edp)

# Gradient of a scalar on primal cells reconstructed to edges
scalar = jnp.zeros(mesh.shape["n_polg"])
gradient = GradientAtEdge()
edge_grad = gradient(scalar, radius=6.371e6, edd=mesh.edd, edpNr_edtTg=mesh.aux.edpnr_eddtg)

# TRiSK-weighted interpolation between neighbouring edges
interp = InterpTriSKWeightedAdd(mesh)
edge_field = interp(flux, mesh.edd)
```

## Data Notes
- The GRiST datasets are not bundled with the Python package and should remain under `data/GRIST_INPUT/`.
- The loader expects zero-based indexing; the conversion is handled internally when reading NetCDF variables.

## Development
- Run checks with `pytest` to execute the available unit tests (the suite is still minimal).
- Rich logging and debugging printouts can be enabled by wrapping JAX operations with `jax.debug.print` inside the geometry routines.

## Roadmap Highlights
- Extend flux reconstruction with the planned least-square high-order schemes.
- Flesh out the atmospheric shallow-water equation set and diagnostic tendencies.
- Add regression tests that cover the interpolation operators against reference solutions.

Contributions that improve documentation, add tests around the implemented operators, or port additional equation sets are very welcome.
