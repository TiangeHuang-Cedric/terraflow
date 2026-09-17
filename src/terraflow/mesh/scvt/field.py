from __future__ import annotations
from typing import (
    TYPE_CHECKING, 
    overload,
    Any, 
    Union,
    Literal,
    Dict,
    Mapping,
    List, 
    Self,
    Optional,
    get_type_hints,
    get_origin,
    get_args
)
import types
import xarray as xr
import uxarray as uxr
import pint
import numpy as np


from terraflow.util import typing, device
from terraflow.util import equinox as eqx
from terraflow.util.field import (
    AbstractMeta, 
    AbstractField,
    AbstractFields,
    ScalarFieldMeta,
    ScalarField
)

if TYPE_CHECKING:
    import terraflow.util.uxarray as tuxr
    from terraflow.util.backend import AbstractBackend
    from terraflow.mesh.scvt.mesh_type import SCVTMesh


FLT = typing.Float
Array = typing.Array


class SCVTFieldMeta(AbstractMeta):
    name           : str  = eqx.field(static=True, default=""   )
    long_name      : str  = eqx.field(static=True, default=""   )
    units          : str  = eqx.field(static=True, default=""   )
    loc            : str  = eqx.field(static=True, default=""   ) 
    output         : str  = eqx.field(static=True, default=""   )
    id             : int  = eqx.field(static=True, default=0    )
    nele           : int  = eqx.field(static=True, default=0    )
    nlev           : int  = eqx.field(static=True, default=0    )
    ids            : int  = eqx.field(static=True, default=0    )
    ide            : int  = eqx.field(static=True, default=0    )
    ims            : int  = eqx.field(static=True, default=0    )
    ime            : int  = eqx.field(static=True, default=0    )
    kds            : int  = eqx.field(static=True, default=0    )
    kde            : int  = eqx.field(static=True, default=0    )
    kms            : int  = eqx.field(static=True, default=0    )
    kme            : int  = eqx.field(static=True, default=0    )
    at_cell        : bool = eqx.field(static=True, default=False)
    at_vtx         : bool = eqx.field(static=True, default=False)
    at_edge        : bool = eqx.field(static=True, default=False)
    at_sfc         : bool = eqx.field(static=True, default=False)
    full_lev       : bool = eqx.field(static=True, default=True )
    is_static      : bool = eqx.field(static=True, default=False) 
    is_tracer      : bool = eqx.field(static=True, default=False)
    ntracers       : int  = eqx.field(static=True, default=0    )
    is_expand_dim0 : bool = eqx.field(static=True, default=False)
    expand_dim0_len: int  = eqx.field(static=True, default=0    )
    restart        : bool = eqx.field(static=True, default=False)
    initialized    : bool = eqx.field(static=True, default=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            long_name=self.long_name,
            units=self.units,
            loc=self.loc,
            output=self.output,
            id=self.id,
            nele=self.nele,
            nlev=self.nlev,
            ids=self.ids,
            ide=self.ide,
            ims=self.ims,
            ime=self.ime,
            kds=self.kds,
            kde=self.kde,
            kms=self.kms,
            kme=self.kme,
            at_cell=self.at_cell,
            at_vtx=self.at_vtx,
            at_edge=self.at_edge,
            at_sfc=self.at_sfc,
            full_lev=self.full_lev,
            is_static=self.is_static,
            is_tracer=self.is_tracer,
            ntracers=self.ntracers,
            is_expand_dim0=self.is_expand_dim0,
            expand_dim0_len=self.expand_dim0_len,
            restart=self.restart,
            initialized=self.initialized,
            backend=self.backend,
        )

    @property
    def dims(self) -> tuple[str, ...]:
        if self.at_cell:
            dims = ("cell",)
        elif self.at_vtx:
            dims = ("vtx",)
        elif self.at_edge:
            dims = ("edge",)
        else:
            raise ValueError(f"All locations are false in SCVT meta: {self.name}!")
        if self.at_sfc:
            pass
        else:
            if self.full_lev:
                dims = ("lev",) + dims
            else:
                dims = ("ilev",) + dims
        return dims
        
    @property
    def output_dims(self) -> tuple[str, ...]:
        if self.at_cell:
            dims = ("cell",)
        elif self.at_vtx:
            dims = ("vtx",)
        elif self.at_edge:
            dims = ("edge",)
        else:
            raise ValueError(f"All locations are false in SCVT meta: {self.name}!")
        if self.at_sfc:
            pass
        else:
            if self.full_lev:
                dims = ("lev",) + dims
            else:
                dims = ("ilev",) + dims
        if self.is_static:
            return dims
        else:
            return ("time",) + dims
        
    @property
    def output_coordinate(self) -> str:
        if self.at_cell:
            return ("cell_lon cell_lat")
        elif self.at_vtx:
            return ("vtx_lon vtx_lat" )
        elif self.at_edge:
            return ("edge_lon edge_lat")
        else:
            raise ValueError(f"All locations are false in SCVT meta: {self.name}!")


class SCVTField(AbstractField[SCVTFieldMeta]):
    d   : Array         = eqx.field(default=0)
    meta: SCVTFieldMeta = eqx.field(default_factory=lambda: SCVTFieldMeta())
    
    @classmethod
    def _from_mesh(
        cls, 
        name: str,
        long_name: str,
        units: str,
        loc: str,
        mesh: SCVTMesh,
        dev: device.AbstractDevice,
        output: str = "",
        at_sfc: bool = False,
        is_static: bool = False,
        is_tracer: bool = False,
        ntracers: int = 0,
        is_expand_dim0: bool = False,
        expand_dim0_len: int = 0,
        restart: bool = False,
    ) -> Self:
        if loc == "cell" or loc == "lev":
            nele = mesh.cell_ime - mesh.cell_ims + 1
            ids, ide = mesh.cell_ids, mesh.cell_ide
            ims, ime = mesh.cell_ims, mesh.cell_ime
        elif loc == "vtx" or loc == "lev_vtx":
            nele = mesh.vtx_ime  - mesh.vtx_ims  + 1
            ids, ide = mesh.vtx_ids , mesh.vtx_ide
            ims, ime = mesh.vtx_ims , mesh.vtx_ime
        elif loc == "edge" or loc == "lev_edge":
            nele = mesh.edge_ime - mesh.edge_ims + 1
            ids, ide = mesh.edge_ids, mesh.edge_ide
            ims, ime = mesh.edge_ims, mesh.edge_ime
        else:
            raise ValueError(f"Unsupported loc {loc} for field {name}!")
        if loc == "cell" or loc == "vtx" or loc == "edge":
            nlev = mesh.full_kme - mesh.full_kms + 1
            kds, kde = mesh.full_kds, mesh.full_kde
            kms, kme = mesh.full_kms, mesh.full_kme
        elif loc == "lev" or loc == "lev_vtx" or loc == "lev_edge":
            nlev = mesh.half_kme - mesh.half_kms + 1
            kds, kde = mesh.half_kds, mesh.half_kde
            kms, kme = mesh.half_kms, mesh.half_kme
        else:
            raise ValueError(f"Unsupported loc {loc} for field {name}!")
        meta = SCVTFieldMeta(
            name=name,
            long_name=long_name,
            units=units,
            loc=loc,
            output=output,
            id=0,
            nele=int(nele),
            nlev=int(nlev),
            ids=ids,
            ide=ide,
            ims=ims,
            ime=ime,
            kds=kds,
            kde=kde,
            kms=kms,
            kme=kme,
            at_cell=True if loc == "cell" or loc == "lev" else False,
            at_vtx=True if loc == "vtx" or loc == "lev_vtx" else False,
            at_edge=True if loc == "edge" or loc == "lev_edge" else False,
            at_sfc=at_sfc,
            full_lev=True if loc == "cell" or loc == "vtx" or loc == "edge" else False,
            is_static=is_static,
            is_tracer=is_tracer,
            ntracers=ntracers,
            is_expand_dim0=is_expand_dim0,
            expand_dim0_len=expand_dim0_len,
            restart=restart,
            initialized=True,
            backend=dev.backend,
        )
        if meta.at_sfc:
            shape = (nele,)
        else:
            shape = (nlev, nele)
        if meta.is_tracer:
            shape = (ntracers,) + shape
        if meta.is_expand_dim0:
            shape = (expand_dim0_len, ) + shape
        data = dev.backend.zeros(shape, device=dev(), dtype=dev.float_type)
        return cls(d=data, meta=meta)
    
    @staticmethod
    def get_meta_cls() -> type[SCVTFieldMeta]:
        return SCVTFieldMeta
    
    @overload
    def get_index_range(
        self,
        with_lev_idx: Literal[True],
        with_halo: bool = False,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        ... 
        
    @overload
    def get_index_range(
        self,
        with_lev_idx: Literal[False] = False,
        with_halo: bool = False,
    ) -> tuple[int, int]:
        ... 
    
    def get_index_range(
        self,
        with_lev_idx: bool = False,
        with_halo: bool = False,
    ) -> Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
        if with_halo:
            i_s = self.meta.ims; i_e = self.meta.ime
        else:
            i_s = self.meta.ids; i_e = self.meta.ide
        
        if not with_lev_idx:
            return (i_s, i_e)
        else:
            if with_halo:
                k_s = self.meta.kms; k_e = self.meta.kme
            else:
                k_s = self.meta.kds; k_e = self.meta.kde
            return (i_s, i_e), (k_s, k_e)
    
    def copy(
        self, 
        other: Self, 
        with_halo: bool = False,
        at_k: Optional[int] = None,
        at_tracer_idx: Optional[int] = None,
        at_expand_dim0_idx: Optional[int] = None
    ) -> Self:
        # Case 1: self (2d), other (1d)
        if not self.meta.at_sfc and other.meta.at_sfc:
            if at_k is None:
                raise ValueError("Must provide ")
            (i_s, i_e), (k_s, k_e) = self.get_index_range(with_lev_idx=True , with_halo=with_halo)
            self_d  = self .d
            other_d = other.d
            if other.meta.is_expand_dim0:
                other_d = other_d[at_expand_dim0_idx]
            if other.meta.is_tracer:
                other_d = other_d[at_tracer_idx     ] 
            if self.meta.is_expand_dim0 and self.meta.is_tracer:
                if at_k is not None:
                    self_d = self_d.at[at_expand_dim0_idx, at_tracer_idx, k_s:k_e+1]
      
        if self.meta.at_sfc:
            i_s, i_e = self.get_index_range(with_lev_idx=False, with_halo=with_halo)
            if self.meta.is_expand_dim0:
                other = other[at_expand_dim0_idx]
            if self.meta.is_tracer:
                other = other[at_tracer_idx     ]
            return self.meta.backend.tree_at(
                where=lambda f: f.d,
                pytree=self,
                replace=other[i_s:i_e+1]
            )
        else:
            (i_s, i_e), (k_s, k_e) = self.get_index_range(with_lev_idx=True , with_halo=with_halo)
            if self.meta.is_expand_dim0:
                other = other[at_expand_dim0_idx]
            if self.meta.is_tracer:
                other = other[at_tracer_idx     ]
            return self.meta.backend.tree_at(
                where=lambda f: f.d,
                pytree=self,
                replace=other[k_s:k_e+1, i_s:i_e+1]
            )
    
    def get_coords(
        self, 
        mesh: SCVTMesh, 
        layout: str,
        with_halo: bool = False
    ) -> dict[str, Array]:
        return mesh.get_coords(
            loc=self.meta.loc,
            layout=layout,
            with_halo=with_halo
        )
    
    def to_ugrid(
        self,
        mesh: SCVTMesh,
        use_dual: bool = False,
        dim0_idx: Optional[int] = None,
        **kwargs
    ) -> tuxr.UGRIDDataArray:
        """Convert the Field to a UgridDataArray."""
        import terraflow.util.uxarray as tuxr

        ds = field2ugrid(
            field=self,
            use_dual=use_dual,
            dim0_idx=dim0_idx
        )
        ugrid = mesh.to_ugrid(use_dual=use_dual, **kwargs).to_uxgrid()
        return tuxr.UGRIDDataArray.from_xarray(ds, ugrid)

    def to_uxarray(
        self,
        mesh: SCVTMesh,
        use_dual: bool = False,
        dim0_idx: Optional[int] = None,
        **kwargs
    ) -> uxr.UxDataArray:
        """Convert the Field to an Uxarray.DataArray."""
        ds = field2ugrid(
            field=self,
            use_dual=use_dual,
            dim0_idx=dim0_idx
        )
        uxgrid = mesh.to_ugrid(use_dual=use_dual, **kwargs).to_uxgrid()
        return uxr.UxDataArray.from_xarray(ds, uxgrid)
    
    def to_dict(
        self, 
        dim0_idx: Optional[int] = None, 
        tracer_idx: Optional[int] = None
    ) -> dict[str, Array]:
        res: dict[str, Array] = {}
        var = self.d
        if isinstance(self.meta, SCVTFieldMeta):
            if self.meta.is_expand_dim0:
                if dim0_idx is not None:
                    var = var[dim0_idx, ...]
                else:
                    raise ValueError("Must provide dim0_idx!")
            if self.meta.is_tracer:
                if tracer_idx is not None:
                    var = var[tracer_idx, ...]
                else:
                    raise ValueError("Must provide tracer_idx!") 
        res[self.meta.name] = var
        return res
        
        
class SCVTField1D(SCVTField):
    
    @classmethod
    def from_mesh(
        cls, 
        name: str,
        long_name: str,
        units: str,
        loc: str,
        mesh: SCVTMesh,
        dev: device.AbstractDevice,
        output: str = "",
        is_static: bool = False,
        restart: bool = False,
    ) -> Self:
        return super()._from_mesh(
            name=name,
            long_name=long_name,
            units=units,
            loc=loc,
            mesh=mesh,
            dev=dev,
            output=output,
            at_sfc=True,
            is_static=is_static,
            is_tracer=False,
            ntracers=0,
            is_expand_dim0=False,
            expand_dim0_len=0,
            restart=restart
        )
        
        
class SCVTField2D(SCVTField):
    
    @classmethod
    def from_mesh(
        cls, 
        name: str,
        long_name: str,
        units: str,
        loc: str,
        mesh: SCVTMesh,
        dev: device.AbstractDevice,
        at_sfc: bool = False,
        is_expand_dim0: bool = False,
        expand_dim0_len=0,
        output: str = "",
        is_static: bool = False,
        restart: bool = False,
    ) -> Self:
        if at_sfc and not is_expand_dim0:
            raise ValueError(
                "SCVTField2D: if at_sfc is True, is_exapnd_dim0 must be True at the same time!"
            )
        return super()._from_mesh(
            name=name,
            long_name=long_name,
            units=units,
            loc=loc,
            mesh=mesh,
            dev=dev,
            output=output,
            at_sfc=at_sfc,
            is_static=is_static,
            is_tracer=False,
            ntracers=0,
            is_expand_dim0=is_expand_dim0,
            expand_dim0_len=expand_dim0_len,
            restart=restart
        )

        
class SCVTField3D(SCVTField):

    @classmethod
    def from_mesh(
        cls, 
        name: str,
        long_name: str,
        units: str,
        loc: str,
        mesh: SCVTMesh,
        dev: device.AbstractDevice,
        output: str = "",
        is_tracer: Optional[bool] = None,
        ntracers: int = 0,
        is_expand_dim0: Optional[bool] = None,
        expand_dim0_len: int = 0,
        is_static: bool = False,
        restart: bool = False,
    ) -> Self:
        if is_tracer is not None and is_expand_dim0 is not None:
            raise ValueError(
                "is_tracer and is_expand_dim0 should not be provided to SCVTField3D at the same time!"
        )
        is_tracer_opt      = False if is_tracer is None else True
        is_expand_dim0_opt = False if is_expand_dim0 is None else True
        return super()._from_mesh(
            name=name,
            long_name=long_name,
            units=units,
            loc=loc,
            mesh=mesh,
            dev=dev,
            output=output,
            at_sfc=False,
            is_static=is_static,
            is_tracer=is_tracer_opt,
            ntracers=ntracers,
            is_expand_dim0=is_expand_dim0_opt,
            expand_dim0_len=expand_dim0_len,
            restart=restart
        )


class SCVTField4D(SCVTField):

    @classmethod
    def from_mesh(
        cls, 
        name: str,
        long_name: str,
        units: str,
        loc: str,
        mesh: SCVTMesh,
        dev: device.AbstractDevice,
        output: str = "",
        ntracers: int = 0,
        expand_dim0_len: int = 0,
        is_static: bool = False,
        restart: bool = False,
    ) -> Self:
        return super()._from_mesh(
            name=name,
            long_name=long_name,
            units=units,
            loc=loc,
            mesh=mesh,
            dev=dev,
            output=output,
            at_sfc=False,
            is_static=is_static,
            is_tracer=True,
            ntracers=ntracers,
            is_expand_dim0=True,
            expand_dim0_len=expand_dim0_len,
            restart=restart
        )



class SCVTFields(AbstractFields):
    fields : Dict[str, SCVTFieldMeta | ScalarFieldMeta] = eqx.field(static=True, default_factory=dict)
    backend: AbstractBackend = eqx.field(static=True, default=None)

    @classmethod
    def get_all_fields(cls) -> tuple[str, ...]:
        def is_field_type(tp: Any) -> bool:
            origin = get_origin(tp)
            if origin is None:
                if tp is SCVTField:
                    return True
                if isinstance(tp, type) and (issubclass(tp, SCVTField) or issubclass(tp, ScalarField)):
                    return True
                return False
            if origin in (Union, types.UnionType):
                return any(is_field_type(arg) for arg in get_args(tp))
            if origin is Literal:
                return False
            return isinstance(origin, type) and (issubclass(origin, SCVTField) or issubclass(origin, ScalarField))

        annotations: Dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            if base is object:
                continue
            try:
                annotations.update(get_type_hints(base, include_extras=True))
            except Exception:
                annotations.update(getattr(base, "__annotations__", {}))

        return tuple(
            name for name, tp in annotations.items() if is_field_type(tp)
        )

    def __iter__(self):
        for name in self.existing_meta().keys():
            yield getattr(self, name)

    def __len__(self) -> int:
        return len(self.existing_meta().keys())
    
    @classmethod
    def from_fields(cls, fields: Mapping[str, SCVTField | ScalarField], **kwargs) -> Self:
        backend = kwargs.pop("backend", None)
        fields_meta: Dict[str, SCVTFieldMeta | ScalarFieldMeta] = {}
        for name, field in fields.items():
            fields_meta[name] = field.meta
            if backend is None:
                backend = field.meta.backend
        kwargs.update(fields)
        return cls(fields=fields_meta, backend=backend, **kwargs)

    def _wrap(self, **kwargs: SCVTField | ScalarField) -> Self:
        names = self.existing_meta().keys()
        new_vals = {n: kwargs.get(n, getattr(self, n)) for n in names}
        if self.backend is None:
            raise ValueError("SCVTFields.backend is required for wrapping field values.")
        for k, v in new_vals.items():
            if not isinstance(v, (SCVTField, ScalarField)):
                raise TypeError(f"Variable '{k}' is not a Field.")
            self = self.backend.tree_at(
                where=lambda vs: object.__getattribute__(vs, k),
                pytree=self,
                replace=v
            )
        return self

    def keys(self) -> List[str]:
        return list(self.existing_meta().keys())

    def values(self) -> List[SCVTField | ScalarField]:
        return [getattr(self, name) for name in self.existing_meta().keys()]

    def items(self) -> List[tuple[str, SCVTField | ScalarField]]:
        return [(name, getattr(self, name)) for name in self.existing_meta().keys()]

    def isel(self, index: int, axis: int) -> Self:
        """Index select along a specific dimension for all variables."""
        new_vars: Dict[str, SCVTField | ScalarField] = {}
        for key, var in self.items():
            new_vars[key] = var.isel(index=index, axis=axis)
        return self._wrap(**new_vars)

    def copy(self) -> Self:
        return self._wrap()

    def existing_values(self) -> Dict[str, SCVTField | ScalarField]:
        """Return existing variables."""
        return dict(self.items())

    def existing_meta(self) -> Dict[str, SCVTFieldMeta | ScalarFieldMeta]:
        """Return metadata of existing variables."""
        return self.fields

    def varnames(self) -> List[str]:
        return list(self.keys())

    def unwrap(self) -> Dict[str, Array]:
        vars: Dict[str, Array] = {}
        for key, var in self.items():
            vars[key] = var.unwrap()
        return vars

    def zeros_like(self) -> Self:
        new_variables: Dict[str, SCVTField | ScalarField] = {}
        for key, var in self.items():
            new_variables[key] = var.zeros_like()
        return self._wrap(**new_variables)

    def to(self, unit: str) -> Self:
        """Convert all parameters to the specified unit and return them."""
        new_variables: Dict[str, SCVTField | ScalarField] = {}
        for key, var in self.items():
            new_variables[key] = var.to(unit)
        return self._wrap(**new_variables)

    def to_base_units(self) -> Self:
        """Convert all parameters to base units and return them."""
        new_variables: Dict[str, SCVTField | ScalarField] = {}
        for key, var in self.items():
            new_variables[key] = var.to_base_units()
        return self._wrap(**new_variables)

    def to_reduced_units(self) -> Self:
        """Convert all parameters to reduced units and return them."""
        new_variables: Dict[str, SCVTField | ScalarField] = {}
        for key, var in self.items():
            new_variables[key] = var.to_reduced_units()
        return self._wrap(**new_variables)

    def uconvert(self, usys: str = "SI") -> Self:
        """Compatibility wrapper for previous unit-system conversion calls."""
        if not (isinstance(usys, str) and usys.lower() == "si"):
            raise ValueError(
                "SCVTFields.uconvert only supports 'SI' after migrating to pint; "
                "use to(unit), to_base_units(), or to_reduced_units()."
            )
        return self.to_base_units()

    def to_si(self) -> Self:
        """Convert all parameters to SI base units and return them."""
        return self.to_base_units()
    
    def to_netcdf(
        self,
        mesh: SCVTMesh,
        path: str,
        format="UGRID",
        use_dual: bool = False,
        expand_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Save variable set to a NetCDF file."""
        if format == "UGRID":
            ds = fields2ugrid(
                fields=self,
                use_dual=use_dual,
                dim0_idx=expand_idx
            )
            grid_ds = mesh.to_ugrid(
                use_dual=use_dual,
                **kwargs
            ).to_xarray()
            ds = xr.merge([ds, grid_ds])
            ds.to_netcdf(path)

    def to_ugrid(
        self,
        mesh: SCVTMesh,
        use_dual: bool = False,
        dim0_idx: Optional[int] = None,
        **kwargs
    ) -> tuxr.UGRIDDataset:
        """Convert variable set to a UXarray.UxGrid."""
        import terraflow.util.uxarray as tuxr

        ds = fields2ugrid(
            fields=self,
            use_dual=use_dual,
            dim0_idx=dim0_idx
        )
        uxgrid = mesh.to_ugrid(
            use_dual=use_dual,
            **kwargs
        ).to_uxgrid()
        return tuxr.UGRIDDataset.from_xarray(ds, uxgrid)

    def to_uxarray(
        self,
        mesh: SCVTMesh,
        use_dual: bool = False,
        dim0_idx: Optional[int] = None,
        **kwargs
    ) -> uxr.UxDataset:
        """Convert variable set to a UXarray.Dataset."""
        ds = fields2ugrid(
            fields=self,
            use_dual=use_dual,
            dim0_idx=dim0_idx
        )
        uxgrid = mesh.to_ugrid(
            use_dual=use_dual,
            **kwargs
        ).to_uxgrid()
        return uxr.UxDataset.from_xarray(ds, uxgrid)
    
    def to_dict(
        self, 
        dim0_idx: Optional[int] = None, 
        tracer_idx: Optional[int] = None
    ) -> dict[str, Array]:
        res: dict[str, Array] = {}
        for field in self.values():
            if isinstance(field, SCVTField):
                res.update(
                    field.to_dict(
                        dim0_idx=dim0_idx,
                        tracer_idx=tracer_idx
                    )
                )
            elif isinstance(field, ScalarField):
                res.update(field.to_dict(dim0_idx=dim0_idx))                
        return res

    
SCVTFieldLike = SCVTField1D | SCVTField2D | SCVTField3D | SCVTField4D


def field2ugrid(
    field: SCVTField, 
    use_dual: bool = False,
    dim0_idx: Optional[int] = None
) -> xr.DataArray:
    """Convert the Field to an xarray.Dataset with UGRID format."""
    if field.meta.is_empty():
        raise ValueError(
            "Cannot convert to xarray when metadata is empty."
        ) 
    return meta2ugrid(
        value=field.d, 
        meta=field.meta, 
        use_dual=use_dual, 
        dim0_idx=dim0_idx
    )
    
    
def fields2ugrid(
    fields: SCVTFields,
    use_dual: bool = False,
    dim0_idx: Optional[int] = None
) -> xr.Dataset:
    """Get the meta needed for UGRID NetCDF Format"""
    var_ds: Dict[str, xr.DataArray] = {}
    for var in fields.values():
        if isinstance(var.meta, SCVTFieldMeta):
            var_ds[var.meta.name] = meta2ugrid(
                value=var.d,
                meta=var.meta,
                use_dual=use_dual,
                dim0_idx=dim0_idx
            )
        elif isinstance(var.meta, ScalarFieldMeta):
            continue
        else:
            raise TypeError(
                f"Variable '{var.meta.name}' metadata is not of type StaggeredFieldMetaData, "
                "which is required for UGRID format conversion."
            )
    return xr.Dataset(data_vars=var_ds)


def meta2ugrid(
    value: Array | pint.Quantity,
    meta: SCVTFieldMeta,
    use_dual: bool = False,
    dim0_idx: Optional[int] = None
):
    """
    Format a TerraFlow variable to UGRID xarray DataArray.

    Args:
        value (Array | pint.Quantity): The variable value.
            Its dimension must follow the order (time, vertical, horizontal).
        meta (Dict[str, Any]): The variable metadata.
        use_dual (bool): Whether to use dual mesh or primal mesh.

    Returns:
        xr.DataArray: The formatted UGRID xarray DataArray.
    """
    if isinstance(value, pint.Quantity):
        var = np.array(value.magnitude)
        units = str(value.units)
    elif meta.backend.is_array(value):
        var = np.array(value)
        units = meta.units
    else:
        raise TypeError(f"Variable has unsupported type.")

    if meta.is_expand_dim0 and dim0_idx is not None:
        var: np.ndarray = var[dim0_idx, ...]

    # Convert location
    location = meta.loc
    if location not in ["cell", "edge", "vtx", "lev", "lev_vtx", "lev_edge"]:
        raise ValueError(
            f"hori_location must be 'cell', 'edge' or 'vtx', got {location}."
        )    
    if not use_dual:
        mesh_name = "primal"
        if location == "cell" or location == "lev":
            location = "face"
        elif location == "vtx" or location == "lev_vtx":
            location = "node"
        else:
            location = location
    else:
        mesh_name = "dual"
        if location == "cell" or location == "lev":
            location = "node"
        elif location == "vtx" or location == "lev_vtx":
            location = "face"
        else:
            location = location

    # Determine dims and coordinates
    dims = meta.dims[:-1] + (f"n_{location}",)

    if meta.is_tracer:
        dims = ("ntracers",) + dims

    coordinates = f"{location}_lon {location}_lat"

    return xr.DataArray(
        data=var,
        dims=dims,
        attrs={
            "name": meta.name,
            "mesh": mesh_name,
            "location": location,
            "coordinates": coordinates,
            "units": units,
            "long_name": meta.long_name,
            "is_static": meta.is_static,
            "is_tracer": meta.is_tracer,
            "output": meta.output,
        }
    )
