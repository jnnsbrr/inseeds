"""Exogenous data accessor with auto-slicing by entity level.

>>> world.exogenous.prices      # Full dataset
>>> country.exogenous.prices    # Sliced for country
>>> cell.exogenous.residue      # Sliced for cell
"""

from typing import Any
import xarray as xr


class Exogenous:
    """Accessor for exogenous datasets with lazy slicing."""
    
    def __init__(self, entity: Any, datasets: dict = None, sources: dict = None, world_ref: "Exogenous" = None):
        self._entity = entity
        self._datasets = datasets or {}
        self._sources = sources or {}
        self._world_ref = world_ref
        self._cache = {}
    
    @property
    def _data(self):
        return self._world_ref._datasets if self._world_ref else self._datasets
    
    @property
    def _srcs(self):
        return self._world_ref._sources if self._world_ref else self._sources
    
    def keys(self):
        return list(self._data.keys())
    
    def items(self):
        """Iterate over (name, dataset) pairs."""
        for name in self.keys():
            yield name, getattr(self, name)
    
    def __getattr__(self, name: str) -> xr.Dataset:
        if name.startswith("_"):
            raise AttributeError(name)
        
        if name not in self._data:
            raise AttributeError(f"No dataset '{name}'. Available: {list(self._data.keys())}")
        
        if name in self._cache:
            return self._cache[name]
        
        ds = self._data[name]
        src = self._srcs.get(name)
        
        # World level - no slicing
        if not self._world_ref:
            return ds
        
        # Country level
        if hasattr(self._entity, "country_code") and not hasattr(self._entity, "grid"):
            result = src.slice_for_country(ds, self._entity.country_code) if src else ds
        
        # Cell level
        elif hasattr(self._entity, "grid"):
            # Prefer local_index (position within country) for country-subsetted data
            cell_idx = getattr(self._entity, "local_index", None) or getattr(self._entity, "_local_index", None)
            if cell_idx is None:
                cell_idx = self._entity.grid.cell.item()  # Fallback to global index
            country = getattr(self._entity, "country_code", None) or getattr(getattr(self._entity, "country", None), "country_code", None)
            result = src.slice_for_cell(ds, cell_idx, country) if src else ds
        
        else:
            result = ds
        
        self._cache[name] = result
        return result
    
    @classmethod
    def for_entity(cls, entity: Any, world_exo: "Exogenous") -> "Exogenous":
        """Create accessor for a country/cell that references world datasets."""
        return cls(entity, world_ref=world_exo)
