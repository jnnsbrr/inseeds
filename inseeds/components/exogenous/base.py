"""Base class for exogenous data sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import xarray as xr


class ExogenousSource(ABC):
    """Base class for exogenous data sources.
    
    Subclasses define loading, caching, and slicing for external datasets.
    """
    
    granularity: Literal["cell", "country"] = "country"
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for accessor (e.g., 'residue', 'prices')."""
        ...
    
    @abstractmethod
    def ensure(self, sim_path: str | Path, **kwargs) -> Path:
        """Ensure data is available. Returns path to cached file."""
        ...
    
    def slice_for_country(self, ds: xr.Dataset, country_code: str) -> xr.Dataset:
        """Slice for a country. Override for custom behavior."""
        if "area_code" in ds.dims and country_code in ds.area_code.values:
            return ds.sel(area_code=country_code)
        return ds
    
    def slice_for_cell(self, ds: xr.Dataset, cell_idx: int, country_code: str | None = None) -> xr.Dataset:
        """Slice for a cell. Uses granularity to determine behavior."""
        if self.granularity == "cell" and "cell" in ds.dims:
            return ds.isel(cell=cell_idx)
        if country_code:
            return self.slice_for_country(ds, country_code)
        return ds
