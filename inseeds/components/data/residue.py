"""MADRaT crop residue usage data handler.

Provides spatially-explicit fractions of residue burnt, removed, and
recycled at 0.5° resolution from Smerald et al. (2023) data.

Data structure (from Jens):
    production = recycled + removed + burnt
    
    - burnt: residues burned (no economic value)
    - removed: animal feed + other purposes (has opportunity cost)
    - recycled: bedding that returns to field with manure (left on field)

The data is CFT-specific (16 crop functional types). At runtime, fractions
are weighted by the actual crop composition (cftfrac) from LPJmL.
"""

from pathlib import Path
import numpy as np
import xarray as xr

from copan_eval.common import CopanData, GridDimensionType


class ResidueData(CopanData[GridDimensionType]):
    """Handler for MADRaT crop residue fraction data.
    
    Follows the ensure/cache pattern used by FAO data handlers.
    Uses copan-eval CopanData base class for consistency.
    
    The cached data retains CFT-specific fractions (dims: cell, cft).
    At runtime, use `weighted_fractions()` to compute cell-specific
    values weighted by actual crop composition.
    """
    
    # Global source paths (cluster)
    DEFAULT_DATA_PATH = Path("/p/projects/copan/data/inseeds/input")
    BURNT_FILE = "madrat_residues_burnt_1965-2015_16bands.nc"
    PRODUCTION_FILE = "madrat_residues_production_1965-2015_16bands.nc"
    REMOVED_FILE = "madrat_residues_removed_1965-2015_16bands.nc"
    RECYCLED_FILE = "madrat_residues_recycled_1965-2015_16bands.nc"
    
    # Number of CFTs in MADRaT data
    N_CFTS = 16
    
    # Cache filename
    CACHE_FILE = "residue_fractions.nc"
    
    @classmethod
    def ensure(
        cls,
        sim_path: str | Path,
        grid: xr.Dataset,
        reference_year: int = 2015,
        overwrite: bool = False,
    ) -> Path:
        """Ensure residue fractions are available for the simulation grid.
        
        Parameters
        ----------
        sim_path : str | Path
            Simulation path (cache saved to {sim_path}/input/).
        grid : xr.Dataset
            LPJmL grid (world.grid) with lat/lon coordinates.
        reference_year : int
            Year to extract (default 2015, latest available).
        overwrite : bool
            Force regeneration of cache.
            
        Returns
        -------
        Path
            Path to cached NetCDF file.
        """
        cache_path = Path(sim_path) / "input" / cls.CACHE_FILE
        
        if cache_path.exists() and not overwrite:
            return cache_path
        
        print(f"Extracting residue fractions for {len(grid.cell)} cells...")
        fractions = cls._extract_fractions(grid, reference_year)
        fractions.to_netcdf(cache_path)
        return cache_path
    
    @classmethod
    def _extract_fractions(
        cls,
        grid: xr.Dataset,
        year: int,
        data_path: Path | None = None,
    ) -> xr.Dataset:
        """Extract CFT-specific fractions for grid cells.
        
        Parameters
        ----------
        grid : xr.Dataset
            LPJmL grid with lat/lon coordinates.
        year : int
            Reference year to extract.
        data_path : Path, optional
            Path to MADRaT files (uses DEFAULT_DATA_PATH if None).
            
        Returns
        -------
        xr.Dataset
            Dataset with frac_burnt, frac_removed, frac_recycled
            variables indexed by (cell, cft). CFT dimension retained
            for runtime weighting by actual crop composition.
        """
        if data_path is None:
            data_path = cls.DEFAULT_DATA_PATH
        
        # Load global data
        burnt = xr.open_dataset(data_path / cls.BURNT_FILE)
        production = xr.open_dataset(data_path / cls.PRODUCTION_FILE)
        removed = xr.open_dataset(data_path / cls.REMOVED_FILE)
        recycled = xr.open_dataset(data_path / cls.RECYCLED_FILE)

        # Select year (or nearest available)
        burnt = burnt.sel(time=year, method="nearest")
        production = production.sel(time=year, method="nearest")
        removed = removed.sel(time=year, method="nearest")
        recycled = recycled.sel(time=year, method="nearest")

        # Get variable names
        burnt_var = list(burnt.data_vars)[0]
        prod_var = list(production.data_vars)[0]
        removed_var = list(removed.data_vars)[0]
        recycled_var = list(recycled.data_vars)[0]

        # Get grid coordinates
        lats = grid.lat.values
        lons = grid.lon.values

        # Extract cells using nearest neighbor (keep CFT dimension)
        burnt_cells = burnt[burnt_var].sel(
            latitude=xr.DataArray(lats, dims="cell"),
            longitude=xr.DataArray(lons, dims="cell"),
            method="nearest",
        )
        prod_cells = production[prod_var].sel(
            latitude=xr.DataArray(lats, dims="cell"),
            longitude=xr.DataArray(lons, dims="cell"),
            method="nearest",
        )
        removed_cells = removed[removed_var].sel(
            latitude=xr.DataArray(lats, dims="cell"),
            longitude=xr.DataArray(lons, dims="cell"),
            method="nearest",
        )
        recycled_cells = recycled[recycled_var].sel(
            latitude=xr.DataArray(lats, dims="cell"),
            longitude=xr.DataArray(lons, dims="cell"),
            method="nearest",
        )
        
        # Compute fractions per CFT (handle division by zero)
        prod_safe = xr.where(prod_cells > 0, prod_cells, np.nan)
        frac_burnt = burnt_cells / prod_safe
        frac_removed = removed_cells / prod_safe
        frac_recycled = recycled_cells / prod_safe
        
        # Fill NaN with 0 for CFTs without production
        frac_burnt = frac_burnt.fillna(0)
        frac_removed = frac_removed.fillna(0)
        frac_recycled = frac_recycled.fillna(0)
        
        return xr.Dataset({
            "frac_burnt": frac_burnt,
            "frac_removed": frac_removed,
            "frac_recycled": frac_recycled,
        })
    
    @staticmethod
    def weighted_fractions(
        residue_ds: xr.Dataset,
        cell_idx: int,
        cftfrac: np.ndarray,
    ) -> dict:
        """Compute weighted residue fractions for a cell.
        
        Weights CFT-specific fractions by actual crop composition.
        
        Parameters
        ----------
        residue_ds : xr.Dataset
            Cached residue fractions with (cell, cft) dimensions.
        cell_idx : int
            Cell index in the dataset.
        cftfrac : np.ndarray
            Crop fractions from LPJmL (rainfed + irrigated summed by crop type,
            excluding non-crops like grassland/biomass).
            
        Returns
        -------
        dict
            Dict with 'burnt', 'removed', 'recycled' weighted fractions.
            Fractions sum to 1 (or close to 1).
        """
        # Get CFT-specific fractions for this cell
        burnt_cft = residue_ds.frac_burnt.isel(cell=cell_idx).values
        removed_cft = residue_ds.frac_removed.isel(cell=cell_idx).values
        recycled_cft = residue_ds.frac_recycled.isel(cell=cell_idx).values
        
        # Use only the crop types present in cftfrac (already filtered upstream)
        n_crops = len(cftfrac)
        burnt_cft = burnt_cft[:n_crops]
        removed_cft = removed_cft[:n_crops]
        recycled_cft = recycled_cft[:n_crops]
        
        weights = np.asarray(cftfrac, dtype=np.float64)
        
        # Identify CFTs with valid MADRaT data (fractions sum > 0)
        # CFTs without production in MADRaT have all zeros
        cft_has_data = (burnt_cft + removed_cft + recycled_cft) > 0
        
        # Zero out weights for CFTs without MADRaT data
        weights = weights * cft_has_data
        
        # Normalize weights (sum to 1, handle zero case)
        weight_sum = np.nansum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback: use equal weights across CFTs with data
            n_valid = np.sum(cft_has_data)
            if n_valid > 0:
                weights = cft_has_data.astype(float) / n_valid
            else:
                # No valid data at all - return zeros
                return {'burnt': 0.0, 'removed': 0.0, 'recycled': 1.0}
        
        # Compute weighted averages
        frac_burnt = float(np.sum(burnt_cft * weights))
        frac_removed = float(np.sum(removed_cft * weights))
        frac_recycled = float(np.sum(recycled_cft * weights))
        
        return {
            'burnt': frac_burnt,
            'removed': frac_removed,
            'recycled': frac_recycled,
        }
