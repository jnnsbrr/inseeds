"""Base class for FAO dataset handlers.

This module provides:

1. **FaoDataset**: Abstract base class for downloading and caching FAOSTAT data
2. **get_value_with_fallback**: Tiered lookup for missing country data

Fallback Tiers
--------------
When country data is missing, the fallback mechanism tries:

1. **Country**: Same country with expanding time window (up to 20 years back)
2. **Neighbours**: Mean from neighbouring countries  
3. **Global**: Mean across all available countries

This ensures complete data coverage even when FAOSTAT has gaps.
"""

from abc import abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from copan_eval.fao import FaoData, fao_definitions, FaoApiAdapter, FaoCropTranslator
from ..base import ExogenousSource

logger = logging.getLogger(__name__)


# =============================================================================
# Fallback Mechanism
# =============================================================================

@dataclass
class FallbackResult:
    """Result from tiered fallback lookup.
    
    Attributes
    ----------
    value : float
        The retrieved value.
    tier : str
        Which tier provided the value: "country", "neighbours", or "global".
    detail : str
        Context about the lookup (e.g., time window used, neighbours consulted).
    """
    value: float
    tier: Literal["country", "neighbours", "global"]
    detail: str


def get_value_with_fallback(
    data: xr.DataArray,
    country_code: str,
    year: int | None = None,
    neighbour_codes: list[str] | None = None,
    avg_years: int = 5,
    max_lookback: int = 20,
    aggregator: Literal["mean", "sum", "last"] = "mean",
) -> FallbackResult:
    """Get value for a country with tiered fallback for missing data.
    
    Parameters
    ----------
    data : xr.DataArray
        Data with "area_code" dimension and optionally "time".
    country_code : str
        ISO3 country code (e.g., "NLD").
    year : int, optional
        Target year. If None, uses most recent available.
    neighbour_codes : list[str], optional
        ISO3 codes of neighbouring countries for Tier B fallback.
    avg_years : int
        Initial time window size for averaging (default: 5 years).
    max_lookback : int
        Maximum years to search backward (default: 20 years).
    aggregator : str
        How to aggregate over time: "mean", "sum", or "last".
        
    Returns
    -------
    FallbackResult
        Value with tier and detail information.
        
    Raises
    ------
    ValueError
        If no valid data found at any tier.
        
    Examples
    --------
    >>> result = get_value_with_fallback(
    ...     ds["depreciation_rate"],
    ...     country_code="NLD",
    ...     year=2020,
    ...     neighbour_codes=["DEU", "BEL"],
    ... )
    >>> print(f"{result.value:.3f} (tier: {result.tier})")
    """
    has_time = "time" in data.dims
    area_codes = list(data.area_code.values) if "area_code" in data.dims else []
    
    # Determine target year
    if has_time:
        available_years = sorted(int(y) for y in data.time.values)
        year = int(year) if year else max(available_years)
    else:
        available_years = []
    
    def aggregate(arr: xr.DataArray, years: list[int]) -> float | None:
        """Aggregate values over time, returning None if all NaN."""
        if not has_time:
            val = float(arr.values)
            return None if np.isnan(val) else val
        
        valid_years = [y for y in years if y in arr.time.values]
        if not valid_years:
            return None
            
        subset = arr.sel(time=valid_years)
        if aggregator == "last":
            val = float(subset.isel(time=-1).values)
        elif aggregator == "sum":
            val = float(subset.sum(dim="time").values)
        else:
            val = float(subset.mean(dim="time").values)
        
        return None if np.isnan(val) else val
    
    def get_window(target: int, size: int) -> list[int]:
        """Get years in window: (target - size, target]."""
        return [y for y in available_years if target - size < y <= target]
    
    def try_country(code: str) -> tuple[float | None, str]:
        """Try to get value for a single country with expanding window."""
        if code not in area_codes:
            return None, ""
        
        country_data = data.sel(area_code=code)
        
        # Try progressively larger windows
        for window_size in range(avg_years, max_lookback + 1):
            years = get_window(year, window_size)
            value = aggregate(country_data, years)
            if value is not None:
                window_str = f"{min(years)}-{max(years)}" if years else str(year)
                return value, window_str
        
        # Last resort: all available years
        if has_time and available_years:
            value = aggregate(country_data, available_years)
            if value is not None:
                return value, f"{min(available_years)}-{max(available_years)}"
        
        return None, ""
    
    # Tier A: Country
    value, window = try_country(country_code)
    if value is not None:
        return FallbackResult(value, "country", f"window={window}")
    
    # Tier B: Neighbours
    if neighbour_codes:
        neighbour_values = []
        used = []
        for nc in neighbour_codes:
            val, _ = try_country(nc)
            if val is not None:
                neighbour_values.append(val)
                used.append(nc)
        
        if neighbour_values:
            mean_val = float(np.mean(neighbour_values))
            detail = f"neighbours={','.join(used)}"
            logger.info(f"{country_code}: using neighbours tier ({detail})")
            return FallbackResult(mean_val, "neighbours", detail)
    
    # Tier C: Global mean
    if area_codes:
        window_years = get_window(year, avg_years) if has_time else []
        
        if has_time and window_years:
            valid_years = [y for y in window_years if y in data.time.values]
            subset = data.sel(time=valid_years) if valid_years else data
            global_data = subset.mean(dim="time") if aggregator == "mean" else subset.isel(time=-1)
        else:
            global_data = data
        
        global_mean = float(global_data.mean(dim="area_code").values)
        
        if not np.isnan(global_mean):
            n_countries = int((~np.isnan(global_data.values)).sum())
            logger.info(f"{country_code}: using global tier (n_countries={n_countries})")
            return FallbackResult(global_mean, "global", f"n_countries={n_countries}")
    
    raise ValueError(f"No valid data for {country_code} at any tier")


# =============================================================================
# Utility Functions
# =============================================================================

def check_fao_api_available(timeout: float = 5.0) -> bool:
    """Check if FAO API is accessible with valid authentication."""
    try:
        adapter = FaoApiAdapter()
        return adapter.ping()
    except Exception:
        return False


def get_fao_country_code(iso3_code: str) -> str | None:
    """Convert ISO3 country code to FAO country code."""
    iso3_to_fao = fao_definitions.get_area_code_dict(
        code_standard_out="FAO", code_standard_in="ISO3"
    )
    return iso3_to_fao.get(iso3_code)


# =============================================================================
# FaoDataset Base Class
# =============================================================================

class FaoDataset(ExogenousSource):
    """Abstract base class for FAO dataset handlers.
    
    Handles downloading, transforming, and caching FAOSTAT data.
    Subclasses define domain-specific behavior.
    
    Inherits from ExogenousSource for unified exogenous data access.
    FAO data is country-level granularity.
    
    Subclass Requirements
    ---------------------
    Must implement:
    - domain: FAOSTAT domain code (e.g., "PP", "CS")
    - elements: Element codes to download
    - name: Human-readable name (also used as accessor key)
    - output_filename: Output NetCDF filename
    - _get_items(): Item codes to download
    - _post_process(ds): Domain-specific transformations
    
    Optional overrides:
    - _translate_to_lpjml(): Whether to map crops to LPJmL names (default: True)
    """
    
    granularity: Literal["cell", "country"] = "country"

    @property
    @abstractmethod
    def domain(self) -> str:
        """FAOSTAT domain code (e.g., 'PP', 'CS', 'QCL')."""
        ...

    @property
    @abstractmethod
    def elements(self) -> list[str]:
        """FAOSTAT element codes to download."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""
        ...

    @property
    @abstractmethod
    def output_filename(self) -> str:
        """Output NetCDF filename."""
        ...

    @abstractmethod
    def _get_items(self) -> pd.Series:
        """Return item codes to download."""
        ...

    @abstractmethod
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply domain-specific post-processing."""
        ...

    def _translate_to_lpjml(self) -> bool:
        """Whether to translate crop codes to LPJmL CFT names."""
        return True

    # -------------------------------------------------------------------------
    # Path Management
    # -------------------------------------------------------------------------
    
    def get_path(self, sim_path: str | Path) -> Path:
        """Path to real FAO data file."""
        return Path(sim_path) / "input" / self.output_filename


    # -------------------------------------------------------------------------
    # Download & Transform
    # -------------------------------------------------------------------------
    
    def download(
        self,
        adapter: FaoApiAdapter,
        years: list[str],
        countries: pd.Series,
    ) -> pd.DataFrame:
        """Download data from FAOSTAT API.
        
        Handles the FAO API's 500-record limit by chunking requests.
        """
        items = self._get_items()
        country_list = list(countries)
        n_countries = len(country_list)
        
        # Chunk by country to stay under 500-record API limit
        # For ~166 crop items, we can batch ~3 countries per call
        api_limit = 500
        batch_size = max(1, api_limit // len(items))
        batches = [
            country_list[i:i + batch_size]
            for i in range(0, n_countries, batch_size)
        ]
        
        country_str = f"{n_countries} countries" if n_countries > 1 else "1 country"
        print(f"Downloading {self.name} ({country_str}) from FAOSTAT...")
        
        dfs = []
        is_cs_domain = self.domain == "CS"
        
        for element in self.elements:
            for year in years:
                if is_cs_domain:
                    # Capital Stock: one item per call (different structure)
                    for item in items:
                        try:
                            df = adapter.download_data(
                                domain=self.domain,
                                element=element,
                                year=[year],
                                items=[item],
                                areas=countries,
                                item_code_format="FAO",
                                area_code_format="FAO",
                            )
                            if len(df) > 0:
                                dfs.append(df)
                        except Exception:
                            pass  # Skip failed items silently
                else:
                    # Other domains: batch by country
                    for batch in batches:
                        try:
                            df = adapter.download_data(
                                domain=self.domain,
                                element=element,
                                year=[year],
                                items=items,
                                areas=batch,
                                item_code_format="FAO",
                                area_code_format="FAO",
                            )
                            if len(df) > 0:
                                dfs.append(df)
                        except Exception:
                            pass  # Skip failed batches silently
        
        if not dfs:
            raise RuntimeError(f"No data downloaded for {self.name}")
        
        result = pd.concat(dfs, ignore_index=True)
        print(f"  Done ({len(result)} records)")
        return result

    def transform(self, df: pd.DataFrame) -> xr.Dataset:
        """Transform DataFrame to xarray Dataset with standard coordinates."""
        fao_data_list = FaoData.from_dataframe(df, multi_element=len(self.elements) > 1)
        fao_data_list = [d for d in fao_data_list if d is not None]
        
        if not fao_data_list:
            raise RuntimeError(f"Failed to convert {self.name} to FaoData")
        
        # Translate crop codes to LPJmL names if applicable
        if self._translate_to_lpjml():
            datasets = []
            for fao_data in fao_data_list:
                translated = fao_data.translate_dimension(
                    FaoCropTranslator(
                        dim="item_code",
                        new_standard="LPJmL",
                        reducer="mean",
                        fail_on_partial=False,
                        dim_rename="npft",
                    )
                )
                datasets.append(translated.dataset)
        else:
            datasets = [d.dataset for d in fao_data_list]
        
        ds = xr.merge(datasets) if len(datasets) > 1 else datasets[0]
        
        # Convert coordinates
        ds = self._convert_area_codes(ds)
        ds = self._convert_time_coords(ds)
        
        return ds

    def _convert_area_codes(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert FAO area codes to ISO3."""
        if "area_code" not in ds.dims:
            return ds
        
        fao_to_iso3 = fao_definitions.get_area_code_dict(
            code_standard_out="ISO3", code_standard_in="FAO"
        )
        
        current = ds.area_code.values
        iso3 = [fao_to_iso3.get(str(c), "") for c in current]
        
        # Filter out unmapped codes
        valid_idx = [i for i, code in enumerate(iso3) if code]
        if len(valid_idx) < len(iso3):
            ds = ds.isel(area_code=valid_idx)
            iso3 = [iso3[i] for i in valid_idx]
        
        return ds.assign_coords(area_code=iso3)

    def _convert_time_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert time coordinates to integer years."""
        if "time" not in ds.dims:
            return ds
        
        years = [
            int(t.year) if hasattr(t, "year") else int(t)
            for t in ds.time.values
        ]
        
        ds = ds.assign_coords(time=years)
        ds.time.attrs.update(units="year", long_name="Year")
        return ds

    # -------------------------------------------------------------------------
    # Main Entry Points
    # -------------------------------------------------------------------------
    
    def prepare(
        self,
        cache_path: str | Path | None = None,
        output_path: str | Path | None = None,
        years: tuple[int, int] = (1990, 2020),
        country_codes: list[str] | None = None,
    ) -> xr.Dataset:
        """Download, transform, and optionally save FAO data.
        
        Parameters
        ----------
        cache_path : Path, optional
            Parquet cache for raw downloaded data.
        output_path : Path, optional
            Where to save the final NetCDF.
        years : tuple[int, int]
            Year range (inclusive).
        country_codes : list[str], optional
            ISO3 codes to download. If None, downloads all countries.
            
        Returns
        -------
        xr.Dataset
            Processed dataset.
        """
        year_list = [str(y) for y in range(years[0], years[1] + 1)]
        
        # Try cache first
        if cache_path and Path(cache_path).exists():
            df = pd.read_parquet(cache_path)
        else:
            if not check_fao_api_available():
                raise RuntimeError("FAO API unavailable. Check authentication.")
            
            # Convert country codes
            if country_codes:
                fao_codes = [get_fao_country_code(c) for c in country_codes]
                fao_codes = [c for c in fao_codes if c]
                if not fao_codes:
                    raise RuntimeError(f"No valid FAO codes for: {country_codes}")
                countries = pd.Series(fao_codes)
            else:
                countries = fao_definitions.get_all_country_codes()
            
            adapter = FaoApiAdapter()
            df = self.download(adapter, year_list, countries)
            
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
        
        ds = self.transform(df)
        ds = self._post_process(ds)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(output_path)
        
        return ds

    def ensure(
        self,
        sim_path: str | Path,
        country_codes: list[str] | None = None,
        reference_year: int = 2020,
        years_before: int = 5,
        force_download: bool = False,
    ) -> Path:
        """Ensure FAO data is available, downloading if needed.
        
        Parameters
        ----------
        sim_path : Path
            Simulation directory.
        country_codes : list[str], optional
            ISO3 codes to download.
        reference_year : int
            Target year for data.
        years_before : int
            Years before reference_year to include.
        force_download : bool
            Re-download even if file exists.
            
        Returns
        -------
        Path
            Path to the data file.
            
        Raises
        ------
        RuntimeError
            If FAO data cannot be downloaded and no real data exists.
        """
        output_path = self.get_path(sim_path)
        
        # Use existing real file if available
        if output_path.exists() and not force_download:
            return output_path
        
        years = (reference_year - years_before, reference_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path = output_path.parent / f"{self.name}_cache.parquet"
        
        try:
            self.prepare(
                cache_path=cache_path,
                output_path=output_path,
                years=years,
                country_codes=country_codes,
            )
            return output_path
            
        except Exception as e:
            # No fallback to dummy data - require real FAO data
            raise RuntimeError(
                f"FAO API failed for {self.name}: {e}\n"
                f"Please ensure FAO API authentication is configured correctly.\n"
                f"Expected output path: {output_path}"
            ) from e

    # -------------------------------------------------------------------------
    # Status Checks
    # -------------------------------------------------------------------------
    
    def is_available(self, sim_path: str | Path) -> bool:
        """Check if real FAO data file exists."""
        return self.get_path(sim_path).exists()
