"""FAO GDP per capita dataset handler.

This module downloads GDP per capita data from FAOSTAT (MK domain - Macro Indicators)
for scaling practice costs based on national income levels.

Item: 22008 - Gross Domestic Product
Element: 6119 - Value US$ per capita

The data is used to compute cost scaling factors based on where a country's
GDP per capita falls relative to global percentiles (10th and 90th).

References:
- FAO (2023). FAOSTAT Macro Indicators methodology.
- World Bank country classification by income.

Output dimensions:
- time: years
- area_code: country codes (ISO3 format)

Output variables:
- gdp_per_capita: GDP in US$ per capita
- gdp_percentile_10: 10th percentile value (computed once)
- gdp_percentile_90: 90th percentile value (computed once)
"""

import logging
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import xarray as xr

from .base import FaoDataset, get_value_with_fallback

logger = logging.getLogger(__name__)


class FaoGDPPerCapita(FaoDataset):
    """FAO GDP per capita dataset handler.

    Downloads GDP per capita from FAOSTAT MK domain (Macro Indicators).
    Used for scaling practice costs based on national income levels.

    Attributes
    ----------
    domain : str
        "MK" (Macro Indicators)
    elements : list[str]
        ["6119"] (Value US$ per capita)
    name : str
        "gdp_per_capita"
    output_filename : str
        "fao_gdp_per_capita.nc"
    """

    @property
    @override
    def domain(self) -> str:
        return "MK"

    @property
    @override
    def elements(self) -> list[str]:
        # 6119: Value US$ per capita
        return ["6119"]

    @property
    @override
    def name(self) -> str:
        return "gdp_per_capita"

    @property
    @override
    def output_filename(self) -> str:
        return "fao_gdp_per_capita.nc"

    @override
    def _translate_to_lpjml(self) -> bool:
        """GDP uses country codes, not crop codes."""
        return False

    @override
    def _get_items(self) -> pd.Series:
        """Return item codes for GDP.
        
        In MK domain:
        - 22008: Gross Domestic Product
        """
        return pd.Series(["22008"])

    @override
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Extract GDP per capita and compute global percentiles.

        Computes 10th and 90th percentiles across all countries for cost scaling.
        Countries below p10 use minimum costs, above p90 use maximum costs.
        """
        print("  Processing GDP per capita data...")
        print(f"    Dataset vars: {list(ds.data_vars)}")
        print(f"    Dataset dims: {dict(ds.sizes)}")

        element_code = "6119"  # Value US$ per capita
        
        # The element code might be the only variable, or data might be structured differently
        if element_code in ds.data_vars:
            data = ds[element_code]
        elif len(ds.data_vars) == 1:
            # If only one variable, use it regardless of name
            var_name = list(ds.data_vars)[0]
            print(f"    Using variable: {var_name}")
            data = ds[var_name]
        else:
            raise RuntimeError(f"Expected element {element_code} not found in MK dataset. Available: {list(ds.data_vars)}")
        
        # MK domain may have item_code dimension - select GDP if present
        if "item_code" in data.dims:
            if "22008" in data.item_code.values:
                data = data.sel(item_code="22008")
            else:
                # Take first/only item
                data = data.isel(item_code=0)
        
        result_ds = xr.Dataset()
        result_ds["gdp_per_capita"] = data
        result_ds["gdp_per_capita"].attrs.update(
            units="USD_per_capita",
            long_name="Gross Domestic Product per capita",
            source="FAOSTAT MK domain",
            area_code_format="ISO3",
        )
        
        # Use FIXED percentiles based on World Bank global GDP distribution
        # This ensures consistency regardless of which countries are in the run.
        #
        # Based on World Bank 2020-2022 GDP per capita (current USD) distribution:
        # - 10th percentile globally: ~$1,500 (low-income countries)
        # - 90th percentile globally: ~$45,000 (high-income OECD countries)
        #
        # Reference: World Bank World Development Indicators
        # https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
        p10 = 1500.0   # Low-income threshold (e.g., Niger, Burundi)
        p90 = 45000.0  # High-income threshold (e.g., Germany, UK, Japan)
        
        result_ds.attrs["gdp_percentile_10"] = p10
        result_ds.attrs["gdp_percentile_90"] = p90
        
        print(f"    GDP percentiles: p10={p10:.0f}, p90={p90:.0f} USD/capita")
        
        return result_ds


def get_gdp_cost_ratio(
    gdp_per_capita: float,
    p10: float,
    p90: float,
) -> float:
    """Compute cost scaling ratio based on GDP per capita.
    
    Maps GDP to [0, 1] range:
    - GDP <= p10: ratio = 0 (minimum costs)
    - GDP >= p90: ratio = 1 (maximum costs)
    - Between: linear interpolation
    
    Parameters
    ----------
    gdp_per_capita : float
        Country's GDP per capita in USD.
    p10 : float
        10th percentile GDP value.
    p90 : float
        90th percentile GDP value.
        
    Returns
    -------
    float
        Cost ratio in [0, 1] range.
    """
    if p90 <= p10:
        return 0.5  # Fallback if percentiles invalid
    
    ratio = (gdp_per_capita - p10) / (p90 - p10)
    return max(0.0, min(1.0, ratio))
