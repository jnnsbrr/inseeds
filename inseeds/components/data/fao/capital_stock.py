"""FAO Capital Stock dataset handler.

This module downloads capital stock data from FAOSTAT (CS domain) including
Gross Fixed Capital Formation (GFCF), Consumption of Fixed Capital (CFC),
and Net Capital Stocks (NCS) for the Agriculture, Forestry and Fishing sector.

Derived rates follow standard capital accounting (Jorgenson, 1963; OECD, 2009):
- Depreciation rate: δ = CFC / NCS (annual capital wear)
- Investment rate: i = GFCF / NCS (gross investment relative to stock)

The investment rate represents the fraction of capital stock renewed annually
through new investment. This can serve as a proxy for savings/reinvestment
behavior at the country level.

References:
- Jorgenson, D.W. (1963). Capital Theory and Investment Behavior. AER.
- OECD (2009). Measuring Capital - OECD Manual.
- FAO (2023). FAOSTAT Capital Stock methodology.

Output dimensions:
- time: years
- area_code: country codes (ISO3 format)

Output variables:
- 6184: Gross Fixed Capital Formation (million USD)
- 6185: Consumption of Fixed Capital (million USD)
- 6186: Net Capital Stocks (million USD)
- depreciation_rate: CFC / NCS (1/year)
- investment_rate: GFCF / NCS (1/year)
"""

from pathlib import Path
from typing import override

import pandas as pd
import xarray as xr

from .base import FaoDataset
from .dummy import generate_dummy_capital_stock


class FaoCapitalStock(FaoDataset):
    """FAO Capital Stock dataset handler.

    Downloads capital stock data from FAOSTAT CS domain and computes
    depreciation and investment rates following standard capital accounting.

    Attributes
    ----------
    domain : str
        "CS" (Capital Stock)
    elements : list[str]
        ["6184", "6185", "6186"] (GFCF, CFC, NCS for Agriculture sector)
    name : str
        "capital_stock"
    output_filename : str
        "fao_capital_stock.nc"
    """

    @property
    @override
    def domain(self) -> str:
        return "CS"

    @property
    @override
    def elements(self) -> list[str]:
        # CS domain uses a single element code for value type
        # 6110: Value US$ (current prices)
        return ["6110"]

    @property
    @override
    def name(self) -> str:
        return "capital_stock"

    @property
    @override
    def output_filename(self) -> str:
        return "fao_capital_stock.nc"

    @override
    def _translate_to_lpjml(self) -> bool:
        """Capital stock uses sector codes, not crop codes."""
        return False

    @override
    def _get_items(self) -> pd.Series:
        """Return item codes for capital stock types.
        
        In CS domain, items represent the capital stock types:
        - 22030: Gross Fixed Capital Formation (GFCF)
        - 22031: Consumption of Fixed Capital (CFC)
        - 22034: Net Capital Stocks (NCS)
        """
        return pd.Series(["22030", "22031", "22034"])

    @override
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute depreciation and investment rates from GFCF, CFC, and NCS.

        In CS domain, items represent capital stock types (GFCF, CFC, NCS).
        The element code (6110) represents "Value US$".

        Following standard capital accounting (Jorgenson 1963, OECD 2009):
        - Depreciation rate δ = CFC / NCS: annual capital wear
        - Investment rate i = GFCF / NCS: gross investment relative to stock
        """
        print("  Computing depreciation and investment rates...")

        # CS domain returns data with item_code dimension containing 22030, 22031, 22034
        # We need to rename these to meaningful variable names
        element_code = "6110"  # Value US$
        
        if element_code not in ds.data_vars:
            raise RuntimeError(f"Expected element {element_code} not found in CS dataset")

        data = ds[element_code]
        
        # Create separate variables for each capital stock type
        item_map = {
            "22030": "gfcf",  # Gross Fixed Capital Formation
            "22031": "cfc",   # Consumption of Fixed Capital
            "22034": "ncs",   # Net Capital Stocks
        }
        
        result_ds = xr.Dataset()
        
        for item_code, var_name in item_map.items():
            if "item_code" in data.dims and item_code in data.item_code.values:
                result_ds[var_name] = data.sel(item_code=item_code)
                result_ds[var_name].attrs["units"] = "million_USD"
                result_ds[var_name].attrs["source"] = "FAOSTAT CS domain"
                result_ds[var_name].attrs["area_code_format"] = "ISO3"
        
        # Add long names
        if "gfcf" in result_ds:
            result_ds["gfcf"].attrs["long_name"] = "Gross Fixed Capital Formation"
        if "cfc" in result_ds:
            result_ds["cfc"].attrs["long_name"] = "Consumption of Fixed Capital"
        if "ncs" in result_ds:
            result_ds["ncs"].attrs["long_name"] = "Net Capital Stocks"

        # Compute derived rates
        if "cfc" in result_ds and "ncs" in result_ds:
            result_ds["depreciation_rate"] = result_ds["cfc"] / result_ds["ncs"]
            result_ds["depreciation_rate"].attrs["units"] = "1/year"
            result_ds["depreciation_rate"].attrs["long_name"] = "Depreciation rate"
            result_ds["depreciation_rate"].attrs["formula"] = "CFC / NCS"
            result_ds["depreciation_rate"].attrs["reference"] = "Jorgenson (1963), OECD (2009)"

        if "gfcf" in result_ds and "ncs" in result_ds:
            result_ds["investment_rate"] = result_ds["gfcf"] / result_ds["ncs"]
            result_ds["investment_rate"].attrs["units"] = "1/year"
            result_ds["investment_rate"].attrs["long_name"] = "Investment rate"
            result_ds["investment_rate"].attrs["formula"] = "GFCF / NCS"
            result_ds["investment_rate"].attrs["reference"] = "OECD (2009) Measuring Capital"

        return result_ds

    @override
    def _generate_dummy_fallback(
        self,
        output_path: Path,
        years: tuple[int, int],
    ) -> None:
        """Generate dummy capital stock data when FAO API fails."""
        generate_dummy_capital_stock(
            years=years,
            output_path=output_path,
        )
        print(f"  Saved DUMMY capital stock to: {output_path}")
        print(f"  ⚠ Delete this file and provide real FAO data for production runs!")
