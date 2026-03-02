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
        # 6184: Gross Fixed Capital Formation (Agriculture, Forestry and Fishing)
        # 6185: Consumption of Fixed Capital (Agriculture, Forestry and Fishing)
        # 6186: Net Capital Stocks (Agriculture, Forestry and Fishing)
        return ["6184", "6185", "6186"]

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
        """Return item code for Agriculture, Forestry and Fishing sector."""
        # Item code 22041 is "Agriculture, Forestry and Fishing"
        return pd.Series(["22041"])

    @override
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute depreciation and investment rates from GFCF, CFC, and NCS.

        Following standard capital accounting (Jorgenson 1963, OECD 2009):
        - Depreciation rate δ = CFC / NCS: annual capital wear
        - Investment rate i = GFCF / NCS: gross investment relative to stock

        The investment rate serves as a country-specific proxy for the fraction
        of capital renewed annually, usable as a savings/reinvestment rate.
        """
        print("  Computing depreciation and investment rates...")

        # Add metadata to raw variables
        if "6184" in ds.data_vars:
            ds["6184"].attrs["units"] = "million_USD"
            ds["6184"].attrs["long_name"] = "Gross Fixed Capital Formation"
            ds["6184"].attrs["source"] = "FAOSTAT CS domain"
            ds["6184"].attrs["sector"] = "Agriculture, Forestry and Fishing"
            ds["6184"].attrs["area_code_format"] = "ISO3"

        if "6185" in ds.data_vars:
            ds["6185"].attrs["units"] = "million_USD"
            ds["6185"].attrs["long_name"] = "Consumption of Fixed Capital"
            ds["6185"].attrs["source"] = "FAOSTAT CS domain"
            ds["6185"].attrs["sector"] = "Agriculture, Forestry and Fishing"
            ds["6185"].attrs["area_code_format"] = "ISO3"

        if "6186" in ds.data_vars:
            ds["6186"].attrs["units"] = "million_USD"
            ds["6186"].attrs["long_name"] = "Net Capital Stocks"
            ds["6186"].attrs["source"] = "FAOSTAT CS domain"
            ds["6186"].attrs["sector"] = "Agriculture, Forestry and Fishing"
            ds["6186"].attrs["area_code_format"] = "ISO3"

        # Depreciation rate: δ = CFC / NCS (Jorgenson 1963)
        if "6185" in ds.data_vars and "6186" in ds.data_vars:
            ds["depreciation_rate"] = ds["6185"] / ds["6186"]
            ds["depreciation_rate"].attrs["units"] = "1/year"
            ds["depreciation_rate"].attrs["long_name"] = "Depreciation rate"
            ds["depreciation_rate"].attrs["source"] = "Computed from FAOSTAT CS domain"
            ds["depreciation_rate"].attrs["formula"] = "CFC / NCS"
            ds["depreciation_rate"].attrs["reference"] = "Jorgenson (1963), OECD (2009)"
            ds["depreciation_rate"].attrs["area_code_format"] = "ISO3"

        # Investment rate: i = GFCF / NCS (gross investment relative to stock)
        if "6184" in ds.data_vars and "6186" in ds.data_vars:
            ds["investment_rate"] = ds["6184"] / ds["6186"]
            ds["investment_rate"].attrs["units"] = "1/year"
            ds["investment_rate"].attrs["long_name"] = "Investment rate"
            ds["investment_rate"].attrs["source"] = "Computed from FAOSTAT CS domain"
            ds["investment_rate"].attrs["formula"] = "GFCF / NCS"
            ds["investment_rate"].attrs["reference"] = "OECD (2009) Measuring Capital"
            ds["investment_rate"].attrs["area_code_format"] = "ISO3"

        return ds

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
