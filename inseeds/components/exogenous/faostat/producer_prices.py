"""FAO Producer Prices dataset handler.

This module downloads producer prices from FAOSTAT (PP domain) and converts them
to the LPJmL CFT format for use in InSEEDS profit calculations.

Output dimensions:
- time: years
- area_code: country codes (ISO3 format)
- npft: LPJmL crop functional types

Unit conversion:
- FAO: USD/tonne (fresh weight)
- Output: USD/tonne (dry matter)
"""

from pathlib import Path
from typing import override

import pandas as pd
import xarray as xr

from copan_eval.fao import fao_definitions, FaoCropTranslator

from .base import FaoDataset


# Dry matter factors from Wirsenius (2000), Table A1.II
# Used to convert fresh weight prices to dry matter prices
DRY_MATTER_FACTORS = pd.Series(
    [
        0.88, 0.87, 0.88, 0.88,
        0.90, 0.24, 0.35, 0.93,
        0.91, 0.94, 0.92, 0.27,
        1.00, 1.00, 1.00, 1.00,
        1.00, 1.00
    ],
    index=[
        "temperate cereals", "rice", "maize", "tropical cereals",
        "pulses", "temperate roots", "tropical roots", "oil crops sunflower",
        "oil crops soybean", "oil crops groundnut", "oil crops rapeseed", "sugarcane",
        "others", "Others, annual", "Others, perennial", "grassland",
        "biomass_grass", "biomass_tree"
    ]
)
DRY_MATTER_FACTORS.index.name = "npft"


class FaoProducerPrices(FaoDataset):
    """FAO Producer Prices dataset handler.

    Downloads producer prices from FAOSTAT PP domain and converts to
    LPJmL CFT format with dry matter basis.

    Attributes
    ----------
    domain : str
        "PP" (Producer Prices)
    elements : list[str]
        ["5532"] (Producer Price USD/tonne)
    name : str
        "pft_prices"
    output_filename : str
        "fao_pft_prices.nc"
    """

    @property
    @override
    def domain(self) -> str:
        return "PP"

    @property
    @override
    def elements(self) -> list[str]:
        return ["5532"]  # Producer Price (USD/tonne)

    @property
    @override
    def name(self) -> str:
        return "pft_prices"

    @property
    @override
    def output_filename(self) -> str:
        return "fao_pft_prices.nc"

    @override
    def _get_items(self) -> pd.Series:
        """Return only FAO crop codes that map to LPJmL CFTs.
        
        This reduces API calls by only requesting the ~166 crops that
        are actually used in LPJmL, instead of all ~176 primary crops.
        """
        translator = FaoCropTranslator(
            dim="item_code",
            new_standard="LPJmL",
            reducer="mean"
        )
        lpjml_mapped = translator.dictionary[
            translator.dictionary["LPJmL_name"].notna()
        ]
        return lpjml_mapped["FAOSTAT_code"].astype(str)

    @override
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply dry matter conversion to prices.

        Converts fresh weight prices to dry matter prices by dividing
        by the dry matter factor. For example, potatoes have DM=0.24,
        so $117/t fresh becomes $487.50/t dry matter.
        """
        dm_factors = DRY_MATTER_FACTORS.to_xarray()

        for var in ds.data_vars:
            ds[var] = ds[var] / dm_factors
            ds[var].attrs["units"] = "USD/tonne_dry_matter"
            ds[var].attrs["long_name"] = "Producer price per tonne dry matter"
            ds[var].attrs["source"] = "FAOSTAT PP domain, element 5532"
            ds[var].attrs["conversion"] = "Fresh weight to dry matter via Wirsenius (2000)"
            ds[var].attrs["area_code_format"] = "ISO3"

        return ds
