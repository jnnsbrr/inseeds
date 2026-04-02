"""FAO Gross Production Value dataset handler.

This module downloads Gross Production Value (GPV) data from FAOSTAT (QV domain)
to provide the raw data needed for computing crop share of agricultural output.

FAOSTAT QV domain provides:
- Gross Production Value (current thousand US$) by commodity
- Element code 57: Gross Production Value (current thousand US$)
- Covers crops and livestock products (NOT forestry or fishing)

Item codes used:
- 1717: Crops (PIN) - Primary crops aggregate
- 2051: Agriculture (PIN) - Total agriculture (crops + livestock)

Output variables:
- gpv_crops: GPV for crops (item code 1717)
- gpv_agriculture: GPV for total agriculture (item code 2051)

The crop share calculation (gpv_crops / gpv_agriculture) is done in
crop_capital_share.py, which combines it with the agriculture share of
Ag+Forestry+Fishing to compute the final crop capital share.

References:
- FAO (2023). Value of Agricultural Production methodology.
- https://www.fao.org/faostat/en/#data/QV
"""

from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import xarray as xr

from .base import FaoDataset


# Item codes for crop aggregates in QV domain
# These are aggregate codes that sum all crops
CROP_AGGREGATE_ITEMS = [
    "1717",  # Crops (PIN) - Primary crops aggregate
]

# Item codes for total agriculture (crops + livestock)
TOTAL_AGRICULTURE_ITEMS = [
    "2051",  # Agriculture (PIN) - Total agriculture aggregate
]


class FaoGrossProductionValue(FaoDataset):
    """FAO Gross Production Value dataset handler.

    Downloads GPV data from FAOSTAT QV domain for computing crop share
    of total agricultural production by country.

    Attributes
    ----------
    domain : str
        "QV" (Value of Agricultural Production)
    elements : list[str]
        ["57"] (Gross Production Value, current thousand US$)
    name : str
        "gross_production_value"
    output_filename : str
        "fao_crop_share.nc"
    """

    @property
    @override
    def domain(self) -> str:
        return "QV"

    @property
    @override
    def elements(self) -> list[str]:
        # 57: Gross Production Value (current thousand US$)
        # Note: FAOSTAT uses "thousand US$" not "million US$"
        return ["57"]

    @property
    @override
    def name(self) -> str:
        return "gross_production_value"

    @property
    @override
    def output_filename(self) -> str:
        return "fao_crop_share.nc"

    @override
    def _translate_to_lpjml(self) -> bool:
        """GPV uses aggregate codes, not individual crop codes."""
        return False

    @override
    def _get_items(self) -> pd.Series:
        """Return item codes for crop and total agriculture aggregates."""
        return pd.Series(CROP_AGGREGATE_ITEMS + TOTAL_AGRICULTURE_ITEMS)

    @override
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Extract GPV for crops and agriculture from QV data.

        Stores raw GPV values - the crop share calculation is done separately
        in crop_capital_share.py using country-specific agriculture shares.
        """
        print("  Extracting GPV for crops and agriculture...")

        element_code = "57"  # Gross Production Value (current thousand US$)

        if element_code not in ds.data_vars:
            raise RuntimeError(f"Expected element {element_code} not found in QV dataset")

        data = ds[element_code]

        # Extract crops and total agriculture GPV
        crop_item = "1717"  # Crops (PIN)
        total_item = "2051"  # Agriculture (PIN)

        result_ds = xr.Dataset()

        if "item_code" in data.dims:
            # Get GPV for crops
            if crop_item in data.item_code.values:
                gpv_crops = data.sel(item_code=crop_item)
                result_ds["gpv_crops"] = gpv_crops
                result_ds["gpv_crops"].attrs["units"] = "thousand_USD"
                result_ds["gpv_crops"].attrs["long_name"] = "Gross Production Value - Crops"
                result_ds["gpv_crops"].attrs["fao_item_code"] = crop_item

            # Get GPV for total agriculture
            if total_item in data.item_code.values:
                gpv_total = data.sel(item_code=total_item)
                result_ds["gpv_agriculture"] = gpv_total
                result_ds["gpv_agriculture"].attrs["units"] = "thousand_USD"
                result_ds["gpv_agriculture"].attrs["long_name"] = "Gross Production Value - Agriculture"
                result_ds["gpv_agriculture"].attrs["fao_item_code"] = total_item
                result_ds["gpv_agriculture"].attrs["note"] = (
                    "Agriculture = crops + livestock (excludes forestry and fishing)"
                )

        return result_ds

    @override
    def _generate_dummy_fallback(
        self,
        output_path: Path,
        years: tuple[int, int],
    ) -> None:
        """Generate dummy crop share data when FAO API fails."""
        from .dummy import generate_dummy_crop_share
        generate_dummy_crop_share(
            years=years,
            output_path=output_path,
        )
        print(f"  Saved DUMMY crop share to: {output_path}")
        print(f"  ⚠ Delete this file and provide real FAO data for production runs!")
