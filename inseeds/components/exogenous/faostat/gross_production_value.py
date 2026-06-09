"""FAO Gross Production Value dataset handler.

This module downloads Gross Production Value (GPV) data from FAOSTAT (QV domain)
to provide the raw data needed for computing crop share of agricultural output.

FAOSTAT QV domain provides:
- Gross Production Value by commodity
- Element code 152: Gross Production Value (constant 2014-2016 thousand I$)
  - Better coverage than current US$ (element 57)
  - Data available up to ~2017 for most countries
- Covers crops and livestock products (NOT forestry or fishing)

Item codes used:
- 2041: Crops - all primary crops aggregate
- 2051: Agriculture - total agriculture (crops + livestock)

Output variables:
- gpv_crops: GPV for crops (item code 2041)
- gpv_agriculture: GPV for total agriculture (item code 2051)

The crop share calculation (gpv_crops / gpv_agriculture) is done in
crop_capital_share.py, which combines it with the agriculture share of
Ag+Forestry+Fishing to compute the final crop capital share.

Note: FAO QV data has a lag of several years. The code downloads a wider
year range (10 years) to ensure data availability and uses the most recent
available year for each country.

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


# Item codes for QV domain aggregates
# 2041: Crops - all primary crops aggregate
# 2051: Agriculture - total agriculture (crops + livestock, excludes forestry/fishing)
CROP_ITEM = "2041"  # Crops
AGRICULTURE_ITEM = "2051"  # Agriculture (crops + livestock)


class FaoGrossProductionValue(FaoDataset):
    """FAO Gross Production Value dataset handler.

    Downloads GPV data from FAOSTAT QV domain for computing crop share
    of total agricultural production by country.

    Attributes
    ----------
    domain : str
        "QV" (Value of Agricultural Production)
    elements : list[str]
        ["152"] (Gross Production Value, constant 2014-2016 thousand I$)
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
        # 152: Gross Production Value (constant 2014-2016 thousand I$)
        # Better coverage than element 57 (current US$)
        return ["152"]

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
        return pd.Series([CROP_ITEM, AGRICULTURE_ITEM])

    @override
    def _post_process(self, ds: xr.Dataset) -> xr.Dataset:
        """Extract GPV for crops and agriculture from QV data.

        Stores raw GPV values - the crop share calculation is done separately
        in crop_capital_share.py using country-specific agriculture shares.
        """
        print("  Extracting GPV for crops and agriculture...")

        element_code = "152"  # Gross Production Value (constant 2014-2016 thousand I$)

        if element_code not in ds.data_vars:
            raise RuntimeError(f"Expected element {element_code} not found in QV dataset")

        data = ds[element_code]

        result_ds = xr.Dataset()

        if "item_code" in data.dims:
            # Get GPV for crops
            if CROP_ITEM in data.item_code.values:
                gpv_crops = data.sel(item_code=CROP_ITEM)
                result_ds["gpv_crops"] = gpv_crops
                result_ds["gpv_crops"].attrs["units"] = "thousand_IntD"
                result_ds["gpv_crops"].attrs["long_name"] = "Gross Production Value - Crops"
                result_ds["gpv_crops"].attrs["fao_item_code"] = CROP_ITEM

            # Get GPV for total agriculture
            if AGRICULTURE_ITEM in data.item_code.values:
                gpv_total = data.sel(item_code=AGRICULTURE_ITEM)
                result_ds["gpv_agriculture"] = gpv_total
                result_ds["gpv_agriculture"].attrs["units"] = "thousand_IntD"
                result_ds["gpv_agriculture"].attrs["long_name"] = "Gross Production Value - Agriculture"
                result_ds["gpv_agriculture"].attrs["fao_item_code"] = AGRICULTURE_ITEM
                result_ds["gpv_agriculture"].attrs["note"] = (
                    "Agriculture = crops + livestock (excludes forestry and fishing)"
                )

        return result_ds
