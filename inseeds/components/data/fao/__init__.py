"""FAO data layer for InSEEDS.

This module provides a generic framework for downloading, processing, and
caching FAOSTAT data for use in InSEEDS simulations.

Classes:
    FaoDataset: Abstract base class for FAO dataset handlers
    FaoProducerPrices: Producer prices (PP domain)
    FaoCapitalStock: Capital stock data (CS domain)
    FaoGrossProductionValue: Gross production value (QV domain) for crop shares

Agriculture and crop capital shares:
    get_ag_share_of_aff: Get agriculture share of Ag+Forestry+Fishing (static table)
    get_crop_share_from_qv: Get crop share of agriculture from FAO QV data
    compute_crop_capital_share: Compute total crop capital share (ag_share × crop_share)
    AG_SHARE_OF_AFF: Dictionary of country-specific agriculture shares
    DEFAULT_AG_SHARE_OF_AFF: Default value for unknown countries

Dummy data functions (for testing when FAO API unavailable):
    generate_dummy_producer_prices: Create synthetic price data
    generate_dummy_capital_stock: Create synthetic capital stock data
    generate_dummy_gpv: Create synthetic GPV data (for crop share calculation)
    ensure_dummy_fao_data: Convenience function to set up test simulations
"""

from .base import FaoDataset
from .producer_prices import FaoProducerPrices
from .capital_stock import FaoCapitalStock
from .gross_production_value import FaoGrossProductionValue
from .crop_capital_share import (
    get_ag_share_of_aff,
    get_crop_share_from_qv,
    compute_crop_capital_share,
    AG_SHARE_OF_AFF,
    DEFAULT_AG_SHARE_OF_AFF,
)
from .dummy import (
    generate_dummy_producer_prices,
    generate_dummy_capital_stock,
    generate_dummy_gpv,
    ensure_dummy_fao_data,
)

__all__ = [
    "FaoDataset",
    "FaoProducerPrices",
    "FaoCapitalStock",
    "FaoGrossProductionValue",
    "get_ag_share_of_aff",
    "get_crop_share_from_qv",
    "compute_crop_capital_share",
    "AG_SHARE_OF_AFF",
    "DEFAULT_AG_SHARE_OF_AFF",
    "generate_dummy_producer_prices",
    "generate_dummy_capital_stock",
    "generate_dummy_gpv",
    "ensure_dummy_fao_data",
]
