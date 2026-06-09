"""FAOSTAT data layer for InSEEDS.

Provides handlers for downloading, processing, and caching FAOSTAT data:
- Producer prices (PP domain)
- Capital stock (CS domain)  
- Gross production value (QV domain) for crop shares

Tiered fallback for missing data:
- get_value_with_fallback: country -> neighbours -> global mean
"""

from .base import FaoDataset, FallbackResult, get_value_with_fallback
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

__all__ = [
    "FaoDataset",
    "FaoProducerPrices",
    "FaoCapitalStock",
    "FaoGrossProductionValue",
    "FallbackResult",
    "get_value_with_fallback",
    "get_ag_share_of_aff",
    "get_crop_share_from_qv",
    "compute_crop_capital_share",
    "AG_SHARE_OF_AFF",
    "DEFAULT_AG_SHARE_OF_AFF",
]
