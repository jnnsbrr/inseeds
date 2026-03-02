"""FAO data layer for InSEEDS.

This module provides a generic framework for downloading, processing, and
caching FAOSTAT data for use in InSEEDS simulations.

Classes:
    FaoDataset: Abstract base class for FAO dataset handlers
    FaoProducerPrices: Producer prices (PP domain)
    FaoCapitalStock: Capital stock data (CS domain)

Dummy data functions (for testing when FAO API unavailable):
    generate_dummy_producer_prices: Create synthetic price data
    generate_dummy_capital_stock: Create synthetic capital stock data
    ensure_dummy_fao_data: Convenience function to set up test simulations
"""

from .base import FaoDataset
from .producer_prices import FaoProducerPrices
from .capital_stock import FaoCapitalStock
from .dummy import (
    generate_dummy_producer_prices,
    generate_dummy_capital_stock,
    ensure_dummy_fao_data,
)

__all__ = [
    "FaoDataset",
    "FaoProducerPrices",
    "FaoCapitalStock",
    "generate_dummy_producer_prices",
    "generate_dummy_capital_stock",
    "ensure_dummy_fao_data",
]
