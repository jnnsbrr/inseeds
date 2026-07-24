"""Exogenous data layer for InSEEDS.

Unified access to external data (FAOSTAT, MADRaT) with auto-slicing by entity level.

>>> world.exogenous = load_all(sim_path, grid, start_year, country_codes)
>>> world.exogenous.prices      # Full dataset
>>> country.exogenous.prices    # Auto-sliced for country
"""

import logging
from pathlib import Path
from typing import Any

import xarray as xr

from .base import ExogenousSource
from .accessor import Exogenous

logger = logging.getLogger(__name__)

# Registry: (SourceClass, name, extra_ensure_kwargs)
SOURCES = [
    ("faostat", "FaoProducerPrices", "prices", {"years_before": 10}),
    ("faostat", "FaoCapitalStock", "capital", {"years_before": 10}),
    ("faostat", "FaoGrossProductionValue", "gpv", {"years_before": 15}),
    ("faostat", "FaoGDPPerCapita", "gdp", {"years_before": 10}),
    ("madrat", "ResidueSource", "residue", {}),
]


def load_all(sim_path: str | Path, grid: xr.Dataset, start_year: int, 
             country_codes: list[str], entity: Any = None) -> Exogenous:
    """Load all registered exogenous data sources."""
    datasets, sources = {}, {}
    
    for module, cls_name, name, extra in SOURCES:
        try:
            # Dynamic import
            mod = __import__(f"inseeds.components.exogenous.{module}", fromlist=[cls_name])
            src = getattr(mod, cls_name)()
            
            # Build ensure kwargs based on source type
            kwargs = {"sim_path": sim_path}
            if src.granularity == "country":
                kwargs.update(country_codes=country_codes, reference_year=start_year, **extra)
            else:
                kwargs.update(grid=grid, **extra)
            
            path = src.ensure(**kwargs)
            datasets[name] = xr.open_dataset(path)
            sources[name] = src
            
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")
    
    return Exogenous(entity, datasets=datasets, sources=sources)


__all__ = ["ExogenousSource", "Exogenous", "load_all"]
