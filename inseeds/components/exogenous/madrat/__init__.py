"""MADRaT data layer for InSEEDS.

Provides handlers for MADRaT-derived datasets:
- Residue fractions (burnt, removed, recycled)
"""

from .residue import ResidueSource

__all__ = ["ResidueSource"]
