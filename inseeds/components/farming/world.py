"""The inseeds_farmer_mnagement.world class."""

import inseeds.components.base as base


class World(base.World):
    """World entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        super().__init__(**kwargs)

    @property
    def farmers(self):
        """Return all farmers (including subclasses), sorted by harvest date.

        Cached and sorted on first access - the order is deterministic and
        reflects the agricultural calendar (early harvesters first).
        """
        if not hasattr(self, "_farmers_cache"):
            # One-time expensive lookup (MRO traversal + string matching)
            farmers_set = {
                ind for ind in self.individuals
                if any("Farmer" in c.__name__ for c in ind.__class__.__mro__)
            }
            # Sort once by avg_hdate for deterministic order
            self._farmers_cache = sorted(farmers_set, key=lambda f: f.avg_hdate)
        return self._farmers_cache

    def update(self, t):
        """Update all farmers (for models without countries)."""
        super().update(t)

        # Update farmers
        for farmer in self.farmers:
            farmer.update(t)
