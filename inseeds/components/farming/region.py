"""The inseeds_farmer.region class."""

import inseeds.components.base as base


class Region(base.World):
    """Region entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of Region."""
        super().__init__(**kwargs)



class Country(Region):
    """Country entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of Country."""
        super().__init__(**kwargs)

    def update(self, t):
        super().update(t)

        farmers_sorted = sorted(
            self.farmers, key=lambda farmer: farmer.avg_hdate
        )
        for farmer in farmers_sorted:
            farmer.update(t)
