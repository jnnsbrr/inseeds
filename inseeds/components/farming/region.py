"""The inseeds_farmer.region class."""
import numpy as np


class Region:
    """Region entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of Region."""
        super().__init__(**kwargs)


class Country(Region):
    """Country entity type mixin class."""

    def __init__(self, **kwargs):
        """Initialize an instance of Country."""
        super().__init__(**kwargs)

    @property
    def farmers(self):
        """Return farmers from this country's cells.

        Uses SocialSystem.individuals from lpjml.Country.
        """
        farmers = {
            farmer
            for farmer in self.individuals
            if farmer.__class__.__name__ == "Farmer"
        }
        return farmers


    @property
    def cropland_area(self):
        """Total cropland area in hectares for this country.

        Computed directly from LPJmL data as sum of (cftfrac * area) across
        all cells in the country. Available immediately after country creation,
        without needing farmers to be initialized.

        Returns
        -------
        float
            Cropland area in hectares (minimum 1.0 to avoid division by zero).
        """
        if not hasattr(self, "_cropland_area"):
            cftfrac = self.from_earth.cftfrac
            area = self.area

            total_cftfrac = cftfrac.sum("band")

            if hasattr(area, "values"):
                area_km2 = np.asarray(area.values)
            else:
                area_km2 = np.asarray(area)

            area_ha = area_km2 * 100.0
            cropland_ha = float((total_cftfrac.values * area_ha).sum())

            self._cropland_area = max(cropland_ha, 1.0)
        return self._cropland_area


    def update(self, t):
        """Update country and its farmers."""
        super().update(t)

        farmers_sorted = sorted(
            self.farmers, key=lambda farmer: farmer.avg_hdate
        )

        for farmer in farmers_sorted:
            farmer.update(t)
