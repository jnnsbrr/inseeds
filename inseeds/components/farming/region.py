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

    def _get_from_earth(self, var_name):
        """Get variable from country's from_earth, handling multi-year data.

        Automatically selects the most recent time step if multiple exist.

        Parameters
        ----------
        var_name : str
            Name of the variable in from_earth (e.g. cftfrac).

        Returns
        -------
        xarray.DataArray
            The data with time dimension removed if multi-year.
        """
        data = getattr(self.from_earth, var_name, None)
        if data is None:
            raise AttributeError(
                f"{var_name} not in country.from_earth"
            )
        # If data has multiple time steps, select the most recent one
        if hasattr(data, 'time') and len(data.time) > 1:
            data = data.isel(time=-1)
        return data

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
            cftfrac = self._get_from_earth("cftfrac")
            area = self.area
            total_cftfrac = cftfrac.sum("band")

            # Area from pycopanlpjml is in m², convert to hectares (1 ha = 10,000 m²)
            if hasattr(area, "values"):
                area_m2 = np.asarray(area.values)
            else:
                area_m2 = np.asarray(area)

            area_ha = area_m2 / 10000.0
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
