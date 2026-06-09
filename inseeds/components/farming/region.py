"""The inseeds_farmer.region class."""
import numpy as np

from inseeds.components.farming.farmer import NON_CROPS


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

    def get_from_earth(self, var_name, drop_band=None):
        """Get variable from country's from_earth, handling multi-year data.

        Automatically selects the most recent time step if multiple exist.

        Parameters
        ----------
        var_name : str
            Name of the variable in from_earth (e.g. cftfrac).
        drop_band : list, optional
            If given, drop the specified bands from the DataArray.

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
        # Drop bands if specified
        if drop_band is not None:
            data = data.drop_sel(band=drop_band)
        return data

    @property
    def farmers(self):
        """Return farmers from this country's cells, sorted by harvest date.

        Includes Farmer and all subclasses (e.g., CAFarmer).
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
            # Exclude managed grassland - it's not a crop
            cftfrac = self.get_from_earth("cftfrac", drop_band=NON_CROPS)
            area = self.area
            total_cftfrac = cftfrac.sum("band")

            # Extract values and squeeze singleton dimensions (e.g., band dim of size 1)
            # This is critical: terr_area has shape (n_cells, 1) due to nbands=1,
            # while cftfrac after sum has shape (n_cells,). Without squeeze,
            # numpy broadcasting would create incorrect results.
            area_values = area.values if hasattr(area, "values") else np.asarray(area)
            cftfrac_values = total_cftfrac.values
            area_values = np.squeeze(area_values)
            cftfrac_values = np.squeeze(cftfrac_values)

            # Area from lpjml is in m², convert to hectares (1 ha = 10,000 m²)
            area_ha = area_values / 10000.0
            cropland_ha = float((cftfrac_values * area_ha).sum())
            self._cropland_area = max(cropland_ha, 1.0)
        return self._cropland_area


    def update(self, t):
        """Update country and its farmers.
        
        inseeds handles entity updates - pycopanlpjml.Region.update() is a no-op.
        """
        super().update(t)

        for farmer in self.farmers:
            farmer.update(t)

