from pycopancore.data_model.variable import Variable
from pycopancore.data_model.master_data_model.dimensions_and_units import (
    DimensionsAndUnits as DAU,
)

from inseeds.components import base
from inseeds.components import farming
from inseeds.components.farming import ConservationAgricultureFarmer
from inseeds.components.farming.ca_country import CACountry
from inseeds.components import lpjml


class Farmer(ConservationAgricultureFarmer):
    """Farmer entity type."""

    output_variables = base.Output(
        aft_id=Variable("AFT ID", "unique identifier for agent"),
        avg_hdate=Variable(
            "average harvest date",
            "weighted average harvest date of grown crops (by crop area)",
            unit=DAU.doy,
        ),
        soilc=Variable(
            "soil organic carbon",
            "soil organic carbon content of agent land",
            unit=DAU.gC_per_m2,
        ),
        cropyield=Variable(
            "average crop yield",
            "average crop yield of agent land weighted by crop area",
            unit=DAU.gC_per_m2,
        ),
        root_moisture=Variable(
            "root moisture",
            "root moisture of agent land (mm)",
        ),
        tillage=Variable(
            "agent tillage behaviour",
            "conventional=1, conservation=0",
            datatype=bool,
        ),
        cover_crop=Variable(
            "cover crop option",
            "0=none, 1=non-legumes, 2=legumes",
        ),
        residue_on_field=Variable(
            "residue on field",
            "residue on field of agent land",
            unit=DAU.gC_per_m2,
        ),
        capital=Variable(
            "farm capital",
            "farm capital stock in USD (total, scaled by farm size)",
        ),
        farm_size=Variable(
            "farm size",
            "farm size in hectares (sum of cftfrac * area)",
            unit=DAU.ha,
        ),
    )


class Cell(lpjml.Cell, farming.Cell):
    """Cell entity type."""

    pass


class Country(lpjml.Country, CACountry, base.Country):
    """Country entity type with FAO data for CA capital initialization.
    """
    pass


class World(lpjml.World, farming.World):
    """World entity type."""

    pass


class Model(lpjml.Model):
    """Model class for the InSEEDS Social model integrating the LPJmL model and
    coupling component as well as the farmer management component.
    """

    name = "InSEEDS farmer management"
    description = "InSEEDS farmer management model representing only social \
    dynamics and decision-making on the basis of the TPB"

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        # Initialize the parent classes first
        super().__init__(**kwargs)

        # Ensure self.lpjml is initialized before accessing it
        if not hasattr(self, "lpjml") or self.lpjml is None:
            raise ValueError("lpjml must be initialized in the parent class.")

        # initialize LPJmL world
        self.world = World(
            model=self,
            input=self.lpjml.read_input(),
            output=self.lpjml.read_historic_output(), # .isel(time=[-1]),
            grid=self.lpjml.grid,
            # country_code is the array of country codes
            country_code=self.lpjml.country,
            area=self.lpjml.terr_area,
        )

        # Initialize countries if country data is available
        if (
            self.lpjml.country is not None
            and hasattr(self.world, "country_code")
            and self.world.country_code is not None
        ):
            self._preload_fao_data()
            self.init_countries(country_class=Country)
        else:
            self.countries = []
            print(
                "Warning: No country data available. "
                "Running without country-level structure."
            )

        # initialize cells
        self.init_cells(cell_class=Cell)

        # initialize farmers (uses historic data for trend initialization)
        self.init_farmers(farmer_class=Farmer)

        # After initialization, trim from_earth to single year for normal updates
        # This is needed because we pass multi-year historic data for initialization
        # but update_lpjml expects single-year data during the simulation loop
        self._trim_from_earth_to_current_year()

    def _trim_from_earth_to_current_year(self):
        """Trim from_earth data to only the most recent year.

        During initialization, we pass multi-year historic output to give
        farmers initial history for trend computation. After initialization,
        we trim back to single year so update_lpjml works correctly.
        """
        if not hasattr(self.world, '_from_earth_data'):
            return

        from_earth = self.world._from_earth_data
        if hasattr(from_earth, 'time') and len(from_earth.time) > 1:
            # Keep only the last time step
            self.world._from_earth_data = from_earth.isel(time=[-1])

    def init_farmers(self, farmer_class, **kwargs):
        """Initialize farmers for cells with crops, sorted by harvest date."""
        farmers = []
        for cell in self.world.cells:
            has_crops = cell.from_earth.cftfrac.sum("band").isel(time=[-1]) > 0
            if not has_crops:
                continue
            farmer = farmer_class(cell=cell, model=self)
            farmers.append(farmer)
        farmers_sorted = sorted(farmers, key=lambda farmer: farmer.avg_hdate)
        self._farmers = farmers_sorted

        # Initialize neighbourhoods
        for farmer in farmers_sorted:
            farmer.init_neighbourhood()

        return farmers_sorted

    def update(self, t):
        """Update all countries and LPJmL for year ``t``."""
        self.update_countries(t)
        self.update_lpjml(t)
        # Collect outputs (if enabled in config)
        self.collect_outputs(t)

    def _preload_fao_data(self):
        """Pre-load FAO data for all countries before initialization.

        Must be called BEFORE init_countries() to ensure FAO data is available
        for parallelization and to avoid issues during country initialization.
        """
        import numpy as np
        from pycopanlpjml.model import _get_country_names

        country_values = self.world.country_code.values
        if hasattr(country_values, "compute"):
            country_values = country_values.compute()

        unique_codes = np.unique(country_values)
        country_names = _get_country_names()
        iso3_codes = [
            country_names[c]["code"]
            for c in unique_codes
            if c in country_names
        ]

        if not iso3_codes:
            return

        try:
            reference_year = self.config.coupled_config.start_year
        except AttributeError:
            reference_year = 2020

        Country.preload_fao_data(
            self.config.sim_path,
            iso3_codes,
            reference_year=reference_year,
        )
