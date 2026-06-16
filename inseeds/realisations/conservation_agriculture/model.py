from pycopancore.data_model.variable import Variable
from pycopancore.data_model.unit import Unit
from pycopancore.data_model.master_data_model.dimensions_and_units import (
    DimensionsAndUnits as DAU,
)

import xarray as xr

from inseeds.components import base
from inseeds.components import farming
from inseeds.components.farming import ConservationAgricultureFarmer
from inseeds.components.farming.ca_country import CACountry
from inseeds.components.farming.farmer import AFT
from inseeds.components import lpjml
from inseeds.components.exogenous import load_all as load_exogenous
from inseeds.components.farming.ca_management import ManagementBundle
from inseeds.components.farming.ca_behaviour import BLOCKER_NAMES, DRIVER_NAMES
from inseeds.components.farming.farmer import NON_CROPS, avg_hdate
from inseeds.components.farming.ca_agroecology import (
    init_agroecological_clusters,
    cluster_management_performance,
)


# Custom unit for millions of dollars
MEGADOLLARS = Unit("megadollars", symbol="M$")

# Mapping from bundle ID (0-7) to display name
BUNDLE_ID_TO_NAME = {
    bundle.id: bundle.label for bundle in ManagementBundle
}
# Add -1 for "no proposed bundle"
BUNDLE_ID_TO_NAME[-1] = ""

# Mapping from AFT ID to name (derived from AFT enum)
AFT_NAMES = {aft.value: aft.name for aft in AFT}


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
        litter_cover=Variable(
            "litter cover",
            "fractional soil cover from crop residues (0-1, CA threshold is 0.3)",
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
            "farm capital stock (total, scaled by farm size)",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        farm_size=Variable(
            "farm size",
            "farm size in hectares (sum of cftfrac * area)",
            unit=DAU.ha,
        ),
        # TPB decision model outputs (accessed via behaviour.X)
        **{
            "behaviour.practice_bundle_id": Variable(
                "practice bundle",
                "current bundle ID (0-7), settable for Dask sync",
            ),
            "behaviour.proposed_bundle_id": Variable(
                "proposed bundle",
                "bundle being evaluated (-1 if none)",
            ),
            "behaviour.tpb": Variable(
                "TPB score",
                "TPB intention score for proposed bundle",
            ),
            "behaviour.attitude": Variable(
                "attitude",
                "attitude component of TPB",
            ),
            "behaviour.social_norm": Variable(
                "social norm",
                "social norm component of TPB",
            ),
            "behaviour.pbc": Variable(
                "PBC",
                "perceived behavioral control component of TPB",
            ),
            "behaviour.transition_blocker": Variable(
                "transition blocker",
                "primary reason for blocked transition (0-13)",
            ),
            "behaviour.transition_driver": Variable(
                "transition driver",
                "primary reason for successful transition (0-9)",
            ),
        },
    )

    # Label mappings for categorical output variables
    # Maps variable name -> {numeric_code: human_readable_label}
    # Used by output.py to populate the 'label' column in CSV/Parquet
    output_label_mappings = {
        "aft_id": AFT_NAMES,
        "behaviour.practice_bundle_id": BUNDLE_ID_TO_NAME,
        "behaviour.proposed_bundle_id": BUNDLE_ID_TO_NAME,
        "behaviour.transition_blocker": BLOCKER_NAMES,
        "behaviour.transition_driver": DRIVER_NAMES,
    }


class Cell(lpjml.Cell, farming.Cell):
    """Cell entity type."""

    pass


class Country(CACountry, lpjml.Country, base.Country):
    """Country entity type with FAO data for CA capital initialization.
    
    Note: CACountry must come first in MRO so its update() method
    (with compute_management_performance) is called instead of the
    base Region.update().
    """

    output_variables = base.Output(
        agroecological_cluster=Variable(
            "agroecological cluster",
            "cluster ID based on climate similarity (-1 if unassigned)",
        ),
    )


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
        super().__init__(**kwargs)

        if not hasattr(self, "lpjml") or self.lpjml is None:
            raise ValueError("lpjml must be initialized in the parent class.")

        # Convert LPJmL numeric country codes to ISO alpha-3 codes
        # This is needed for proper country identification in outputs
        if hasattr(self.lpjml, "code_to_name"):
            self.lpjml.code_to_name(to_iso_alpha_3=True)

        self.world = World(
            model=self,
            input=self.lpjml.read_input(),
            output=self.lpjml.read_historic_output(),
            grid=self.lpjml.grid,
            country_code=self.lpjml.country,
            area=self.lpjml.terr_area,
        )

        # Initialize exogenous data from FAO and MADRaT
        self.load_exogenous_data()

        # Initialize countries (social systems)
        self.init_countries(country_class=Country)
        
        # Initialize agroecological clusters for tele-coupled social learning
        self.init_agroecological_clusters()

        # Initialize cells and farmers
        self.init_cells(cell_class=Cell)
        self.init_farmers(farmer_class=Farmer)

        # Cut off historical data (from_earth) to only the most recent year
        self.cutoff_historical_data()

    def cutoff_historical_data(self):
        """Cut off historical output to only the most recent year.

        During initialization, we pass multi-year historic output to give
        farmers initial history for trend computation. After initialization,
        we cut off the historical data to single year so update_lpjml works
        correctly.
        """
        if not hasattr(self.world, '_from_earth_data'):
            return

        from_earth = self.world._from_earth_data
        if hasattr(from_earth, 'time') and len(from_earth.time) > 1:
            # Keep only the last time step
            self.world._from_earth_data = from_earth.isel(time=[-1])
            # Refresh cell views to point to the new (cut-off) data
            self.refresh_cell_views()

    def init_farmers(self, farmer_class, **kwargs):
        """Initialize farmers for cells with crops, sorted by harvest date."""
        # Collect cells with crops and their harvest dates
        cells_with_hdate = []
        for cell in self.world.cells:
            has_crops = (
                cell.from_earth.cftfrac
                .drop_sel(band=NON_CROPS)
                .sum("band")
                .isel(time=[-1]) > 0
            )
            if not has_crops:
                continue
            hdate = avg_hdate(cell, self)
            cells_with_hdate.append((cell, hdate))

        # Sort by harvest date for deterministic ordering
        cells_with_hdate.sort(key=lambda x: x[1])

        # Create farmers
        farmers = []
        for cell, _ in cells_with_hdate:
            farmers.append(farmer_class(cell=cell, model=self))

        self._farmers = farmers

        # Initialize neighbourhoods (ordering preserved from avg harvest date)
        for farmer in farmers:
            farmer.init_neighbourhood()

        return farmers

    def update(self, t):
        """Update all countries and LPJmL for year ``t``."""
        self.update_countries(t)

        # Aggregate country stats for cross-border social learning
        # This runs AFTER all country updates, so stats reflect current year
        # and are available for NEXT year's social learning decisions
        self.update_cluster_management_performance()
        self.update_countries_management_performance()

        self.update_lpjml(t)
        # Collect outputs (if enabled in config)
        self.collect_outputs(t)

    def update_cluster_management_performance(self):
        """Aggregate country-level stats into agroecological cluster stats.

        Called after all country updates complete. Collects
        management_performance from each country's statistic and aggregates by
        cluster. Results stored in world.statistic["cluster_management_performance"]
        for tele-coupled social learning.
        """
        cluster_management_performance(self.world)

    def update_countries_management_performance(self):
        """Collect all countries' management_performance for cross-border learning.

        Stores a dict mapping country_code -> management_performance in
        world.statistic["countries_management_performance"]. This allows
        workers in parallel mode to access neighbouring countries' stats
        for blended social learning (own country + adjacent countries).
        """
        countries_stats = {}
        for country in self.world.countries:
            store = country.statistic.get("management_performance")
            if store is not None:
                countries_stats[country.code] = store

        self.world.statistic.set("countries_management_performance", countries_stats)

    def init_agroecological_clusters(self, n_clusters=None, k_range=(5, 10)):
        """Initialize country-level agroecological clusters for tele-coupled
        social learning across countries with similar agroecological conditions.

        Clusters countries by agroecological similarity (mean + seasonal amplitude of
        temperature, precipitation, and PET). Countries in the same cluster can
        share social learning information.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. If None, automatically determined via elbow method.
        k_range : tuple
            (min_k, max_k) range for elbow search when n_clusters is None.
        """
        if hasattr(self, "countries"):
            countries = list(self.countries)
        else:
            countries = list(self.world.countries)

        init_agroecological_clusters(self.world, countries, n_clusters, k_range)

    def load_exogenous_data(self):
        """Load all exogenous data sources (FAO, MADRaT).

        Stores unified accessor on world.exogenous for access at all levels.
        Must be called BEFORE init_countries() and init_farmers().
        """
        import numpy as np

        # Get ISO3 country codes
        country_values = self.world.country_code.values
        iso3_codes = np.unique(country_values)

        # Load all exogenous data sources
        self.world.exogenous = load_exogenous(
            sim_path=self.config.sim_path,
            grid=self.lpjml.grid,
            start_year=self.config.start_coupling,
            country_codes=iso3_codes,
            entity=self.world,
        )
