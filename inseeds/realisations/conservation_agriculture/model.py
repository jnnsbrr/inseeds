from pycopancore.data_model.variable import Variable
from pycopancore.data_model.unit import Unit
from pycopancore.data_model.master_data_model.dimensions_and_units import (
    DimensionsAndUnits as DAU,
)

from inseeds.components import base
from inseeds.components import farming
from inseeds.components.farming import ConservationAgricultureFarmer
from inseeds.components.farming.ca_country import CACountry
from inseeds.components.farming.ca_world import CAWorld
from inseeds.components.farming.farmer import AFT
from inseeds.components import lpjml
from inseeds.components.exogenous import load_all as load_exogenous
from inseeds.components.farming.ca_management import (
    ManagementBundle,
    log_active_score_weights,
)
from inseeds.components.farming.ca_behaviour import BLOCKER_NAMES, DRIVER_NAMES
from inseeds.components.farming.farmer import NON_CROPS, avg_hdate
from inseeds.components.farming.ca_agroecology import (
    init_agroecological_clusters,
    cluster_management_performance,
)


# Custom unit for millions of dollars
MEGADOLLARS = Unit("megadollars", symbol="M$")
CONCENTRATION = Unit("mgN/l", symbol="mgN/l")
N_FLUX = Unit("gN/m2", symbol="gN/m2")

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
            "which cover crop is in the ground, as LPJmL needs it to pick the "
            "PFT: 0=none, 1=non-legume catch crop, 2=legume. Whether one is "
            "grown at all follows the bundle, and which one it is is settled "
            "with it at apply_bundle, so 1 against 2 is a decision about "
            "nitrogen and seed cost and not a property of the bundle",
        ),
        residue_on_field=Variable(
            "residue on field",
            "share of this year's residue left on the field (0-1), which is "
            "what LPJmL is told. Not the bundle dimension of the same name: "
            "that one is binary, and a farm reading below 1 here may be either "
            "an uncommitted farm at its own removal rate or a committed one in "
            "a year it left less. Read behaviour.residue_committed for the "
            "commitment itself",
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
        baseline_revenue=Variable(
            "baseline revenue",
            "historic average revenue frozen at sim start ($ for whole farm); "
            "the anchor delta_revenue is measured against",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        current_revenue=Variable(
            "current revenue",
            "this year's harvest valued at producer prices ($ for whole farm)",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        delta_revenue=Variable(
            "delta revenue",
            "revenue change from baseline ($ for whole farm); "
            "= current_revenue - baseline_revenue",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        delta_costs=Variable(
            "delta costs",
            "cost change from baseline bundle ($ for whole farm); positive "
            "when current bundle costs more than baseline",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        carbon_payment=Variable(
            "carbon payment",
            "payment for soil C sequestration above baseline ($ for whole farm)",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        practice_subsidy=Variable(
            "practice subsidy",
            "payment for holding CA practices ($ for whole farm)",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        nitrogen_charge=Variable(
            "nitrogen charge",
            "charge on nitrogen leaching above allowance ($ for whole farm)",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        raw_profit=Variable(
            "raw profit",
            "delta_revenue - delta_costs + transfers ($ for whole farm); the "
            "year's change in capital net of transition_cost and "
            "residue_sale_income",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        residue_opportunity_cost=Variable(
            "residue opportunity cost",
            "this year's straw priced at current revenue ($ for whole farm); "
            "the dynamic figure, against which baseline_residue_cost is frozen",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        baseline_residue_cost=Variable(
            "baseline residue cost",
            "straw value frozen on baseline revenue ($ for whole farm); what "
            "retention is charged and the cost-share subsidy is paid on",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        transition_cost=Variable(
            "transition cost",
            "one-off cost of this year's practice change ($ for whole farm); "
            "charged to capital outside raw_profit",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        residue_sale_income=Variable(
            "residue sale income",
            "cash from selling standing straw on a forced deselection ($ for "
            "whole farm); credited to capital outside raw_profit",
            unit=MEGADOLLARS,
            output_scale=1e-6,
        ),
        leaching=Variable(
            "leaching",
            "nitrogen lost below the root zone of agent land (gN/m2/yr); the "
            "quantity the model tracks and scores",
            unit=N_FLUX,
        ),
        leaching_concentration=Variable(
            "leaching concentration",
            "nitrate concentration of the leachate (mgN/l), for comparison "
            "against a nitrate standard; nothing decides on it, and it reads "
            "0.0 in a year too dry for a concentration to be defined",
            unit=CONCENTRATION,
        ),
        taxable_leaching=Variable(
            "taxable leaching",
            "nitrogen lost above the levy-free allowance (gN/m2/yr), the base "
            "the nitrogen charge is actually levied on; equals leaching when "
            "policy.levy_free_concentration is 0, and 0 for a farm draining "
            "cleaner than the standard",
            unit=N_FLUX,
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
            "behaviour.residue_committed": Variable(
                "residue commitment",
                "whether the farm undertook to leave whatever residue grows, "
                "as against leaving some because that is what it always did. "
                "The commitment is what carries the opportunity cost and draws "
                "the subsidy, and it is not recoverable from residue_on_field",
                datatype=bool,
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


class World(lpjml.World, CAWorld):
    """World entity type with global reference value computation."""

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
        self.init_agroecological_clusters(
            n_clusters=self.config.coupled_config.agroecological_clustering.n_clusters,
            k_range=self.config.coupled_config.agroecological_clustering.k_range,
        )

        # Initialize cells
        self.init_cells(cell_class=Cell)


        # Initialize farmers (needs reference_scales for baseline_trend computation)
        self.init_farmers(farmer_class=Farmer)
        # Compute static reference scales for performance scoring
        # Must be done AFTER farmers exist but BEFORE simulation starts
        self.world.compute_reference_scales()

        # Put the weights that actually reached the score on the record, so a
        # finished run can be attributed to a scenario
        log_active_score_weights(self.config.coupled_config)

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
        # A cell needs a material share of cropland to represent a farm, not
        # merely a non-zero one: a few hectares inside a half-degree cell is a
        # rounding artefact of the land-use input.
        min_crop_fraction = self.config.coupled_config.materiality.min_crop_fraction

        # Collect cells with crops and their harvest dates
        cells_with_hdate = []
        for cell in self.world.cells:
            has_crops = (
                cell.from_earth.cftfrac
                .drop_sel(band=NON_CROPS)
                .sum("band")
                .isel(time=[-1]) > min_crop_fraction
            )
            if not has_crops:
                continue
            hdate = avg_hdate(cell, self)
            cells_with_hdate.append((cell, hdate))

        # Sort by harvest date for deterministic ordering, with the cell index
        # breaking ties. Farmers are created in this order and draw their AFT
        # and their first observation offset as they go, so an order left to
        # object identity would give a different population from the same seed.
        cells_with_hdate.sort(key=lambda pair: (pair[1], int(pair[0].cell_index)))

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
        # Rescale before anyone decides, so every farmer in this year is scored
        # against the same ruler. The level ruler is the cross-farmer spread of
        # the rates, which lengthens as pools like soil carbon drift apart, so it
        # has to track the population rather than stay frozen at init.
        self.world.compute_reference_scales()

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
        world.statistic["countries_management_performance"], which gives every
        farmer the stats of its own country and the adjacent ones for blended
        social learning.
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
