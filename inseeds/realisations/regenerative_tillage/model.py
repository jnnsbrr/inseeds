from pycopancore.data_model.variable import Variable
from pycopancore.data_model.master_data_model.dimensions_and_units import (
    DimensionsAndUnits as DAU,
)

from inseeds.components import base
from inseeds.components import farming
from inseeds.components.farming import TillageFarmer
from inseeds.components import lpjml


class Farmer(TillageFarmer):
    """Farmer entity type.
    
    Overrides cell properties with simple direct access for performance
    (matches old inseeds behavior).
    """

    @property
    def cell_cropyield(self):
        """Return the average crop yield of the cell (simple, fast)."""
        val = self.cell.from_earth.harvestc.values.mean()
        return max(val, 1e-3)

    @property
    def cell_soilc(self):
        """Return the average soil carbon of the cell (simple, fast)."""
        val = self.cell.from_earth.soilc_agr_layer.values[0].item()
        return max(val, 1e-3)

    @property
    def cell_avg_hdate(self):
        """Return the average harvest date of the cell (simple, fast)."""
        import numpy as np
        cftfrac = self.cell.from_earth.cftfrac
        hdate = self.cell.from_earth.hdate
        if cftfrac.sum().item() == 0:
            return 365.0
        return np.average(hdate.values, weights=cftfrac.values)

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
        tillage=Variable(
            "agent tillage behaviour",
            "conventional=1, conservation=0",
            datatype=bool,
        ),
        pbc=Variable(
            "perceived behavioural control",
            "own appraisal of how much efficacy agent posesses",
        ),
        tpb=Variable(
            "theory of planned behaviour",
            "attitude, subjective norm, perceived behavioural control",
        ),
        social_norm=Variable(
            "social norm",
            "social norm based on observation of own and\
                                 neighboring land",
        ),
        attitude=Variable(
            "attitude",
            "farmer attitude based on observation of yield and soilC of\
                             own land and neighboring land",
        ),
        attitude_own_land=Variable(
            "attitude towards own land",
            "attitude based on observation of yield and soilC\
                                      of own land",
        ),
        attitude_social_learning=Variable(
            "attitude based on social learning",
            "attitude based on observation of yield and\
                                             soilC of neighboring land",
        ),
    )


class Cell(lpjml.Cell, farming.Cell):
    """Cell entity type."""

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
            output=self.lpjml.read_historic_output().isel(time=[-1]),
            grid=self.lpjml.grid,
            # country_code is the array of country codes
            country_code=self.lpjml.country,
            area=self.lpjml.terr_area,
        )

        # initialize cells
        self.init_cells(cell_class=Cell)

        # initialize farmers
        self.init_farmers(farmer_class=Farmer)

    def init_farmers(self, farmer_class, **kwargs):
        """Initialize farmers for cells with crops, sorted by harvest date."""
        farmers = []
        for cell in self.world.cells:
            has_crops = cell.from_earth.cftfrac.sum("band") > 0
            if not has_crops:
                continue
            farmer = farmer_class(cell=cell, model=self)
            farmers.append(farmer)
        farmers_sorted = sorted(farmers, key=lambda farmer: farmer.avg_hdate)
        for farmer in farmers_sorted:
            farmer.init_neighbourhood()
        self._farmers = farmers_sorted
        return farmers_sorted

    def update(self, t):
        # Cache cell-level values before updating farmers (avoids repeated xarray ops)
        self._cache_cells()
        # Update farmers directly using cached list (avoids expensive world.farmers property)
        for farmer in self._farmers:
            farmer.update(t)
        self.update_lpjml(t)
        # Collect outputs (if enabled in config)
        self.collect_outputs(t)

    def _cache_cells(self):
        """Cache computed values on cells to avoid repeated xarray operations.
        
        Only caches cells that have farmers (via _farmers list), not all world cells.
        """
        from inseeds.components.farming.farmer import cache_yearly
        # Use set to get unique cells from farmers (avoids iterating all 67k cells)
        cached_cells = set()
        for farmer in self._farmers:
            cell = farmer.cell
            if cell not in cached_cells:
                cache_yearly(cell, self)
                cached_cells.add(cell)
