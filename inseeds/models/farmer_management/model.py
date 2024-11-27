"""
Social sub-component of the InSEEDS model.
This does not inlude any communication to LPJmL via the pycoupler, but can be
regarded as a first step towards coupled runs.

Based on the exploit model, inluding novel decision-making dynamics
on tha basis of the Theory of Planned behaviour (TPB)

Conceptualization by Luana Schwarz, implementation based on Ronja Hotz'
Exploit model MOL CC SN, with adjustments by Luana Schwarz.

"""

import pycopancore.model_components.base as core
import pycopanlpjml as lpjml

import inseeds.components.farmer_management as farmer_management
import inseeds.components.base as base


class Farmer(farmer_management.Farmer):
    """Farmer entity type."""

    pass


class Cell(lpjml.Cell, base.Cell):
    """Cell entity type."""

    pass


class World(lpjml.World, farmer_management.World):
    """World entity type."""

    pass


class Model(lpjml.Component, farmer_management.Component):
    """Class representing the whole model."""

    name = "InSEEDS Social"
    description = "Subcomponent of the InSEEDS model representing only social \
    dynamics and decision-making on the basis of the TPB"

    entity_types = [World, Cell, Farmer]
    """List of entity types used in the model"""

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
            country=self.lpjml.country,
            area=self.lpjml.terr_area,
        )

        # initialize cells
        self.init_cells()

        # initialize farmers
        self.init_farmers()

        self.world.write_output_table(
            self.lpjml.sim_year - 1,
            init=True,
            file_format=self.config.coupled_config.output_settings.file_format,
        )

    def update(self, t):
        self.update_farmers(t)
        self.world.write_output_table(
            t, file_format=self.config.coupled_config.output_settings.file_format
        )

        self.update_lpjml(t)
