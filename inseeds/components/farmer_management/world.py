"""The inseeds_farmer_mnagement.world class.

"""

# This file is part of pycopancore.
#
# Copyright (C) 2016-2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license

from . import documentation as doc


class World(doc.World):
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        super(World, self).__init__(**kwargs)

    def init_farmers(self, **kwargs):
        """Initialize farmers."""
        cells = self.init_cells(**kwargs)
        farmers = []

        for cell in cells:
            if cell.output.cftfrac.sum("band") == 0:
                continue

            farmer = self.model.Farmer(
                cell=cell, config=self.lpjml.config.coupled_config
            )
            farmers.append(farmer)

        farmers_sorted = sorted(farmers, key=lambda farmer: farmer.avg_hdate)
        for farmer in farmers_sorted:
            farmer.init_neighbourhood()

        self.update_output_table(self.lpjml.sim_year - 1, init=True)

        return farmers_sorted, cells

    @property
    def farmers(self):
        """Return the set of all farmers."""
        farmers = {
            farmer
            for farmer in self.individuals
            if farmer.__class__.__name__ == "Farmer"  # noqa
        }
        return farmers

    def update_farmers(self, t):
        farmers_sorted = sorted(self.farmers, key=lambda farmer: farmer.avg_hdate)
        for farmer in farmers_sorted:
            farmer.update_behaviour(t)

    def update(self, t):
        self.update_farmers(t)
        self.update_output_table(t)

        self.update_lpjml(t)
