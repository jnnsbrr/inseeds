"""Model mixing class for inseeds_farmer_management
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

# import all needed entity type implementation classes:
from . import World, Farmer


class Component(doc.Component):
    """Model mixin class."""

    # list of entity types augmented by this component
    entity_types = [World, Farmer]

    def init_farmers(self, **kwargs):
        """Initialize farmers."""
        farmers = []

        for cell in self.world.cells:
            if cell.output.cftfrac.sum("band") == 0:
                continue

            farmer = Farmer(cell=cell, model=self)
            farmers.append(farmer)

        farmers_sorted = sorted(farmers, key=lambda farmer: farmer.avg_hdate)
        for farmer in farmers_sorted:
            farmer.init_neighbourhood()

        # self.world.farmers = set(farmers_sorted

    def update_farmers(self, t):
        farmers_sorted = sorted(self.world.farmers, key=lambda farmer: farmer.avg_hdate)
        for farmer in farmers_sorted:
            farmer.update_behaviour(t)
