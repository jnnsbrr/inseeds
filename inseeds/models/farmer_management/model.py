"""
Social sub-component of the InSEEDS model.
This does not inlude any communication to LPJmL via the pycoupler, but can be
regarded as a first step towards coupled runs.

Based on the exploit model, inluding novel decision-making dynamics
on tha basis of the Theory of Planned behaviour (TPB)

Conceptualization by Luana Schwarz, implementation based on Ronja Hotz'
Exploit model MOL CC SN, with adjustments by Luana Schwarz.

TODO: Go through the file and adjust all parts of the code marked with the TODO
flag. Pay attention to those variables and object written in capital letters.
These are placeholders and must be adjusted as needed. For further details see
also the model development tutorial.
"""

# This file is part of pycopancore.
#
# Copyright (C) 2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>

#
# TODO: import all other needed model components (adjust as needed):
#
import pycopancore.model_components.base as core
import pycopancore.model_components.lpjml as lpjml

import inseeds.components.farmer_management as farmer_management
import inseeds.components.base as base


class Farmer(farmer_management.Farmer):
    """Farmer entity type."""

    pass


class Cell(core.Cell, lpjml.Cell):
    """Cell entity type."""

    pass


# TODO: list all mixin classes needed:
class World(core.World, lpjml.World, farmer_management.World, base.World):
    """World entity type."""

    pass


class Model(core.Model, lpjml.Model, farmer_management.Component):
    """Class representing the whole model."""

    name = "InSEEDS Social"
    description = "Subcomponent of the InSEEDS model representing only social \
    dynamics and decision-making on the basis of the TPB"

    entity_types = [World, Cell, Farmer]
    """List of entity types used in the model"""
