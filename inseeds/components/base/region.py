import os
import sys
import pandas as pd

from . import Entity


class Region(Entity):
    pass

class Country(Region):
    pass

class WorldRegion(Region):
    pass