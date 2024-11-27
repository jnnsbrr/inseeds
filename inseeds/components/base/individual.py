import os
import sys

from . import Entity


class Individual(Entity):
    """Define properties.
    Inherits from I.World as the interface with all necessary variables
    and parameters.
    """

    def __init__(self, **kwargs):
        """Initialize an instance of World."""
        super().__init__(**kwargs)
