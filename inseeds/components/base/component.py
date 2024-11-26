"""Model mixing class for inseeds base
"""

from . import World


class Component:
    """Model mixin class."""

    # mixins provided by this model component:
    entity_types = [World]
