"""Entity base class for inseeds."""

from pycopanlpjml.output import OutputDefinitionMixin


class Entity(OutputDefinitionMixin):
    """Base class for inseeds entities.

    Inherits OutputDefinitionMixin for output collection support.
    Model access is provided by pycopancore._Mixin.model property.
    """

    def update(self, t):
        """Update method called each timestep (override in subclasses)."""
        pass
