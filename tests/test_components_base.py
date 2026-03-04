"""Unit tests for base components (Entity, etc.)."""

import pytest

from inseeds.components.base import Entity


class TestEntity:
    """Tests for the base Entity class."""

    def test_entity_update_noop(self):
        """Entity.update(t) should be a no-op (does not raise)."""
        entity = Entity()
        entity.update(2020)  # Should not raise

    def test_entity_inherits_output_mixin(self):
        """Entity should inherit from OutputDefinitionMixin."""
        from pycopanlpjml.output import OutputDefinitionMixin
        assert issubclass(Entity, OutputDefinitionMixin)
