"""Tests for exogenous data layer.

Tests cover:
- ExogenousSource base class
- Exogenous accessor
- Entity-level slicing (world, country, cell)
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from inseeds.components.exogenous.base import ExogenousSource
from inseeds.components.exogenous.accessor import Exogenous


class MockSource(ExogenousSource):
    """Mock source for testing."""
    
    granularity = "country"
    
    @property
    def name(self):
        return "mock"
    
    def ensure(self, sim_path, **kwargs):
        return sim_path / "input" / "mock.nc"


class MockCellSource(ExogenousSource):
    """Mock cell-level source for testing."""
    
    granularity = "cell"
    
    @property
    def name(self):
        return "mock_cell"
    
    def ensure(self, sim_path, **kwargs):
        return sim_path / "input" / "mock_cell.nc"


class TestExogenousSource:
    """Tests for ExogenousSource base class."""
    
    def test_slice_for_country_with_area_code(self):
        """Should slice by area_code dimension."""
        source = MockSource()
        ds = xr.Dataset({
            "value": (["area_code", "time"], np.array([[1, 2], [3, 4], [5, 6]])),
        }, coords={"area_code": ["NLD", "DEU", "BEL"], "time": [2020, 2021]})
        
        result = source.slice_for_country(ds, "NLD")
        assert "NLD" not in result.dims  # Selected, not dimension anymore
        assert result["value"].values.tolist() == [1, 2]
    
    def test_slice_for_country_without_area_code(self):
        """Should return dataset unchanged if no area_code dimension."""
        source = MockSource()
        ds = xr.Dataset({"value": (["time"], [1, 2, 3])})
        
        result = source.slice_for_country(ds, "NLD")
        assert result.identical(ds)
    
    def test_slice_for_cell_country_granularity(self):
        """Cell-level slice for country granularity should delegate to country."""
        source = MockSource()
        ds = xr.Dataset({
            "value": (["area_code"], [1, 2]),
        }, coords={"area_code": ["NLD", "DEU"]})
        
        result = source.slice_for_cell(ds, cell_idx=0, country_code="NLD")
        assert float(result["value"]) == 1
    
    def test_slice_for_cell_cell_granularity(self):
        """Cell-level slice for cell granularity should select by cell index."""
        source = MockCellSource()
        ds = xr.Dataset({
            "value": (["cell", "cft"], np.array([[1, 2], [3, 4], [5, 6]])),
        }, coords={"cell": [0, 1, 2], "cft": [0, 1]})
        
        result = source.slice_for_cell(ds, cell_idx=1)
        assert result["value"].values.tolist() == [3, 4]


class TestExogenousAccessor:
    """Tests for Exogenous accessor."""
    
    @pytest.fixture
    def mock_datasets(self):
        """Create mock datasets for testing."""
        return {
            "prices": xr.Dataset({
                "price": (["area_code", "time"], np.array([[100, 110], [200, 210]])),
            }, coords={"area_code": ["NLD", "DEU"], "time": [2020, 2021]}),
            "residue": xr.Dataset({
                "frac_burnt": (["cell", "cft"], np.array([[0.1, 0.2], [0.3, 0.4]])),
            }, coords={"cell": [0, 1], "cft": [0, 1]}),
        }
    
    @pytest.fixture
    def mock_sources(self):
        """Create mock sources for testing."""
        return {
            "prices": MockSource(),
            "residue": MockCellSource(),
        }
    
    def test_world_level_returns_full_dataset(self, mock_datasets, mock_sources):
        """World-level accessor should return full datasets."""
        world = MagicMock()
        world.exogenous = None  # No world_ref means this IS world
        
        exo = Exogenous(world, datasets=mock_datasets, sources=mock_sources)
        
        assert "prices" in exo.keys()
        assert "residue" in exo.keys()
        assert exo.prices.identical(mock_datasets["prices"])
        assert exo.residue.identical(mock_datasets["residue"])
    
    def test_country_level_slices_by_country(self, mock_datasets, mock_sources):
        """Country-level accessor should slice by country code."""
        world = MagicMock()
        world_exo = Exogenous(world, datasets=mock_datasets, sources=mock_sources)
        
        country = MagicMock()
        country.code = "NLD"
        # No grid attribute = not a cell
        del country.grid
        
        country_exo = Exogenous.for_entity(country, world_exo)
        
        prices = country_exo.prices
        assert float(prices["price"].isel(time=0)) == 100
    
    def test_cell_level_slices_by_cell(self, mock_datasets, mock_sources):
        """Cell-level accessor should slice by cell index."""
        world = MagicMock()
        world_exo = Exogenous(world, datasets=mock_datasets, sources=mock_sources)
        
        cell = MagicMock()
        cell.grid.cell.item.return_value = 1
        cell.country = MagicMock()
        cell.country.code = "DEU"
        
        cell_exo = Exogenous.for_entity(cell, world_exo)
        
        residue = cell_exo.residue
        assert residue["frac_burnt"].values.tolist() == [0.3, 0.4]
    
    def test_raises_for_unknown_dataset(self, mock_datasets, mock_sources):
        """Should raise AttributeError for unknown dataset names."""
        world = MagicMock()
        exo = Exogenous(world, datasets=mock_datasets, sources=mock_sources)
        
        with pytest.raises(AttributeError, match="No dataset 'unknown'"):
            _ = exo.unknown
    
    def test_keys_returns_dataset_names(self, mock_datasets, mock_sources):
        """keys() should return list of available dataset names."""
        world = MagicMock()
        exo = Exogenous(world, datasets=mock_datasets, sources=mock_sources)
        
        keys = exo.keys()
        assert set(keys) == {"prices", "residue"}
    
    def test_items_iterates_over_datasets(self, mock_datasets, mock_sources):
        """items() should iterate over (name, dataset) pairs."""
        world = MagicMock()
        exo = Exogenous(world, datasets=mock_datasets, sources=mock_sources)
        
        items = dict(exo.items())
        assert set(items.keys()) == {"prices", "residue"}
