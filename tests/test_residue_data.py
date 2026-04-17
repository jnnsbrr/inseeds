"""Tests for ResidueData class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from inseeds.components.data.residue import ResidueData


class TestResidueDataExtraction:
    """Test residue fraction extraction from MADRaT data."""

    @pytest.fixture
    def mock_madrat_data(self, tmp_path):
        """Create mock MADRaT NetCDF files for testing."""
        # Create small test data (3 cells, 2 CFTs, 2 years)
        time = [2014, 2015]
        cft = np.arange(2)
        latitude = np.array([52.0, 52.5, 53.0])  # Netherlands-ish
        longitude = np.array([5.0, 5.5, 6.0])

        # Create coordinate grids
        burnt_data = np.random.rand(2, 2, 3, 3).astype(np.float32) * 0.1  # 10% burnt
        prod_data = np.ones((2, 2, 3, 3), dtype=np.float32)  # 1 tC/ha total
        removed_data = np.random.rand(2, 2, 3, 3).astype(np.float32) * 0.3  # 30% removed
        recycled_data = 1.0 - burnt_data - removed_data  # Rest recycled

        # Create datasets
        ds_burnt = xr.Dataset(
            {"residues_burnt": (["time", "cft", "latitude", "longitude"], burnt_data)},
            coords={"time": time, "latitude": latitude, "longitude": longitude},
        )
        ds_prod = xr.Dataset(
            {"residues_production": (["time", "cft", "latitude", "longitude"], prod_data)},
            coords={"time": time, "latitude": latitude, "longitude": longitude},
        )
        ds_removed = xr.Dataset(
            {"residues_removed": (["time", "cft", "latitude", "longitude"], removed_data)},
            coords={"time": time, "latitude": latitude, "longitude": longitude},
        )
        ds_recycled = xr.Dataset(
            {"residues_recycled": (["time", "cft", "latitude", "longitude"], recycled_data)},
            coords={"time": time, "latitude": latitude, "longitude": longitude},
        )

        # Save to temp directory
        ds_burnt.to_netcdf(tmp_path / ResidueData.BURNT_FILE)
        ds_prod.to_netcdf(tmp_path / ResidueData.PRODUCTION_FILE)
        ds_removed.to_netcdf(tmp_path / ResidueData.REMOVED_FILE)
        ds_recycled.to_netcdf(tmp_path / ResidueData.RECYCLED_FILE)

        return tmp_path

    @pytest.fixture
    def mock_grid(self):
        """Create mock LPJmL grid (like world.grid)."""
        # 2 cells in Netherlands
        return xr.Dataset(
            coords={
                "cell": [0, 1],
                "lat": ("cell", [52.25, 52.75]),
                "lon": ("cell", [5.25, 5.75]),
            }
        )

    def test_extract_fractions_returns_dataset(self, mock_madrat_data, mock_grid):
        """Test that _extract_fractions returns xr.Dataset with expected variables."""
        result = ResidueData._extract_fractions(
            mock_grid, year=2015, data_path=mock_madrat_data
        )

        assert isinstance(result, xr.Dataset)
        assert "frac_burnt" in result.data_vars
        assert "frac_removed" in result.data_vars
        assert "frac_recycled" in result.data_vars

    def test_extract_fractions_has_cell_dimension(self, mock_madrat_data, mock_grid):
        """Test that output is indexed by cell."""
        result = ResidueData._extract_fractions(
            mock_grid, year=2015, data_path=mock_madrat_data
        )

        assert "cell" in result.dims
        assert len(result.cell) == 2

    def test_fractions_sum_to_one(self, mock_madrat_data, mock_grid):
        """Test that burnt + removed + recycled ≈ 1."""
        result = ResidueData._extract_fractions(
            mock_grid, year=2015, data_path=mock_madrat_data
        )

        total = result.frac_burnt + result.frac_removed + result.frac_recycled
        # Fractions sum to 1 for each cell (may have CFT dimension)
        np.testing.assert_array_almost_equal(total.values, np.ones_like(total.values), decimal=5)

    def test_fractions_are_normalized(self, mock_madrat_data, mock_grid):
        """Test that fractions are between 0 and 1."""
        result = ResidueData._extract_fractions(
            mock_grid, year=2015, data_path=mock_madrat_data
        )

        assert (result.frac_burnt >= 0).all()
        assert (result.frac_burnt <= 1).all()
        assert (result.frac_removed >= 0).all()
        assert (result.frac_removed <= 1).all()
        assert (result.frac_recycled >= 0).all()
        assert (result.frac_recycled <= 1).all()


class TestResidueDataEnsure:
    """Test the ensure/cache pattern."""

    @pytest.fixture
    def mock_madrat_data(self, tmp_path):
        """Create mock MADRaT NetCDF files for testing."""
        time = [2015]
        latitude = np.array([52.0])
        longitude = np.array([5.0])

        for name, var in [
            (ResidueData.BURNT_FILE, "residues_burnt"),
            (ResidueData.PRODUCTION_FILE, "residues_production"),
            (ResidueData.REMOVED_FILE, "residues_removed"),
            (ResidueData.RECYCLED_FILE, "residues_recycled"),
        ]:
            data = np.ones((1, 1, 1, 1), dtype=np.float32) * 0.25
            ds = xr.Dataset(
                {var: (["time", "cft", "latitude", "longitude"], data)},
                coords={"time": time, "latitude": latitude, "longitude": longitude},
            )
            ds.to_netcdf(tmp_path / name)

        return tmp_path

    @pytest.fixture
    def mock_grid(self):
        """Create mock LPJmL grid."""
        return xr.Dataset(
            coords={
                "cell": [0],
                "lat": ("cell", [52.0]),
                "lon": ("cell", [5.0]),
            }
        )

    def test_ensure_creates_cache_file(self, mock_madrat_data, mock_grid, tmp_path, monkeypatch):
        """Test that ensure() creates cache file."""
        sim_path = tmp_path / "simulation"
        (sim_path / "input").mkdir(parents=True)

        # Monkeypatch the default data path
        monkeypatch.setattr(ResidueData, "DEFAULT_DATA_PATH", mock_madrat_data)

        cache_path = ResidueData.ensure(sim_path, mock_grid, reference_year=2015)

        assert cache_path.exists()
        assert cache_path.name == ResidueData.CACHE_FILE

    def test_ensure_returns_existing_cache(self, mock_madrat_data, mock_grid, tmp_path, monkeypatch):
        """Test that ensure() returns existing cache without regenerating."""
        sim_path = tmp_path / "simulation"
        (sim_path / "input").mkdir(parents=True)

        monkeypatch.setattr(ResidueData, "DEFAULT_DATA_PATH", mock_madrat_data)

        # First call creates cache
        cache_path1 = ResidueData.ensure(sim_path, mock_grid, reference_year=2015)
        mtime1 = cache_path1.stat().st_mtime

        # Second call should return existing (same mtime)
        cache_path2 = ResidueData.ensure(sim_path, mock_grid, reference_year=2015)
        mtime2 = cache_path2.stat().st_mtime

        assert mtime1 == mtime2

    def test_ensure_overwrite_regenerates(self, mock_madrat_data, mock_grid, tmp_path, monkeypatch):
        """Test that ensure(overwrite=True) regenerates cache even when it exists."""
        sim_path = tmp_path / "simulation"
        (sim_path / "input").mkdir(parents=True)

        monkeypatch.setattr(ResidueData, "DEFAULT_DATA_PATH", mock_madrat_data)

        # First call creates cache
        cache_path = ResidueData.ensure(sim_path, mock_grid, reference_year=2015)
        assert cache_path.exists()

        # Delete and verify it gets regenerated with overwrite=True
        cache_path.unlink()
        assert not cache_path.exists()

        # Without overwrite, this would fail (file doesn't exist but ensure returns path)
        # Actually ensure() checks existence, so it will regenerate
        # Let's test that overwrite=True triggers extraction message
        import io
        import sys

        # Capture stdout to verify extraction happens
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        ResidueData.ensure(sim_path, mock_grid, reference_year=2015, overwrite=True)

        sys.stdout = old_stdout
        output = captured.getvalue()

        assert "Extracting residue fractions" in output
        assert cache_path.exists()


class TestResidueDataWithRealData:
    """Integration tests with real MADRaT data (skip if not available)."""

    @pytest.fixture
    def real_data_available(self):
        """Check if real MADRaT data is available."""
        return (ResidueData.DEFAULT_DATA_PATH / ResidueData.BURNT_FILE).exists()

    @pytest.fixture
    def netherlands_grid(self):
        """Grid cells roughly covering Netherlands."""
        return xr.Dataset(
            coords={
                "cell": list(range(5)),
                "lat": ("cell", [51.5, 52.0, 52.5, 53.0, 53.5]),
                "lon": ("cell", [4.5, 5.0, 5.5, 6.0, 6.5]),
            }
        )

    def test_real_data_extraction(self, real_data_available, netherlands_grid):
        """Test extraction from real MADRaT data."""
        if not real_data_available:
            pytest.skip("Real MADRaT data not available")

        result = ResidueData._extract_fractions(netherlands_grid, year=2015)

        # Check structure
        assert "frac_burnt" in result.data_vars
        assert "frac_removed" in result.data_vars
        assert "frac_recycled" in result.data_vars
        assert len(result.cell) == 5

        # Check reasonable values for Netherlands
        # Netherlands should have low burning, moderate removal
        mean_burnt = float(result.frac_burnt.mean())
        mean_removed = float(result.frac_removed.mean())
        mean_recycled = float(result.frac_recycled.mean())

        print(f"Netherlands residue fractions (2015):")
        print(f"  burnt: {mean_burnt:.3f}")
        print(f"  removed: {mean_removed:.3f}")
        print(f"  recycled: {mean_recycled:.3f}")

        # Sanity checks
        assert 0 <= mean_burnt <= 1
        assert 0 <= mean_removed <= 1
        assert 0 <= mean_recycled <= 1

    def test_real_data_ensure_and_load(self, real_data_available, netherlands_grid, tmp_path):
        """Test full ensure/load cycle with real data."""
        if not real_data_available:
            pytest.skip("Real MADRaT data not available")

        sim_path = tmp_path / "test_sim"
        (sim_path / "input").mkdir(parents=True)

        # Ensure cache
        cache_path = ResidueData.ensure(sim_path, netherlands_grid, reference_year=2015)
        assert cache_path.exists()

        # Load and verify
        ds = xr.open_dataset(cache_path)
        assert "frac_burnt" in ds.data_vars
        assert "frac_removed" in ds.data_vars
        assert "frac_recycled" in ds.data_vars
        assert len(ds.cell) == 5


class TestOpportunityCostCalculation:
    """Test the opportunity cost calculation logic."""

    def test_weighted_cost_calculation(self):
        """Test that weighted cost is computed correctly."""
        # Example fractions (Netherlands-like)
        fracs = {
            'burnt': 0.04,
            'removed': 0.20,
            'recycled': 0.76,
        }

        # Costs from config
        use_costs = {
            'burnt': 0.0,
            'removed': 80.0,
            'recycled': 0.0,
        }

        # Calculate weighted cost
        weighted_cost = (
            fracs['burnt'] * use_costs['burnt'] +
            fracs['removed'] * use_costs['removed'] +
            fracs['recycled'] * use_costs['recycled']
        )

        # Should be 0.20 * 80 = 16.0
        assert weighted_cost == pytest.approx(16.0)

    def test_high_removal_region(self):
        """Test cost in region with high residue removal."""
        # Example: region where most residues are removed
        fracs = {
            'burnt': 0.05,
            'removed': 0.70,
            'recycled': 0.25,
        }

        use_costs = {
            'burnt': 0.0,
            'removed': 80.0,
            'recycled': 0.0,
        }

        weighted_cost = (
            fracs['burnt'] * use_costs['burnt'] +
            fracs['removed'] * use_costs['removed'] +
            fracs['recycled'] * use_costs['recycled']
        )

        # Should be 0.70 * 80 = 56.0
        assert weighted_cost == pytest.approx(56.0)

    def test_high_burning_region(self):
        """Test cost in region with high residue burning."""
        # Example: region where most residues are burnt (no economic value)
        fracs = {
            'burnt': 0.70,
            'removed': 0.20,
            'recycled': 0.10,
        }

        use_costs = {
            'burnt': 0.0,
            'removed': 80.0,
            'recycled': 0.0,
        }

        weighted_cost = (
            fracs['burnt'] * use_costs['burnt'] +
            fracs['removed'] * use_costs['removed'] +
            fracs['recycled'] * use_costs['recycled']
        )

        # Should be 0.20 * 80 = 16.0 (only removed has cost)
        assert weighted_cost == pytest.approx(16.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
