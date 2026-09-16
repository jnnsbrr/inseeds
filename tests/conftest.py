import os
import pickle
import sys

import pytest

# Fix for Intel OneAPI causing pandas import hangs on HPC
# Set single-threaded Intel libraries to prevent hanging
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


@pytest.fixture(scope="session")
def test_path():
    """Fixture for the test path."""
    return os.path.dirname(os.path.abspath(__file__))


#: Cell-level outputs the CA farmer reads that the pickled output predates,
#: with a plausible constant in the variable's own unit. Only their presence is
#: under test; the numbers are not calibrated against anything.
CA_CELL_VARIABLES = {
    "litcover_agr": 0.35,  # fraction
    # Whole-cell leaching, consistent with the boundaries package measurement.
    "leaching": 2.0,  # gN/m2 of cell
    "nleaching_agr": 2.0,  # gN/m2 of cell (kept for backward compatibility)
    "runoff": 100.0,  # mm
    "irrig": 50.0,  # mm
    "nfert_agr": 10.0,  # gN/m2
}

#: Harvest per crop band the CA farmer prices, gC/m2.
CA_PFT_HARVESTC = 250.0


def _repair_band_coords(data):
    """Point each per-variable band coordinate at its own dimension.

    The pickled output was written when every band coordinate still carried the
    plain 'band' dimension, so 'band' claims 32 entries for cftfrac and 1 for
    harvestc at once. xarray only notices once a dimension recalculation, such
    as the isel that builds the cell views, forces the sizes to agree.
    """
    import xarray as xr

    for name in list(data.coords):
        coord = data[name]
        if (
            str(name).startswith("band (")
            and coord.dims == ("band",)
            and str(name) in data.sizes
        ):
            data._variables[name] = xr.Variable((str(name),), coord.values)
    return data


def _add_ca_variables(data):
    """Add the outputs the CA farmer reads and the pickled output lacks.

    The fixture was recorded for the tillage realisation, which reads a single
    aggregate harvest, while the CA farmer reads harvest per crop band plus the
    water and nitrogen outputs its scoring is built on.
    """
    import numpy as np
    import xarray as xr

    ncell = data.sizes["cell"]
    ntime = data.sizes.get("time", 1)

    for name, value in CA_CELL_VARIABLES.items():
        if name not in data.data_vars:
            data[name] = xr.DataArray(
                np.full((ncell, ntime), value, dtype=np.float32),
                dims=["cell", "time"],
                coords={"cell": data.coords["cell"], "time": data.coords["time"]},
            )

    if "pft_harvestc" not in data.data_vars and "band (cftfrac)" in data.coords:
        bands = data.coords["band (cftfrac)"].values
        data["pft_harvestc"] = xr.DataArray(
            np.full((ncell, len(bands), ntime), CA_PFT_HARVESTC, dtype=np.float32),
            dims=["cell", "band (pft_harvestc)", "time"],
            coords={
                "cell": data.coords["cell"],
                "band (pft_harvestc)": bands,
                "time": data.coords["time"],
            },
        )

    return data


def _patch_lpjml_for_testing(lpjml_obj, test_path):
    """Patch lpjml object with read_input/read_output that read from pickle
    files."""

    if hasattr(lpjml_obj, "config") and lpjml_obj.config is not None:
        from pycoupler.data import LPJmLInputType

        LPJmLInputType.load_config(lpjml_obj.config)

    def read_input():
        with open(f"{test_path}/data/lpjml_input.pkl", "rb") as inp:
            return pickle.load(inp)

    def read_output():
        import numpy as np
        import xarray as xr

        with open(f"{test_path}/data/lpjml_output.pkl", "rb") as out:
            data = _repair_band_coords(pickle.load(out))
        # Fill NaN in harvestc so farmer.get_from_earth does not raise
        if "harvestc" in data.data_vars:
            harvestc = data["harvestc"]
            vals = harvestc.values
            if np.any(np.isnan(vals)):
                harvestc.values[:] = np.nan_to_num(vals, nan=0.0)
        return _add_ca_variables(data)

    lpjml_obj.read_input = read_input
    lpjml_obj.read_output = read_output
    lpjml_obj.read_historic_output = read_output
    return lpjml_obj


@pytest.fixture(scope="session")
def lpjml_data(test_path):
    """Load LPJmL data once for all tests and patch for pickle-based
    testing."""

    with open(f"{test_path}/data/lpjml.pkl", "rb") as lpj:
        data = pickle.load(lpj)
    return _patch_lpjml_for_testing(data, test_path)


@pytest.fixture(scope="session")
def regions_model_instance(lpjml_data):
    """Create regions model instance once for all tests."""
    from inseeds.realisations.regenerative_tillage_regions import Model

    return Model(lpjml=lpjml_data)


@pytest.fixture(scope="session")
def regular_model_instance(lpjml_data):
    """Create regular model instance once for all tests."""
    from inseeds.realisations.regenerative_tillage import Model

    return Model(lpjml=lpjml_data)


@pytest.fixture(scope="session")
def run_regions_model_instance(regions_model_instance):
    """Run the regions model once and return it."""
    for year in regions_model_instance.lpjml.get_sim_years():
        regions_model_instance.update(year)
    return regions_model_instance


@pytest.fixture(scope="session")
def quick_model_instance(lpjml_data):
    """Create a model instance without running simulation - for quick tests."""
    from inseeds.realisations.regenerative_tillage_regions import Model

    return Model(lpjml=lpjml_data)


@pytest.fixture(scope="session")
def quick_regular_model_instance(lpjml_data):
    """Create a regular model instance without running simulation - for quick
    tests."""
    from inseeds.realisations.regenerative_tillage import Model

    return Model(lpjml=lpjml_data)


@pytest.fixture(scope="session")
def run_regular_model_instance(regular_model_instance):
    """Run the regular model once and return it."""
    for year in regular_model_instance.lpjml.get_sim_years():
        regular_model_instance.update(year)
    return regular_model_instance


@pytest.fixture(scope="session")
def single_step_model_instance(quick_model_instance):
    """Run just one simulation step for tests that need some simulation."""
    initial_year = list(quick_model_instance.lpjml.get_sim_years())[0]
    quick_model_instance.update(initial_year)
    return quick_model_instance


@pytest.fixture(scope="session")
def cached_output_table(run_regions_model_instance):
    """Pre-compute and cache the expensive output table."""
    return run_regions_model_instance.output_table.copy()


@pytest.fixture(scope="session")
def cached_regular_output_table(run_regular_model_instance):
    """Pre-compute and cache the expensive output table for regular model."""
    return run_regular_model_instance.output_table.copy()


@pytest.fixture(scope="session")
def cached_test_output_table(test_path):
    """Pre-load and cache the expected test output CSV."""
    import pandas as pd

    test_output = pd.read_csv(f"{test_path}/data/test_output_table.csv")
    return test_output.where(test_output.notna(), None)


@pytest.fixture(scope="session")
def minimal_model_instance(lpjml_data):
    """Create minimal model instance with minimal data for structure tests."""
    from inseeds.realisations.regenerative_tillage_regions import Model

    # Create model but don't initialize expensive components
    model = Model(lpjml=lpjml_data)
    return model


def build_ca_model(test_path, seed=None, output_transform=None):
    """Build a Conservation Agriculture model on the pickled test fixture.

    Kept as a plain function rather than only a fixture so a test that needs a
    private, freshly seeded model of its own (the golden trajectory) builds it
    the same way the shared fixture does.

    Parameters
    ----------
    test_path : str
        Directory holding the pickled LPJmL fixture and the sim path.
    seed : int, optional
        Seeds the global numpy RNG before the model is built. AFT assignment and
        the initial observation offsets are drawn there, so a model is only
        reproducible if the seed is set before construction, not after.
    output_transform : callable, optional
        Applied to the LPJmL output before the model sees it, for both the
        historic read that sets the baselines and the runtime read. A test that
        needs a fixture value at a different magnitude has to change both, or
        the baseline and the runtime value stop being comparable.

    Returns
    -------
    Model
        Conservation agriculture model, initialised but not stepped.
    """
    from pathlib import Path
    import xarray as xr
    import numpy as np
    from pycoupler.config import read_yaml, CoupledConfig
    from inseeds.realisations.conservation_agriculture import Model

    # Load fresh lpjml data (don't use shared session fixture)
    with open(f"{test_path}/data/lpjml.pkl", "rb") as lpj:
        lpjml_data = pickle.load(lpj)
    lpjml_data = _patch_lpjml_for_testing(lpjml_data, test_path)

    # Path to the CA config.yaml
    config_path = Path(__file__).parent.parent / "inseeds" / "realisations" / "conservation_agriculture" / "config.yaml"

    # Use test data directory for simulation
    sim_path = Path(test_path) / "data" / "ca_test_sim"
    sim_path.mkdir(parents=True, exist_ok=True)

    # Load the config using pycopanlpjml's proper config loading
    coupled_config = read_yaml(str(config_path), CoupledConfig)

    # Set config on this fixture's lpjml_data (not the shared one)
    lpjml_data.config.coupled_config = coupled_config
    lpjml_data.config.sim_path = sim_path

    # Add dummy coupled variables to lpjml input data
    original_read_input = lpjml_data.read_input

    def patched_read_input():
        data = original_read_input()
        ncell = data.sizes['cell']
        ntime = data.sizes.get('time', 1)
        # Match the shape of existing variables (cell, time)
        if 'cover_crop' not in data:
            data['cover_crop'] = xr.DataArray(
                np.zeros((ncell, ntime), dtype=np.int32),
                dims=['cell', 'time']
            )
        if 'residue_on_field' not in data:
            data['residue_on_field'] = xr.DataArray(
                np.ones((ncell, ntime), dtype=np.float32) * 0.5,
                dims=['cell', 'time']
            )
        return data

    lpjml_data.read_input = patched_read_input

    if output_transform is not None:
        original_read_output = lpjml_data.read_output

        def patched_read_output():
            return output_transform(original_read_output())

        lpjml_data.read_output = patched_read_output
        lpjml_data.read_historic_output = patched_read_output

    if seed is not None:
        np.random.seed(seed)

    # Create model - config is already set on lpjml_data.config
    return Model(lpjml=lpjml_data)


@pytest.fixture(scope="module")
def ca_model_builder(test_path):
    """Factory for a private CA model, for tests that must not share one."""

    def _build(seed=None, output_transform=None):
        return build_ca_model(
            test_path, seed=seed, output_transform=output_transform
        )

    return _build


@pytest.fixture(scope="module")
def ca_model_instance(test_path):
    """Create Conservation Agriculture model instance for CA farmer tests.

    This fixture creates its own lpjml_data copy to avoid polluting the shared
    session-scoped lpjml_data fixture used by other model tests.

    Uses module scope so all CA integration tests share the same model instance,
    avoiding repeated file operations that cause HDF locking issues.
    """
    return build_ca_model(test_path)


def pytest_configure(config):
    """Mark that we're running from pytest so pycopanlpjml uses test mode."""
    sys._called_from_test = True


def pytest_unconfigure(config):
    """Clean up test marker after pytest exits."""
    if hasattr(sys, "_called_from_test"):
        del sys._called_from_test
