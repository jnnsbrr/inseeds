import os
import pytest
import sys
import pickle

# Fix for Intel OneAPI causing pandas import hangs on HPC
# Set single-threaded Intel libraries to prevent hanging
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


@pytest.fixture(scope="session")
def test_path():
    """Fixture for the test path."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def lpjml_data(test_path):
    """Load LPJmL data once for all tests."""
    with open(f"{test_path}/data/lpjml.pkl", "rb") as lpj:
        return pickle.load(lpj)


@pytest.fixture(scope="session")
def regions_model_instance(lpjml_data, test_path):
    """Create regions model instance once for all tests."""
    from inseeds.models.regenerative_tillage_regions import Model

    return Model(lpjml=lpjml_data, test_path=test_path)


@pytest.fixture(scope="session")
def regular_model_instance(lpjml_data, test_path):
    """Create regular model instance once for all tests."""
    from inseeds.models.regenerative_tillage import Model

    return Model(lpjml=lpjml_data, test_path=test_path)


@pytest.fixture(scope="session")
def run_regions_model_instance(regions_model_instance):
    """Run the regions model once and return it."""
    for year in regions_model_instance.lpjml.get_sim_years():
        regions_model_instance.update(year)
    return regions_model_instance


@pytest.fixture(scope="session")
def quick_model_instance(lpjml_data, test_path):
    """Create a model instance without running simulation - for quick tests."""
    from inseeds.models.regenerative_tillage_regions import Model

    return Model(lpjml=lpjml_data, test_path=test_path)


@pytest.fixture(scope="session")
def quick_regular_model_instance(lpjml_data, test_path):
    """Create a regular model instance without running simulation - for quick tests."""
    from inseeds.models.regenerative_tillage import Model

    return Model(lpjml=lpjml_data, test_path=test_path)


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
def minimal_model_instance(lpjml_data, test_path):
    """Create minimal model instance with minimal data for structure tests."""
    from inseeds.models.regenerative_tillage_regions import Model

    # Create model but don't initialize expensive components
    model = Model(lpjml=lpjml_data, test_path=test_path)
    return model


def pytest_configure(config):
    sys._called_from_test = True


def pytest_unconfigure(config):
    del sys._called_from_test
