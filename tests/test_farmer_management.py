import pickle
import pytest

import inseeds.components.base as base
import inseeds.components.farming as farming
from inseeds.models.regenerative_tillage import Cell, Farmer, World, Model


def test_run_model(test_path):
    """Test the LPJmLCoupler class."""

    with open(f"{test_path}/data/lpjml.pkl", "rb") as lpj:
        lpjml = pickle.load(lpj)

    model = Model(lpjml=lpjml, test_path=test_path)

    for year in model.lpjml.get_sim_years():
        model.update(year)

    last_year = (
        model.world.output.time.values[0].astype("datetime64[Y]").astype(int).item()
        + 1970
    )

    # last year set to 2030 in test data set
    assert last_year == 2030
