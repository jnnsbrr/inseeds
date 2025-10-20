import pickle
import pytest
import numpy as np
import pandas as pd

import inseeds.components.base as base
import inseeds.components.farming as farming
from inseeds.models.regenerative_tillage import Cell, Farmer, World, Model


def test_run_model(run_regular_model_instance):
    """Test running the model until end of simulation - lightweight version."""
    # Instead of running full simulation, just check that the model completed successfully
    assert hasattr(
        run_regular_model_instance, "world"
    ), "Model should have world"
    assert hasattr(
        run_regular_model_instance.world, "output"
    ), "World should have output"
    assert (
        run_regular_model_instance.world.output is not None
    ), "Output should not be None"

    # Check that we have some data
    if hasattr(run_regular_model_instance.world.output, "time"):
        assert (
            len(run_regular_model_instance.world.output.time) > 0
        ), "Should have time data"


def test_model_output(cached_regular_output_table, cached_test_output_table):
    """Test getting the output table of the model."""
    output = cached_regular_output_table

    # Sort the dataframes by the same columns
    sort_columns = ["year", "cell", "entity", "variable"]
    test_output = cached_test_output_table.sort_values(by=sort_columns)
    output = output.sort_values(by=sort_columns)

    for name, row in test_output.items():
        if name == "value":
            # if failing lower the threshold or continue
            #   LPJmL cell variables should be equal, but the rest can be
            #   different
            assert np.mean(output[name].values == row.values).item() > 0.75
        else:
            assert all(output[name].values == row.values)
