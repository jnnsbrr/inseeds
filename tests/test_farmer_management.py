import os
import pytest
import pickle

from inseeds.models.farmer_management import model as M


def test_run_model(test_path):
    """Test the LPJmLCoupler class."""

    with open(
        f'{test_path}/data/lpjml.pkl',
        'rb'
    ) as lpj:
        lpjml = pickle.load(lpj)

    with open(
        f'{test_path}/data/lpjml_input.pkl',
        'rb'
    ) as inp:
        lpjml_input = pickle.load(inp)

    with open(
        f'{test_path}/data/lpjml_output.pkl',
        'rb'
    ) as out:
        lpjml_output = pickle.load(out)


    # initialize (LPJmL) world
    world = M.World(
        model=M,
        lpjml=lpjml,
        input = lpjml_input,
        output = lpjml_output,
    )

    # initialize (cells and) individuals
    farmers, cells = world.init_individuals()

    for year in world.lpjml.get_sim_years():
        world.update(year)


    last_year =  world.output.time.values[0].astype(
        'datetime64[Y]'
    ).astype(int).item() + 1970

    # last year set to 2030 in test data set
    assert last_year == 2030