"""Soil carbon to a configured depth is the sum of the layers that reach it.

Depth 200 is the top layer and has to be the band-0 read the model made
before a depth existed, so a default run changes no number.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from inseeds.components.farming.farmer import (
    Farmer,
    GC_M2_TO_TCO2_HA,
    SOIL_CARBON_LAYER_BOUNDS_MM,
    agricultural_soilc,
    get_cell_var,
    soil_carbon_depth_mm,
    soilc,
    validate_soil_carbon_depth,
)


CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "inseeds",
    "realisations",
    "conservation_agriculture",
    "config.yaml",
)

# Exact in float32, so a sum of them is exact in float64 too.
LAYERS = np.array(
    [
        [1000.5, 1100.0, 900.25],
        [200.25, 180.5, 220.0],
        [50.125, 40.25, 60.5],
        [10.5, 12.0, 8.75],
        [1.25, 2.5, 0.5],
    ],
    dtype=np.float32,
)


def _cell(layers=LAYERS):
    data = xr.Dataset(
        {
            "soilc_agr_layer": (
                ("band", "time"),
                layers,
                {"units": "gC/m2"},
            )
        },
        coords={
            "band": np.arange(layers.shape[0]),
            "time": np.arange(layers.shape[1]),
        },
    )
    return SimpleNamespace(from_earth=data)


def _farmer(cell, depth_mm=200):
    farmer = SimpleNamespace(
        cell=cell,
        model=SimpleNamespace(
            config=SimpleNamespace(
                coupled_config=SimpleNamespace(soil_carbon_depth=depth_mm)
            )
        ),
    )
    farmer.get_from_earth = lambda *args, **kwargs: Farmer.get_from_earth(
        farmer, *args, **kwargs
    )
    farmer._cached = lambda name: Farmer._cached(farmer, name)
    return farmer


def _read_cell_soilc(farmer):
    """Call the property body. The mixin hides it from class access."""
    return Farmer.__dict__["cell_soilc"].fget(farmer)


def _band0_via_get_from_earth(farmer, time_idx):
    """The read cell_soilc, the baseline and the history used to make."""
    return Farmer.get_from_earth(
        farmer, "soilc_agr_layer", as_scalar=True, band=0, time_idx=time_idx
    )


def _band0_via_soilc(cell):
    """The read ``soilc(cell)`` used to make."""
    data = get_cell_var(cell, "soilc_agr_layer", band=0)
    values = data.values
    return (
        float(values.item())
        if hasattr(values, "item")
        else float(np.nanmean(values))
    )


def test_each_depth_is_the_sum_of_the_layers_above_it():
    cell = _cell()
    for depth in SOIL_CARBON_LAYER_BOUNDS_MM:
        n_layers = SOIL_CARBON_LAYER_BOUNDS_MM.index(depth) + 1
        farmer = _farmer(cell, depth)
        for time_idx in range(LAYERS.shape[1]):
            expected = sum(float(LAYERS[i, time_idx]) for i in range(n_layers))
            assert agricultural_soilc(cell, depth, time_idx=time_idx) == expected
        # The latest year is what the farmer's soilc property reads.
        assert _read_cell_soilc(farmer) == agricultural_soilc(
            cell, depth, time_idx=-1, farmer=farmer
        )


def test_depth_200_equals_band0_exactly():
    cell = _cell()
    farmer = _farmer(cell, 200)
    assert soilc(cell) == _band0_via_soilc(cell)
    assert soilc(cell, 200) == soilc(cell)
    for time_idx in range(LAYERS.shape[1]):
        via_helper = agricultural_soilc(
            cell, 200, time_idx=time_idx, farmer=farmer
        )
        via_old = _band0_via_get_from_earth(farmer, time_idx)
        assert via_helper == via_old
        assert via_helper == float(LAYERS[0, time_idx])


def test_depth_200_farmer_paths_match_the_old_reads():
    """Baseline, accumulation and the payment are functions of that stock."""
    cell = _cell()
    farmer = _farmer(cell, 200)
    n_years = LAYERS.shape[1]
    old_stocks = [_band0_via_get_from_earth(farmer, i) for i in range(n_years)]
    new_stocks = [
        agricultural_soilc(cell, 200, time_idx=i, farmer=farmer)
        for i in range(n_years)
    ]
    assert new_stocks == old_stocks

    # The latest year is what Farmer.cell_soilc reads, and what the soilc
    # output is copied from.
    assert _read_cell_soilc(farmer) == old_stocks[-1]
    assert soilc(cell) == _band0_via_soilc(cell)

    baseline_old = float(np.mean(old_stocks))
    baseline_new = float(np.mean(new_stocks))
    assert baseline_new == baseline_old
    slope = 0.0125
    accumulation_old = float(baseline_old * np.expm1(slope))
    accumulation_new = float(baseline_new * np.expm1(slope))
    assert accumulation_new == accumulation_old
    payment_old = accumulation_old * GC_M2_TO_TCO2_HA * 185.0 * 12.5
    payment_new = accumulation_new * GC_M2_TO_TCO2_HA * 185.0 * 12.5
    assert payment_new == payment_old


def test_missing_top_layer_keeps_each_old_fallback():
    layers = LAYERS.copy()
    layers[0, -1] = np.float32(np.nan)
    cell = _cell(layers)
    farmer = _farmer(cell, 200)
    assert np.isnan(soilc(cell))
    assert agricultural_soilc(cell, 200, time_idx=-1, farmer=farmer) == 0.0
    assert _band0_via_get_from_earth(farmer, -1) == 0.0


def test_invalid_depth_raises():
    for depth in (0, 30, 300, 4000, -200, "200", None, True):
        with pytest.raises(ValueError, match="soil_carbon_depth"):
            validate_soil_carbon_depth(depth)
        with pytest.raises(ValueError, match="soil_carbon_depth"):
            agricultural_soilc(_cell(), depth)


def test_integer_valued_float_is_a_boundary():
    assert validate_soil_carbon_depth(200.0) == 200
    assert validate_soil_carbon_depth(np.float64(3000.0)) == 3000


def test_too_few_bands_raises():
    cell = _cell(LAYERS[:1])
    with pytest.raises(ValueError, match="1 band"):
        agricultural_soilc(cell, 500)


def test_config_default_is_the_top_layer():
    from pycoupler.config import CoupledConfig, read_yaml

    config = read_yaml(os.path.abspath(CONFIG), CoupledConfig)
    assert config.soil_carbon_depth == 200
    model = SimpleNamespace(config=SimpleNamespace(coupled_config=config))
    assert soil_carbon_depth_mm(model) == 200


def test_missing_config_key_is_the_top_layer():
    model = SimpleNamespace(config=SimpleNamespace(coupled_config=SimpleNamespace()))
    assert soil_carbon_depth_mm(model) == 200
