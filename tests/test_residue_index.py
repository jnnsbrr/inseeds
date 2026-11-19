"""The residue fractions a farm is charged come from its own cell."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

import inseeds.components.farming.ca_farmer as ca_farmer_module
from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer


N_CFT = 16
CROPS = 13


def _residue_cache(n_cells):
    """A grid-wide cache in which cell i removes the share i / 10."""
    removed = np.tile(np.arange(n_cells) / 10.0, (N_CFT, 1))
    zeros = np.zeros((N_CFT, n_cells))
    return xr.Dataset(
        {
            "frac_burnt": (("cft", "cell"), zeros),
            "frac_removed": (("cft", "cell"), removed),
            "frac_recycled": (("cft", "cell"), 1.0 - removed),
        }
    )


def _farmer(monkeypatch, cell_index, local_index, n_cells=8):
    bands = [
        f"{water} crop{i}" for water in ("rainfed", "irrigated")
        for i in range(CROPS)
    ]
    cftfrac = xr.DataArray(
        np.full(len(bands), 0.05), dims=("band",), coords={"band": bands}
    )
    monkeypatch.setattr(
        ca_farmer_module, "get_cell_var", lambda *args, **kwargs: cftfrac
    )

    exogenous = SimpleNamespace(
        residue=_residue_cache(n_cells), keys=lambda: ["residue"]
    )
    farmer = object.__new__(ConservationAgricultureFarmer)
    farmer._cell = SimpleNamespace(
        cell_index=cell_index,
        local_index=local_index,
        grid=SimpleNamespace(cell=SimpleNamespace(item=lambda: cell_index)),
    )
    farmer._model = SimpleNamespace(
        world=SimpleNamespace(exogenous=exogenous)
    )
    return farmer


def test_fractions_follow_the_global_cell_not_the_rank_in_the_country(
    monkeypatch,
):
    """Cell 5 of the grid, third in its country, reads cell 5's fractions."""
    farmer = _farmer(monkeypatch, cell_index=5, local_index=2)
    fractions = farmer.get_residue_fractions()
    assert np.isclose(fractions["removed"], 0.5)


def test_first_cell_of_the_grid_is_read_from_position_zero(monkeypatch):
    """Global index 0 is a position, not a missing value."""
    farmer = _farmer(monkeypatch, cell_index=0, local_index=3)
    fractions = farmer.get_residue_fractions()
    assert np.isclose(fractions["removed"], 0.0)
