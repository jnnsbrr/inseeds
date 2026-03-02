"""Add rootmoist_agr to existing LPJmL output pickle if missing.

Run from inseeds repo root:
    python tests/data/add_rootmoist_agr.py

Required for Farmer tests (cell_root_moisture). When regenerating test data
via write_testdata.py, rootmoist_agr is now included in coupled_output.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def add_rootmoist_agr(pkl_path: Path) -> bool:
    """Add rootmoist_agr to pickle if missing. Returns True if modified."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if hasattr(data, "data_vars") and "rootmoist_agr" in data.data_vars:
        return False

    ncell = data.sizes["cell"]
    ntime = data.sizes["time"]
    var = xr.Variable(
        ("cell", "time"),
        np.full((ncell, ntime), 0.3),
        attrs={"units": "fraction", "long_name": "Root zone soil moisture"},
    )
    data._variables["rootmoist_agr"] = var

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return True


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    pkl_path = script_dir / "lpjml_output.pkl"
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found")
        sys.exit(1)
    if add_rootmoist_agr(pkl_path):
        print(f"Added rootmoist_agr to {pkl_path}")
    else:
        print(f"rootmoist_agr already present in {pkl_path}")
