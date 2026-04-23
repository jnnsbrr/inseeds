"""Farmer entity type class of inseeds_farmer_management"""

# This file is part of pycopancore.
#
# Copyright (C) 2016-2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license
import numpy as np
from enum import Enum

import pycopancore.model_components.base as core
import inseeds.components.base as base

# Bands to exclude from crop calculations - managed grassland as well as
# biomass grass and tree are not crops
NON_CROPS = [
    'rainfed grassland',
    'irrigated grassland',
    'rainfed biomass grass',
    'irrigated biomass grass',
    'rainfed biomass tree',
    'irrigated biomass tree'
]


def sigmoid(x):
    """Map real values to (0, 1) for TPB attitude/norm scores.

    Uses tanh-based sigmoid: output of 0.5 when x=0, approaches 0/1 at extremes.
    Useful for converting unbounded scores to probability-like values.

    Parameters
    ----------
    x : float or array-like
        Input value(s).

    Returns
    -------
    float or ndarray
        Sigmoid output in (0, 1). Zero maps to 0.5.
    """
    return 0.5 * (np.tanh(x) + 1)


class AFT(Enum):
    """AFT types for the farmers."""

    traditionalist: int = 0
    pioneer: int = 1

    @staticmethod
    def random(pioneer_share=0.5):
        return np.random.choice(
            [AFT.pioneer, AFT.traditionalist],
            p=[pioneer_share, 1 - pioneer_share],
        )


class Farmer(core.Individual, base.Individual):
    """Farmer (Individual) entity type mixin class."""

    # standard methods:
    def __init__(self, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        # initialize the AFT specific attributes
        self.init_aft()

        # initialize the coupled (lpjml mapped) attributes
        self.init_coupled_attributes()

        # average harvest date of the cell is used as a proxy for the order
        # of the agents making decisions in time through the year
        self.avg_hdate = self.cell_avg_hdate

        # soilc is the last "measured" soilc value of the farmer whereas the
        #   cell_soilc value is the actual status of soilc of the cell
        self.soilc = self.cell_soilc

        # Same applies for cropyield (as for soilc)
        self.cropyield = self.cell_cropyield

        # Optional outputs (only available in some model versions)
        if "rootmoist_agr" in self.cell.from_earth:
            self.root_moisture = self.cell_root_moisture
        if "litcover_agr" in self.cell.from_earth:
            self.litter_cover = self.cell_litter_cover

    def init_aft(self):
        """Initialize the AFT of the agent."""

        # assign aft to farmer
        self.aft = AFT.random(self.model.config.coupled_config.pioneer_share)
        self.aft_id = self.aft.value

        # assign configuration to aft specific farmer
        self.__dict__.update(
            getattr(
                self.model.config.coupled_config.aftpar, self.aft.name
            ).to_dict()
        )

    def init_coupled_attributes(self):
        """Initialize the mapped variables from the LPJmL input to the farmers.

        Reads from cell.to_earth (LPJmL input) for each variable in coupling_map.
        Raises AttributeError if any required variable is missing.
        """
        self.coupling_map = (
            self.model.config.coupled_config.coupling_map.to_dict()
        )
        self.control_run = self.model.config.coupled_config.control_run

        for attribute, lpjml_attribute in self.coupling_map.items():
            if not isinstance(lpjml_attribute, list):
                lpjml_attribute = [lpjml_attribute]

            value_set = False
            for single_var in lpjml_attribute:
                if single_var not in self.cell.to_earth:
                    continue
                input_data = self.cell.to_earth[single_var].values
                flat = input_data.flatten()

                if len(flat) == 1:
                    setattr(self, attribute, input_data.item())
                elif len(flat) > 1:
                    setattr(self, attribute, flat[0].item())
                value_set = True
                break

            if not value_set:
                lpjml_vars = ", ".join(lpjml_attribute)
                raise AttributeError(
                    f"{lpjml_vars} not in cell.to_earth; "
                    f"ensure LPJmL coupled_input includes these variables "
                    f"for coupling_map.{attribute}"
                )

        for attribute in self.coupling_map:
            if hasattr(self, attribute):
                self.set_lpjml(attribute=attribute)

    def init_neighbourhood(self):
        """Initialize the neighbourhood of the agent."""
        self.neighbourhood = [
            neighbour
            for cell_neighbours in self.cell.neighbourhood
            if len(cell_neighbours.individuals) > 0
            for neighbour in cell_neighbours.individuals
        ]

    def _get_from_earth(self, var_name, as_scalar=False, band=None, drop_band=None, time_idx=-1):  # noqa: E501
        """Get variable from cell.from_earth, handling multi-year data.

        This is the single entry point for accessing from_earth data.
        By default selects the most recent time step if multiple exist.

        Parameters
        ----------
        var_name : str
            Name of the variable in cell.from_earth (e.g. harvestc, cftfrac).
        as_scalar : bool, default False
            If True, return a scalar value (mean or band-specific).
            If False, return the DataArray (with time dimension removed if multi-year).
        band : int, optional
            If given with as_scalar=True, return value at specific band index.
            If given with as_scalar=False, select that band from the DataArray.
        drop_band : list, optional
            If given, drop the specified bands from the DataArray.
        time_idx : int, default -1
            Which time step to select if multiple exist.
            -1 = most recent (default), 0 = first (for initialization with history).

        Returns
        -------
        float or xarray.DataArray
            Scalar value if as_scalar=True, DataArray otherwise.

        Raises
        ------
        AttributeError
            If var_name is not in cell.from_earth.
        ValueError
            If as_scalar=True and value cannot be parsed.
        """
        data = getattr(self.cell.from_earth, var_name, None)
        if data is None:
            raise AttributeError(
                f"{var_name} not in cell.from_earth; "
                f"ensure LPJmL output includes {var_name}"
            )

        # If data has multiple time steps, select the specified one
        if hasattr(data, 'time') and len(data.time) > 1:
            data = data.isel(time=time_idx)

        # Drop bands if specified
        if drop_band is not None:
            data = data.drop_sel(band=drop_band)

        # Select band if specified
        if band is not None:
            data = data.isel(band=band)

        # Return DataArray or convert to scalar
        if not as_scalar:
            return data

        # Convert to scalar
        try:
            if data.size == 1:
                val = float(data.item())
            else:
                val = float(np.nanmean(data.values))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot parse {var_name} from cell.from_earth: {e}"
            ) from e

        if np.isnan(val):
            val = 0.0  # Fallback for test data or missing values
        return val

    @property
    def cell_cropyield(self):
        """Return the average crop yield of the cell."""
        return (
            self._get_from_earth(
                "pft_harvestc",
                as_scalar=False,
                drop_band=NON_CROPS)
                .weighted(
                    self._get_from_earth(
                        "cftfrac",
                        as_scalar=False,
                        drop_band=NON_CROPS)
                )
                .sum("band")
            ).item()

    @property
    def cell_pft_yield(self):
        """Return the average crop yield of the cell."""
        return self._get_from_earth(
            "pft_harvestc",
            as_scalar=True,
            drop_band=NON_CROPS
        )

    @property
    def cell_pft_production(self):
        """Return the average crop yield of the cell."""
        return (
            self._get_from_earth(
                "pft_harvestc",
                as_scalar=True,
                drop_band=NON_CROPS) * \
            self._get_from_earth(
                "cftfrac",
                as_scalar=True,
                drop_band=NON_CROPS) * \
            self.farm_size
        )

    @property
    def cell_soilc(self):
        """Return the top-layer soil carbon of the cell (gC/m2)."""
        return self._get_from_earth("soilc_agr_layer", as_scalar=True, band=0)

    @property
    def cell_root_moisture(self):
        """Return the average rootzone soil moisture of the cells."""
        return self._get_from_earth("rootmoist_agr", as_scalar=True)

    @property
    def cell_litter_cover(self):
        """Return fractional soil cover from litter on agricultural stands (0-1)."""
        return self._get_from_earth("litcover_agr", as_scalar=True)

    @property
    def cell_runoff(self):
        """Return runoff of cell (mm/yr) from LPJmL."""
        return self._get_from_earth("runoff", as_scalar=True)

    @property
    def cell_leaching(self):
        """Return N leaching (gN/m2/yr) from LPJmL (whole cell)."""
        return self._get_from_earth("leaching", as_scalar=True)

    @property
    def cell_fertilizer(self):
        """Return N fertilizer input (gN/m2/yr) from LPJmL."""
        return self._get_from_earth("nfert_agr", as_scalar=True)

    @property
    def cell_irrig(self):
        """Return irrigation water use (mm/yr) from LPJmL.

        Placeholder for future CA irrigation savings calculation:
        CA practices -> more root moisture -> less irrigation demand.
        This happens automatically in LPJmL; tracking here enables
        connecting irrigation savings to profit/capital.
        """
        return self._get_from_earth("irrig", as_scalar=True)

    @property
    def farm_size(self):
        """Return farm size in hectares (sum of cftfrac * area).

        Calculated as the sum of crop functional type fractions times cell area.
        Used for:
        - Scaling maintenance costs
        - Calculating total profit (yield × area × price)
        - Determining equipment investment thresholds

        Returns
        -------
        float
            Farm size in hectares (ha). Cell area from pycopanlpjml is in m²,
            converted to ha (1 ha = 10,000 m²).
        """
        cftfrac = (
            self._get_from_earth("cftfrac")
            .drop_sel(band=NON_CROPS)
        )

        # Sum of all crop fractions
        total_cftfrac = float(np.sum(cftfrac.values))

        # Get area from cell (pycopanlpjml provides this from world.area)
        # Area is in m², convert to hectares (1 ha = 10,000 m²)
        area = self.cell.area
        if hasattr(area, "values"):
            area_m2 = float(np.asarray(area.values).mean())
        else:
            area_m2 = float(area)

        area_ha = area_m2 / 10000.0

        return total_cftfrac * area_ha

    @property
    def cell_avg_hdate(self):
        """Return the average harvest date of the cell."""
        hdate_data = (
            self._get_from_earth("hdate")
        )
        cftfrac_data = (
            self._get_from_earth("cftfrac")
        )

        # Get band values for both variables
        hdate_bands = hdate_data.band.values
        cftfrac_bands = cftfrac_data.band.values

        # Find crop indices in hdate bands that match cftmap
        hdate_crop_idx = [
            i
            for i, item in enumerate(hdate_bands)
            if any(x in item for x in self.model.config.cftmap)
        ]

        # Find matching bands in cftfrac (same crop names)
        hdate_crop_names = [hdate_bands[i] for i in hdate_crop_idx]
        cftfrac_crop_idx = [
            i
            for i, item in enumerate(cftfrac_bands)
            if item in hdate_crop_names
        ]

        if len(cftfrac_crop_idx) == 0:
            return 365

        # Get the selected data
        hdate_selected = hdate_data.isel(band=hdate_crop_idx)
        cftfrac_selected = cftfrac_data.isel(band=cftfrac_crop_idx)

        if np.sum(cftfrac_selected.values) == 0:
            return 365
        else:
            return np.average(
                hdate_selected.values, weights=cftfrac_selected.values
            )

    def set_lpjml(self, attribute):
        """Set the mapped variables from the farmers to the LPJmL input."""
        lpjml_attribute = self.coupling_map[attribute]

        if not isinstance(lpjml_attribute, list):
            lpjml_attribute = [lpjml_attribute]

        for single_var in lpjml_attribute:
            self.cell.to_earth[single_var][:] = getattr(self, attribute)

    def update(self, t):
        super().update(t)

        # update cell-level observations from LPJmL output
        self.avg_hdate = self.cell_avg_hdate
        self.cropyield = self.cell_cropyield
        self.soilc = self.cell_soilc
        if "rootmoist_agr" in self.cell.from_earth:
            self.root_moisture = self.cell_root_moisture
        if "litcover_agr" in self.cell.from_earth:
            self.litter_cover = self.cell_litter_cover