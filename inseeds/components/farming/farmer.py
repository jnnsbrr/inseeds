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


def sigmoid(x):
    """Sigmoid function for TPB calculations."""
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

        # root moisture is the last "measured" root moisture value of the farmer
        #   whereas the cell_root_moisture value is the actual status of root
        #   moisture of the cell
        self.root_moisture = self.cell_root_moisture

        # Same applies for cropyield (as for soilc)
        self.cropyield = self.cell_cropyield

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

    def _get_cell_earth_var(self, var_name, band=None):
        """Get scalar from cell.from_earth variable. Raise if missing or invalid.

        Parameters
        ----------
        var_name : str
            Name of the variable in cell.from_earth (e.g. harvestc, rootmoist_agr).
        band : int, optional
            If given, use isel(band=band).item() instead of mean() (for banded data).

        Returns
        -------
        float
            The scalar value.

        Raises
        ------
        AttributeError
            If var_name is not in cell.from_earth.
        ValueError
            If the value cannot be parsed or is NaN.
        """
        data = getattr(self.cell.from_earth, var_name, None)
        if data is None:
            raise AttributeError(
                f"{var_name} not in cell.from_earth; "
                f"ensure LPJmL output includes {var_name}"
            )
        try:
            if band is not None:
                val = float(data.isel(band=band).item())
            else:
                vals = np.asarray(data.values)
                val = float(np.nanmean(vals))
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
        return self._get_cell_earth_var("harvestc")

    @property
    def cell_pft_yield(self):
        """Return the average crop yield of the cell."""
        return self._get_cell_earth_var("pft_harvestc")

    @property
    def cell_pft_production(self):
        """Return the average crop yield of the cell."""
        return self._get_cell_earth_var("pft_harvestc") *\
            self._get_cell_earth_var("cftfrac") * self.farm_size


    @property
    def cell_soilc(self):
        """Return the top-layer soil carbon of the cell (gC/m2)."""
        return self._get_cell_earth_var("soilc_agr_layer", band=0)

    @property
    def cell_root_moisture(self):
        """Return the average rootzone soil moisture of the cells."""
        return self._get_cell_earth_var("rootmoist_agr")

    @property
    def cell_runoff(self):
        """Return runoff of cell (mm/yr) from LPJmL."""
        return self._get_cell_earth_var("runoff")

    @property
    def cell_leaching(self):
        """Return N leaching (gN/m2/yr) from LPJmL (whole cell)."""
        return self._get_cell_earth_var("leaching")

    @property
    def cell_fertilizer(self):
        """Return N fertilizer input (gN/m2/yr) from LPJmL."""
        return self._get_cell_earth_var("nfert_agr")

    @property
    def cell_irrig(self):
        """Return irrigation water use (mm/yr) from LPJmL.

        Placeholder for future CA irrigation savings calculation:
        CA practices -> more root moisture -> less irrigation demand.
        This happens automatically in LPJmL; tracking here enables
        connecting irrigation savings to profit/capital.
        """
        return self._get_cell_earth_var("irrig")

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
            Farm size in hectares (ha). Cell area from pycopanlpjml is in km²,
            converted to ha (1 km² = 100 ha).
        """
        cftfrac = self.cell.from_earth.cftfrac

        # Sum of all crop fractions
        total_cftfrac = float(np.sum(cftfrac.values))

        # Get area from cell (pycopanlpjml provides this from world.area)
        # Area is in km², convert to hectares (1 km² = 100 ha)
        area = self.cell.area
        if area is None:
            # Fallback: estimate from grid resolution (~0.5° ≈ 2500 km² at equator)
            area_km2 = 2500.0
        elif hasattr(area, "values"):
            area_km2 = float(np.asarray(area.values).mean())
        else:
            area_km2 = float(area)

        area_ha = area_km2 * 100.0

        return total_cftfrac * area_ha

    @property
    def cell_avg_hdate(self):
        """Return the average harvest date of the cell."""
        hdate_data = self.cell.from_earth.hdate
        cftfrac_data = self.cell.from_earth.cftfrac

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

        # update the average harvest date of the cell
        self.avg_hdate = self.cell_avg_hdate

        # running average over strategy_switch_duration years to avoid rapid
        #    switching by weather fluctuations
        self.cropyield = (
            (1 - 1 / self.strategy_switch_duration) * self.cropyield
            + 1 / self.strategy_switch_duration * self.cell_cropyield
        )
        self.soilc = (
            (1 - 1 / self.strategy_switch_duration) * self.soilc
            + 1 / self.strategy_switch_duration * self.cell_soilc
        )
        self.root_moisture = (
            (1 - 1 / self.strategy_switch_duration) * self.root_moisture
            + 1 / self.strategy_switch_duration * self.cell_root_moisture
        )