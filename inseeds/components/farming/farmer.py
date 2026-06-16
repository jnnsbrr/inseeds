"""Farmer entity type class of inseeds_farmer_management"""

import numpy as np
from enum import Enum

import pycopancore.model_components.base as core
import inseeds.components.base as base

NON_CROPS = [
    'rainfed grassland', 'irrigated grassland',
    'rainfed biomass grass', 'irrigated biomass grass',
    'rainfed biomass tree', 'irrigated biomass tree'
]


# =============================================================================
# Cell Cache - unified caching for cell-level values
# =============================================================================

class CellCache:
    """Simple cache for cell-level computed values.
    
    Usage:
        cell.cache = CellCache()
        cell.cache.avg_hdate = 180.0
        cell.cache.net_farm_size_ha = 30.5
        cell.cache.gross_farm_size_ha = 50.5

        # Access:
        value = cell.cache.avg_hdate
        if "avg_hdate" in cell.cache: ...
    """
    
    def __init__(self):
        self._data = {}
    
    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __getattr__(self, name):
        if name == "_data":
            return super().__getattribute__(name)
        if name not in self._data:
            raise AttributeError(f"Cache has no '{name}'. Call cache_cell() first?")
        return self._data[name]
    
    def get(self, name, default=None):
        return self._data.get(name, default)
    
    def __contains__(self, name):
        return name in self._data


# =============================================================================
# Cell data helpers
# =============================================================================

def get_cell_var(cell, var_name, time_idx=-1, drop_band=None, band=None,
                 source="from_earth", partial_match=False):
    """Read variable from cell.from_earth or cell.to_earth with optional slicing.
    
    Parameters
    ----------
    cell : Cell
        The cell to read data from.
    var_name : str
        Name of the variable to read.
    time_idx : int, default=-1
        Time index to select (only if data has multiple time steps).
    drop_band : list, optional
        Band names to exclude.
    band : int, optional
        Band index to select.
    source : str, default="from_earth"
        Data source: "from_earth" (LPJmL output) or "to_earth" (LPJmL input).
    partial_match : bool, default=False
        If True, use partial matching for drop_band (e.g., 'biomass grass' drops
        'rainfed biomass grass' and 'drip irrigated biomass grass').
        If False (default), use fast exact matching.
    
    Returns
    -------
    xarray.DataArray
        The requested variable with slicing applied.
    """
    if source == "from_earth":
        container = cell.from_earth
    elif source == "to_earth":
        container = cell.to_earth
    else:
        raise ValueError(f"source must be 'from_earth' or 'to_earth', got '{source}'")
    
    data = getattr(container, var_name, None)
    if data is None:
        raise AttributeError(f"{var_name} not in cell.{source}")
    if hasattr(data, "time") and len(data.time) > 1:
        data = data.isel(time=time_idx)
    if drop_band is not None:
        if partial_match and "band" in data.dims:
            bands_to_drop = [
                b for b in data.band.values
                if any(pattern in b for pattern in drop_band)
            ]
            if bands_to_drop:
                data = data.drop_sel(band=bands_to_drop)
        else:
            data = data.drop_sel(band=drop_band)
    if band is not None:
        data = data.isel(band=band)
    return data


def avg_hdate(cell, model):
    """Weighted mean harvest day-of-year for cell."""
    hdate = get_cell_var(cell, "hdate")
    cftfrac = get_cell_var(cell, "cftfrac")
    cftmap = model.config.cftmap
    
    hdate_idx = [i for i, b in enumerate(hdate.band.values) 
                 if any(x in b for x in cftmap)]
    cft_idx = [i for i, b in enumerate(cftfrac.band.values) 
               if b in [hdate.band.values[j] for j in hdate_idx]]
    
    if not cft_idx:
        return 365
    
    hdate_sel = hdate.isel(band=hdate_idx)
    cft_sel = cftfrac.isel(band=cft_idx)
    
    if np.sum(cft_sel.values) == 0:
        return 365
    return float(np.average(hdate_sel.values, weights=cft_sel.values))


def farm_size_ha(cell, net=True, average_over_spinup=False):
    """Farm size in hectares (cftfrac * area).
    
    Parameters
    ----------
    cell : Cell
        The cell to compute farm size for.
    net : bool, default True
        If True, subtract non-crop landuse from total landuse.
    average_over_spinup : bool
        If True, average cftfrac over spinup years for robust baseline.
    """
    if net:
        landuse = getattr(cell.from_earth, "cftfrac", None).isel(time=-1)
    else:
        landuse = get_cell_var(
            cell,
            "landuse",
            source="to_earth",
            drop_band=NON_CROPS,
            partial_match=True,
            time_idx=-1,
        )
    
    total = float(np.sum(landuse.values))
    area = cell.area
    area_m2 = float(np.asarray(area.values).mean()) if hasattr(area, "values") else float(area)
    return total * (area_m2 / 10000.0)


def cropyield(cell):
    """Weighted crop yield (gC/m2)."""
    harvestc = get_cell_var(cell, "pft_harvestc", drop_band=NON_CROPS)
    cftfrac = get_cell_var(cell, "cftfrac", drop_band=NON_CROPS)
    w = cftfrac.values.flatten()
    v = harvestc.values.flatten()
    return float((v * w).sum() / w.sum()) if w.sum() > 0 else 0.0


def soilc(cell):
    """Top-layer soil carbon (gC/m2)."""
    data = get_cell_var(cell, "soilc_agr_layer", band=0)
    v = data.values
    return float(v.item()) if hasattr(v, "item") else float(np.nanmean(v))


def root_moisture(cell):
    """Root zone soil moisture (mm)."""
    data = get_cell_var(cell, "rootmoist_agr")
    v = data.values
    return float(v.item()) if hasattr(v, "item") else float(np.nanmean(v))


def litter_cover(cell):
    """Litter cover fraction (0-1)."""
    data = get_cell_var(cell, "litcover_agr")
    v = data.values
    return float(v.item()) if hasattr(v, "item") else float(np.nanmean(v))


def cache_yearly(cell, model):
    """Cache yearly values on cell after LPJmL output refresh."""
    if not hasattr(cell, "cache"):
        cell.cache = CellCache()
    cell.cache.avg_hdate = avg_hdate(cell, model)
    cell.cache.cropyield = cropyield(cell)
    cell.cache.soilc = soilc(cell)
    if "rootmoist_agr" in cell.from_earth:
        cell.cache.root_moisture = root_moisture(cell)
    if "litcover_agr" in cell.from_earth:
        cell.cache.litter_cover = litter_cover(cell)


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

    def get_from_earth(self, var_name, as_scalar=False, band=None, drop_band=None, time_idx=None):  # noqa: E501
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
        if hasattr(data, 'time') and len(data.time) > 1 and time_idx is not None:
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

    def _cached(self, name):
        """Get cached value if available."""
        cache = getattr(self.cell, "cache", None)
        return cache.get(name) if cache else None

    @property
    def cell_cropyield(self):
        """Return the average crop yield of the cell (cache-aware)."""
        cached = self._cached("cropyield")
        if cached is not None:
            return cached
        return (
            self.get_from_earth("pft_harvestc", as_scalar=False, drop_band=NON_CROPS, time_idx=-1)
            .weighted(self.get_from_earth("cftfrac", as_scalar=False, drop_band=NON_CROPS, time_idx=-1))
            .sum("band")
        ).item()

    @property
    def cell_pft_yield(self):
        """Return average crop yield across PFTs (gC/m²)."""
        return self.get_from_earth("pft_harvestc", as_scalar=True, drop_band=NON_CROPS, time_idx=-1)

    @property
    def cell_pft_production(self):
        """Return total crop production (gC).
        
        Calculated as sum of (yield × crop fraction × area) across all PFTs.
        """
        pft_harvestc = self.get_from_earth("pft_harvestc", as_scalar=False, drop_band=NON_CROPS, time_idx=-1)
        cftfrac = self.get_from_earth("cftfrac", as_scalar=False, drop_band=NON_CROPS, time_idx=-1)
        area_m2 = self.cell.area.item()
        return float((pft_harvestc * cftfrac * area_m2).sum())

    @property
    def cell_soilc(self):
        """Return top-layer soil carbon (gC/m2, cache-aware)."""
        cached = self._cached("soilc")
        if cached is not None:
            return cached
        return self.get_from_earth("soilc_agr_layer", as_scalar=True, band=0, time_idx=-1)

    @property
    def cell_root_moisture(self):
        """Return rootzone soil moisture (cache-aware)."""
        cached = self._cached("root_moisture")
        if cached is not None:
            return cached
        return self.get_from_earth("rootmoist_agr", as_scalar=True, time_idx=-1)

    @property
    def cell_litter_cover(self):
        """Return fractional soil cover from litter (0-1, cache-aware)."""
        cached = self._cached("litter_cover")
        if cached is not None:
            return cached
        return self.get_from_earth("litcover_agr", as_scalar=True, time_idx=-1)

    @property
    def cell_runoff(self):
        """Return runoff of cell (mm/yr) from LPJmL."""
        return self.get_from_earth("runoff", as_scalar=True, time_idx=-1)

    @property
    def cell_leaching(self):
        """Return N leaching (gN/m2/yr) from LPJmL (whole cell)."""
        return self.get_from_earth("leaching", as_scalar=True, time_idx=-1)

    @property
    def cell_fertilizer(self):
        """Return N fertilizer input (gN/m2/yr) from LPJmL."""
        return self.get_from_earth("nfert_agr", as_scalar=True, time_idx=-1)

    @property
    def cell_irrig(self):
        """Return irrigation water use (mm/yr) from LPJmL.

        Placeholder for future CA irrigation savings calculation:
        CA practices -> more root moisture -> less irrigation demand.
        This happens automatically in LPJmL; tracking here enables
        connecting irrigation savings to profit/capital.
        """
        return self.get_from_earth("irrig", as_scalar=True, time_idx=-1)

    @property
    def net_farm_size(self):
        """Return farm size in hectares (sum of cftfrac * area).

        Calculated as the sum of crop functional type fractions times cell area.
        Uses cached value if available (set during initialization to average
        over spinup years for robust baseline).

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
        cached = self._cached("farm_size_ha")
        if cached is not None:
            return cached
        return farm_size_ha(self.cell, net=True)

    @property
    def gross_farm_size(self):
        """Return farm size in hectares (sum of cftfrac * area)."""
        return farm_size_ha(self.cell, net=False)

    @property
    def cell_avg_hdate(self):
        """Return average harvest date (cache-aware)."""
        cached = self._cached("avg_hdate")
        if cached is not None:
            return cached
        return avg_hdate(self.cell, self.model)

    def set_lpjml(self, attribute):
        """Set the mapped variables from the farmers to the LPJmL input."""
        lpjml_attribute = self.coupling_map[attribute]

        if not isinstance(lpjml_attribute, list):
            lpjml_attribute = [lpjml_attribute]

        for single_var in lpjml_attribute:
            da = self.cell.to_earth[single_var]
            value = getattr(self, attribute)
            # Handle both 0D (scalar) and 1D+ arrays
            if da.ndim == 0:
                da.values[()] = value
            else:
                da[:] = value

    def update(self, t):
        super().update(t)

        # update cell-level observations from LPJmL output (using cached values)
        # avg_hdate is set once at init and used only for deterministic update order
        self.cropyield = self.cell_cropyield
        self.soilc = self.cell_soilc
        # Use cached values if available (check once during init, not every year)
        cache = getattr(self.cell, "cache", None)
        if cache is not None:
            if "root_moisture" in cache:
                self.root_moisture = cache.root_moisture
            if "litter_cover" in cache:
                self.litter_cover = cache.litter_cover
        else:
            # Fallback to direct access (slower)
            if "rootmoist_agr" in self.cell.from_earth:
                self.root_moisture = self.cell_root_moisture
            if "litcover_agr" in self.cell.from_earth:
                self.litter_cover = self.cell_litter_cover