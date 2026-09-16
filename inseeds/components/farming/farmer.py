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
        cell.cache.farm_size_ha = 30.5

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
    """Weighted mean harvest day-of-year for cell.

    Weighted over the bands hdate and cftfrac share, which is what keeps the
    dates and their weights aligned: the two outputs do not carry the same
    bands. The landuse meta map narrows them to the crops the run grows, and
    where it is absent the shared bands do that on their own.
    """
    hdate = get_cell_var(cell, "hdate")
    cftfrac = get_cell_var(cell, "cftfrac")
    landusemap = model.config.landusemap

    hdate_bands = [
        b for b in hdate.band.values
        if landusemap is None or any(x in b for x in landusemap)
    ]
    shared = [b for b in cftfrac.band.values if b in hdate_bands]

    if not shared:
        return 365

    hdate_sel = hdate.sel(band=shared)
    cft_sel = cftfrac.sel(band=shared)

    if np.sum(cft_sel.values) == 0:
        return 365
    return float(np.average(hdate_sel.values, weights=cft_sel.values))


def farm_size_ha(cell):
    """Cropped area of a cell in hectares (crop-band cftfrac * cell area).

    Managed grassland and biomass plantations are dropped, so this is the area
    that earns the revenue. Every monetary quantity is scaled on it - costs, the
    capital endowment, the min_capital threshold, the baseline revenue floor -
    because a cost charged on more land than the revenue would not be comparable
    with it. In rangeland-dominated cells the two differ by an order of
    magnitude.

    Parameters
    ----------
    cell : Cell
        The cell to compute the cropped area for.

    Returns
    -------
    float
        Cropped area in hectares.
    """
    landuse = get_cell_var(cell, "cftfrac", drop_band=NON_CROPS, time_idx=-1)

    total = float(np.sum(landuse.values))
    area = cell.area
    area_m2 = float(np.asarray(area.values).mean()) if hasattr(area, "values") else float(area)
    return total * (area_m2 / 10000.0)


def cropyield(cell):
    """Crop yield per m2 of cropland (gC/m2).

    Harvest weighted by cftfrac and merely summed gives production per m2 of
    *cell*, which scales with how much of the cell is cropped. That makes
    yields incomparable between cells and lets a shrinking cropland extent
    masquerade as falling yields. Dividing by the cropped fraction turns it
    back into a per-hectare yield.
    """
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


# Root moisture is taken from the crop stands alone. rootmoist_agr averages over
# every agricultural stand including setaside, and setaside is where the cover
# crop grows, so it mixes the water a cover crop transpires on fallow land into
# what is meant to represent the water available to the crop - a cost read as a
# state. rootmoist_agr stays as a fallback for output written before
# rootmoist_crop existed.
MOISTURE_VARS = ("rootmoist_crop", "rootmoist_agr")


def moisture_var(source):
    """Name of the root-moisture variable present in source, or None."""
    for name in MOISTURE_VARS:
        if name in source:
            return name
    return None


def root_moisture(cell):
    """Root zone soil moisture of the crop stands (mm)."""
    data = get_cell_var(cell, moisture_var(cell.from_earth))
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
    if moisture_var(cell.from_earth):
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


# Guards on the reported concentration, which is a ratio and so goes to
# infinity as drainage goes to zero. They only shape the output series now: the
# flux the model tracks needs no such guard, being bounded by the nitrogen that
# went onto the field in the first place.
MIN_RUNOFF_MM = 1.0     # mm/yr; below this, concentration is undefined
MAX_CONC_MGN_L = 200.0  # mgN/l; ~18x the EU limit of 50 mg NO3/l = 11.3 mgN/l

# LPJmL reports per square metre and policy instruments are written per hectare
# and per tonne, so every priced quantity crosses one of these on its way into
# a cash flow. Kept together so the two halves of an instrument - what the farm
# is paid at year end and what it expects when deciding - cannot drift apart.
GC_M2_TO_TCO2_HA = 0.01 * 44.0 / 12.0  # gC/m2 -> tCO2e/ha
GN_M2_TO_KGN_HA = 10.0                 # gN/m2 -> kgN/ha


def agricultural_leaching(farmer, time_idx=-1):
    """Nitrogen leached from this cell, gN per m2 of cell.

    Uses the whole-cell ``leaching`` output from LPJmL, which is what reaches
    the water body regardless of which stand it came from. This matches the
    boundaries package measurement and keeps the tax base consistent with
    environmental outcomes: the aquifer does not know which hectare the
    nitrogen came from.

    Parameters
    ----------
    farmer : Farmer
        Farmer whose cell is read.
    time_idx : int, optional
        Index into the output time axis, last year by default.

    Returns
    -------
    float
        Leached nitrogen in gN/m2 of cell.
    """
    flux = farmer.get_from_earth(
        "leaching", as_scalar=True, time_idx=time_idx
    )
    return float(flux)


def nitrate_concentration(leaching, runoff):
    """Nitrate concentration of the leachate in mgN/l, or None if undefined.

    Reported, not scored. Nothing in the model decides on this: a concentration
    is an intensive quantity, so it cannot be summed over farms or multiplied by
    a price, and it rises as drainage falls, which makes the driest cells - the
    ones losing least nitrogen - look like the worst polluters. The flux is what
    the model tracks and what an instrument would be written on. This is here
    because a concentration is what a nitrate standard is written on, and so
    what makes the output comparable to one.

    The numerator is per m2 of cropland (see ``agricultural_leaching``) and the
    denominator is the cell's runoff, so this assumes cropland drains at about
    the rate the cell does. LPJmL does not report runoff by stand, and the
    alternative - dividing agricultural nitrogen by the whole cell's water -
    would say a farm on a tenth of its cell pollutes a tenth as much, which is
    worse. The same assumption sits behind the levy-free allowance, which is
    this concentration read against a standard.

    None when runoff is below MIN_RUNOFF_MM: with no water leaving the profile
    there is no concentration to measure, and the ratio is 0/0.

    Above the threshold the concentration is capped at MAX_CONC_MGN_L. The cap
    is a numerical guard, not a regulatory reference: the EU drinking-water
    limit of 50 mg NO3/l corresponds to 11.3 mgN/l, so the cap sits well above
    it. Unguarded, near-zero runoff produces concentrations up to 47,000 mgN/l.

    Parameters
    ----------
    leaching : float
        Leached nitrogen flux (gN/m2).
    runoff : float
        Runoff (mm/yr).

    Returns
    -------
    float or None
        Concentration in mgN/l, or None when runoff is too low to define one.
    """
    if runoff < MIN_RUNOFF_MM:
        return None
    return min(leaching * 1e3 / runoff, MAX_CONC_MGN_L)


def log_rate(value, baseline, negligible):
    """Symmetric relative deviation from baseline, ln((v + n) / (b + n)).

    Log rather than the plain ratio v/b - 1, which is bounded at -1 below but
    unbounded above and so weights an improvement more heavily than a
    deterioration of the same multiplicative size. Under the log a doubling
    (+0.69) and a halving (-0.69) are equidistant from zero, and for small
    deviations ln(v/b) ~ v/b - 1, so a modest change still reads as a
    percentage change. It also matches diminishing sensitivity to magnitude
    (Weber-Fechner).

    Both sides are offset by ``negligible``, the level of this metric at which
    a difference stops being worth noticing. That offset is what lets a bare
    ratio be taken at all: ln(v/b) needs a positive baseline and grows without
    bound as either side approaches zero, so a cell whose historic mean is
    almost nothing reports a spectacular change from a physically trivial one.
    Offsetting instead of clamping the denominator keeps the rate exactly zero
    whenever nothing changed, however small the baseline, and bounds a collapse
    to zero at -ln(1 + b/n) rather than an arbitrary sentinel.

    Lives here rather than next to its callers because both the farmer metrics
    and the historic series that calibrates the scoring scales have to apply
    the same transform, or the ruler and the values it measures end up on
    different scales.

    Parameters
    ----------
    value : float
        Current metric value.
    baseline : float
        Reference baseline value.
    negligible : float
        Level of this metric that counts as negligible, in the metric's own
        physical unit. Must be positive; see materiality in the config.

    Returns
    -------
    float
        ln((value + negligible) / (baseline + negligible)).
    """
    if baseline + negligible <= 0:
        return 0.0
    return float(np.log((value + negligible) / (baseline + negligible)))


def annuity_factor(horizon, rate):
    """Years of annual benefit a farmer counts over horizon, discounted at rate.

    Returns (1 - (1+r)^-H) / r, the present value of one unit received annually
    for H years. Discounting rather than a bare multiplication by the horizon,
    because an undiscounted window is a cliff rather than a preference: every
    year inside it counts fully and every year beyond it counts nothing.

    The two arguments trade off sharply, which is why both are stated: ten years
    at 10% counts 6.14 years of benefit, and at 15% counts 5.02. A long horizon
    is therefore only as generous as the discount rate allows.

    Parameters
    ----------
    horizon : float
        Years ahead the farmer considers.
    rate : float
        Annual discount rate. Zero falls back to the horizon itself.

    Returns
    -------
    float
        Discounted number of benefit-years.
    """
    if horizon <= 0:
        return 0.0
    if rate <= 0:
        return float(horizon)
    return float((1.0 - (1.0 + rate) ** -horizon) / rate)


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
        if moisture_var(self.cell.from_earth):
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
        """Initialize the neighbourhood of the agent, ordered by cell.

        Sorted rather than taken as it comes, because ``individuals`` is a set:
        iterating it gives an order that depends on object identity and so
        differs between processes. Local social learning sums over this list and
        takes a weighted median of it, both of which can land differently under
        a different order, which would leave a seeded run irreproducible.
        """
        self.neighbourhood = sorted(
            (
                neighbour
                for cell_neighbours in self.cell.neighbourhood
                if len(cell_neighbours.individuals) > 0
                for neighbour in cell_neighbours.individuals
            ),
            key=lambda neighbour: int(neighbour.cell.cell_index),
        )

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
    def country(self):
        """Return the country of the cell."""
        return self.social_system if hasattr(self.social_system,"entity") and self.social_system.entity == "country" else None

    @property
    def cell_cropyield(self):
        """Return the average crop yield of the cell (cache-aware)."""
        cached = self._cached("cropyield")
        if cached is not None:
            return cached
        return cropyield(self.cell)

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
        """Return rootzone soil moisture of the crop stands (cache-aware)."""
        cached = self._cached("root_moisture")
        if cached is not None:
            return cached
        return self.get_from_earth(
            moisture_var(self.cell.from_earth), as_scalar=True, time_idx=-1
        )

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
        """Nitrogen leached from this cell's cropland, gN/m2 of cropland/yr.

        Agricultural stands only and normalised to the cropped area; see
        ``agricultural_leaching`` for why both matter.
        """
        return agricultural_leaching(self, time_idx=-1)

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
    def farm_size(self):
        """Cropped area in hectares (crop-band cftfrac × cell area).

        Grassland and biomass plantations are excluded, so this is the area
        that earns the revenue. Costs, capital and the min_capital threshold
        are all scaled on it. Uses the cell cache when it holds
        ``farm_size_ha``.
        """
        cached = self._cached("farm_size_ha")
        if cached is not None:
            return cached
        return farm_size_ha(self.cell)

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
            if moisture_var(self.cell.from_earth):
                self.root_moisture = self.cell_root_moisture
            if "litcover_agr" in self.cell.from_earth:
                self.litter_cover = self.cell_litter_cover
