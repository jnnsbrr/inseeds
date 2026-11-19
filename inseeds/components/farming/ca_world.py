"""CA-specific World entity holding the reference scales for performance scoring."""

import numpy as np

from inseeds.components.farming.ca_management import ManagementBundle
from inseeds.components.farming.metrics import METRIC_KEYS
from inseeds.components.farming.world import World

# For normally distributed data, std ≈ 1.4826 × MAD
_MAD_SCALE = 1.4826

# Baseline covariates the land-similarity kernel measures peer distance in.
# Pre-coupling levels, so they describe the land and not the management.
#
# Nitrogen mineralisation rather than soil carbon, although the carbon baseline
# is also to hand. Carbon is a stock and an ambiguous one: a lot of it means
# nitrogen is abundant if the C:N ratio is low and immobilised if it is high,
# so it cannot say on its own whether a practice that supplies nitrogen is
# worth anything here. Mineralisation is the flux that ambiguity resolves to -
# how much plant-available nitrogen this soil actually releases in a year -
# which is the quantity a legume's fixation competes with directly.
#
# Moisture and yield complete the description: whether the water a practice
# conserves is scarce here, and how productive the land is to begin with.
LAND_COVARIATES = (
    "baseline_nmineralization",
    "baseline_moisture",
    "baseline_yield",
)

#: Key the global fallback sits under in the conventional reference, so that a
#: country holding too few conventional farms of its own still has a drift to
#: read. Not a country code, and deliberately not one that could collide.
GLOBAL_REFERENCE = "__global__"

#: Conventional farms a country needs before its own median is used instead of
#: the global one. Below this the median is a sample of one or two farms and
#: would carry their weather into every credit written in that country.
MIN_REFERENCE_FARMS = 5

#: Observations a tracker needs before its slope is read at all. Matches the
#: guard in ``ManagementPerformanceTracker.trend``: a slope over two points is
#: a difference, and carries the whole of one year's weather.
MIN_REFERENCE_YEARS = 3


def _robust_std(values: np.ndarray) -> float:
    """Robust standard deviation via MAD × 1.4826.

    Median absolute deviation resists outliers (e.g. a few bankrupt or windfall
    farms) better than sample std or IQR. The 1.4826 factor matches std under
    normality.

    Parameters
    ----------
    values : np.ndarray
        Values to summarise.

    Returns
    -------
    float
        Robust std estimate, or 0.0 if empty.
    """
    if values.size == 0:
        return 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(mad * _MAD_SCALE)


class CAWorld(World):
    """World-level reference scales for performance scoring.

    Recomputed from farmer trackers at initialisation and at the start of each
    simulated year. Used as the common scaling factors for social learning
    (between-farm comparisons).

    Self-referenced rates
    ---------------------
    Each metric is a relative deviation from the farm's own historic baseline:
    ``ln((v+n)/(b+n))`` for environmental metrics and ``asinh`` for profit
    (see ``farmer.log_rate``). Rates are dimensionless and centred at zero, so
    climate, soil and farm size cancel by construction.

    Level and trend scales
    ----------------------
    Level scale = max(cross-farmer robust spread of current rates, median
    per-farmer interannual rate std, 0.01). Trend scale = robust MAD of
    projected trends (rate slope × horizon), floored at 0.01.

    Cross-farmer spread puts stocks and flows on one footing. Yield/profit
    deviations stay of order one year's fluctuation; soil carbon integrates
    increments, so its deviation from baseline grows while year-to-year noise
    stays small — against an interannual-only ruler it would produce large z
    and override other metrics' weights. The population spread lengthens on
    the same timescale as that stock signal. At initialisation every farm sits
    on its baseline (spread ≈ 0), so the interannual std is the lower bound
    until the population diverges. Annual recomputation is required because
    that spread changes as farms diverge.
    """

    def compute_reference_scales(self):
        """Recompute reference scales from current farmer trackers.

        Called after farmers exist at init, then each year before decisions.
        For every metric in ``METRIC_KEYS``:

        - centre = 0.0 (self-referenced by construction)
        - ``metric_std`` = max(cross-farmer MAD of rates, median per-farmer
          interannual rate std, 0.01)
        - ``metric_trend_std`` = max(MAD of rate_slope × horizon, 0.01)

        Stored in ``world.statistic["reference_scales"]``.
        """
        levels = {m: [] for m in METRIC_KEYS}
        noise_stds = {m: [] for m in METRIC_KEYS}
        proj_trends = {m: [] for m in METRIC_KEYS}

        for farmer in self.farmers:
            if not (
                hasattr(farmer, "behaviour")
                and hasattr(farmer.behaviour, "performance_tracker")
            ):
                continue

            tracker = farmer.behaviour.performance_tracker
            # Must match the horizon used in compute_performance_score
            horizon = farmer.behaviour.projection_weight()
            trend = tracker.trend
            std = tracker.level_std

            for metric in METRIC_KEYS:
                level = getattr(tracker, f"mean_{metric}")
                if np.isfinite(level):
                    levels[metric].append(level)
                if std[metric] > 0:
                    noise_stds[metric].append(std[metric])
                proj_trends[metric].append(trend[metric] * horizon)

        scales = {}
        for metric in METRIC_KEYS:
            # Between-farm dispersion of this year's rates (stock-safe ruler)
            spread = _robust_std(np.asarray(levels[metric], dtype=float))
            # Typical within-farm year-to-year rate noise (init / early floor)
            noise = float(np.median(noise_stds[metric])) if noise_stds[metric] else 0.0
            # Dispersion of projected trends; same horizon as scoring
            trend_std = _robust_std(np.asarray(proj_trends[metric], dtype=float))

            scales[metric] = 0.0  # centre: rates already self-referenced
            # Early years: noise often binds. Later: spread grows (soilc slowest).
            scales[f"{metric}_std"] = max(spread, noise, 0.01)
            scales[f"{metric}_trend_std"] = max(trend_std, 0.01)

        self.statistic.set("reference_scales", scales)

    def compute_land_scales(self):
        """Cross-farmer spread of each baseline covariate in ``LAND_COVARIATES``.

        The rulers the land-similarity kernel divides peer distances by, so that
        a difference in nitrogen mineralisation and a difference in root-zone
        moisture are comparable before they are summed. They are also what makes
        the kernel self-calibrating: once distances are in units of the
        population's own dispersion, how sharply the kernel discriminates is set
        by the data rather than by a tuned width. Robust spread for the same
        reason the reference scales use one, that a handful of extreme cells
        would otherwise set the ruler for every comparison in the model.

        Baselines are frozen once the pre-coupling period ends, so these are
        constant after the first call. Recomputed alongside the reference scales
        anyway, because the cost is one pass over the farmers and a scale that
        silently went stale would be hard to see in the output.

        Stored in ``world.statistic["land_scales"]``.
        """
        scales = {}
        for name in LAND_COVARIATES:
            values = [
                float(getattr(farmer, name))
                for farmer in self.farmers
                if np.isfinite(getattr(farmer, name, np.nan))
            ]
            spread = _robust_std(np.asarray(values, dtype=float))
            if spread <= 0:
                # A zero spread would send every distance to infinity and make
                # every peer maximally dissimilar. Fall back to the median
                # level, which puts distances in units of the quantity itself.
                median = float(np.median(values)) if values else 0.0
                spread = abs(median) or 1.0
            scales[name] = spread

        self.statistic.set("land_scales", scales)

    def compute_conventional_reference(self):
        """Median soil carbon slope on land that has not changed practice.

        The drift term of the avoided-loss baseline: how fast cropland soil
        carbon is moving where management stayed put, which is climate and
        nothing the farmer chose. A credit written against the farm's own
        pre-adoption slope alone would hand the farmer the climate too, in
        whichever direction it happened to run - and over this century it runs
        against them, so the farm would be paid less for the same practice the
        later it adopted. Differencing against this leaves what the practice
        did.

        Conventional farms only, because the reference has to be a
        counterfactual: a pool that included adopters would have the effect
        being measured sitting inside the thing it is measured against, and
        would drift as adoption spread rather than as the climate did.

        Per country, with a global median as the fallback below
        ``MIN_REFERENCE_FARMS``. Median rather than mean for the same reason
        the reference scales use a robust spread, that a handful of extreme
        cells would otherwise set the rate for everyone.

        Slopes stay in the log-deviation units the tracker reports, so each
        farm converts on its own stock; see
        ``CAFarmer.counterfactual_soilc_accumulation``. A relative reference is
        the only one that transfers between farms holding different stocks.

        Stored in ``world.statistic["conventional_soilc_trend"]``. Read only
        under an avoided-loss ``policy.carbon_price_basis``, but computed
        unconditionally: the cost is one pass over the farmers, and a statistic
        that silently went stale would be hard to see in the output.
        """
        per_country: dict[object, list[float]] = {}
        pooled: list[float] = []

        for farmer in self.farmers:
            behaviour = getattr(farmer, "behaviour", None)
            tracker = getattr(behaviour, "performance_tracker", None)
            if tracker is None or tracker.n < MIN_REFERENCE_YEARS:
                continue
            if behaviour.committed_bundle is not ManagementBundle.conventional:
                continue
            slope = tracker.clean_trend["soilc"]
            if not np.isfinite(slope):
                continue
            code = getattr(getattr(farmer.cell, "country", None), "code", None)
            per_country.setdefault(code, []).append(float(slope))
            pooled.append(float(slope))

        reference = {
            code: float(np.median(slopes))
            for code, slopes in per_country.items()
            if len(slopes) >= MIN_REFERENCE_FARMS
        }
        # Zero rather than absent where no farm qualifies, which is every year
        # before the trackers have three observations: an unknown drift has to
        # read as "nothing is known to have moved" and leave the anchor alone,
        # not send the baseline somewhere arbitrary.
        reference[GLOBAL_REFERENCE] = float(np.median(pooled)) if pooled else 0.0

        self.statistic.set("conventional_soilc_trend", reference)
