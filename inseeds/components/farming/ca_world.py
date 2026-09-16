"""CA-specific World entity holding the reference scales for performance scoring."""

import numpy as np

from inseeds.components.farming.metrics import METRIC_KEYS
from inseeds.components.farming.world import World

# For normally distributed data, std ≈ 1.4826 × MAD
_MAD_SCALE = 1.4826


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
