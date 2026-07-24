"""CA-specific World entity with static reference scales."""

import numpy as np

from inseeds.components.farming.world import World

# Metrics tracked by the performance scoring system. All are scored on the same
# self-referenced, fractional-rate footing (value / own_baseline - 1).
_METRICS = ("yield", "soilc", "moisture", "profit", "leaching")

# MAD scaling factor: for normally distributed data, std ≈ 1.4826 × MAD
_MAD_SCALE = 1.4826


def _robust_std(values: np.ndarray) -> float:
    """Compute robust standard deviation using MAD (Median Absolute Deviation).

    MAD is the most outlier-resistant measure of spread:
    - std: one extreme value can inflate it massively
    - IQR: more robust, but still uses specific quantiles
    - MAD: based on median of absolute deviations from median

    Scaled by 1.4826 so it equals std for normally distributed data,
    but resists outliers (e.g., a few near-bankrupt or windfall farmers).

    Parameters
    ----------
    values : np.ndarray
        Array of values to compute spread for.

    Returns
    -------
    float
        Robust std estimate (MAD × 1.4826). Returns 0.0 if array is empty.
    """
    if values.size == 0:
        return 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(mad * _MAD_SCALE)


class CAWorld(World):
    """World entity with static reference scales for performance scoring.

    Computes world-level statistics from farmer data at initialization.
    These serve as fixed scaling factors for social learning
    (between-farmer comparisons).

    Self-Referenced, Fractional-Rate Scoring
    ----------------------------------------
    Every metric (yield, soilc, moisture, leaching, profit) is expressed
    as a fractional deviation from each farmer's OWN historic baseline:

        rate = value / own_baseline - 1
        score = 1 + rate / scale_level + trend_weight × (trend × horizon) / scale_trend

    This makes all five metrics dimensionless, centred at zero, and comparable
    across farmers regardless of climate, soil, farm size or income level - the
    ambient conditions are divided out by construction, so no country medians or
    divisors are needed.

    Scale Calibration (data-driven, no tuning constant)
    ---------------------------------------------------
    - Level scale = median per-farmer INTERANNUAL std of the rate. The
      cross-farmer spread of self-referenced rates is degenerate at init
      (everyone sits at their own baseline ≈ 0), so it cannot serve as a
      ruler; the year-to-year fluctuation is real and non-zero. A sustained
      deviation is "significant" when it exceeds typical interannual noise -
      one consistent, one-fluctuation-equals-one-unit logic for all metrics.
    - Trend scale = robust MAD of projected trends (rate slope × horizon)
      across farmers.

    Both are floored at 0.01 (1% relative spread) to prevent division by zero.
    """

    def compute_reference_scales(self):
        """Compute static reference scales from initial farmer data.

        Called ONCE during model initialization, after farmers are created.
        Uses the performance trackers (populated from the historic period,
        e.g. 2015-2025) so levels and trends reflect the same data that is
        scored during the coupled run.

        For every metric:
        - Center = 0.0 (fractional rate, by construction)
        - Level std = median of per-farmer interannual rate std (data-driven)
        - Trend std = robust MAD of projected trends (rate slope × horizon)

        Results stored in world.statistic["reference_scales"] as a dict of
        {metric: 0.0, metric_std: ..., metric_trend_std: ...}.
        """
        # Per metric: per-farmer interannual rate std (level ruler) and the
        # projected trend across farmers (trend ruler).
        level_stds = {m: [] for m in _METRICS}
        proj_trends = {m: [] for m in _METRICS}

        for farmer in self.farmers:
            if not (
                hasattr(farmer, "behaviour")
                and hasattr(farmer.behaviour, "performance_tracker")
            ):
                continue

            tracker = farmer.behaviour.performance_tracker
            horizon = farmer.behaviour.get_aft_param("min_observation_years")
            trend = tracker.trend
            std = tracker.level_std

            for metric in _METRICS:
                if std[metric] > 0:
                    level_stds[metric].append(std[metric])
                proj_trends[metric].append(trend[metric] * horizon)

        scales = {}
        for metric in _METRICS:
            level_std = float(np.median(level_stds[metric])) if level_stds[metric] else 0.0
            trend_std = _robust_std(np.asarray(proj_trends[metric], dtype=float))

            scales[metric] = 0.0  # Centered at own baseline (fractional rate)
            scales[f"{metric}_std"] = max(level_std, 0.01)
            scales[f"{metric}_trend_std"] = max(trend_std, 0.01)

        self.statistic.set("reference_scales", scales)
