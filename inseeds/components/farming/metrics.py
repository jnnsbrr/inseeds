"""The declared set of performance metrics farmers score themselves on.

Everything that used to name ``yield``, ``soilc``, ``moisture``, ``leaching``
and ``profit`` one at a time now reads this registry instead. Adding a sixth
metric should mean adding one ``MetricSpec`` and the config weights it names,
not editing the scoring function, three dataclasses and the reference scales.

What a metric is
----------------
A *rate*: a self-referenced deviation from the farm's own historic baseline,
``ln((v + n) / (b + n))`` for the environmental metrics and ``asinh`` for
profit. Rates are dimensionless and centred on zero, so climate, soil and farm
size cancel and the five can be weighed against each other.

That is a different quantity from the physical one a policy instrument would
price. A carbon payment is owed on tonnes of carbon and a nitrogen charge on
kilograms of nitrogen, neither of which can be recovered from a log deviation
without the baseline it was taken against. ``physical_attr`` and
``physical_baseline_attr`` name the pair, so the conversion happens in one place
rather than at every call site.

Every metric is scored on the same quantity it would be priced on. That is a
constraint worth keeping: leaching was once scored as a concentration and
priceable only as a flux, and converting between them silently assumed the two
practices being compared drained alike - which cover crops do not.

See Also
--------
ca_management.compute_performance_score : weighs the metrics into one score
ca_world.CAWorld.compute_reference_scales : builds the rulers they are divided by
"""

from __future__ import annotations

from dataclasses import dataclass


#: Clip bounds applied to per-farmer rates before they enter a region aggregate
#: or a peer comparison (Winsorising). Rates outside this range are capped,
#: which stops one farm with a pathological rate from setting a whole country's
#: social-learning signal. Plus or minus 10 is a thousandfold change under
#: ``log_rate``: extreme but reachable, and beyond it the number is noise.
#:
#: Held per metric because the bound does not mean the same thing for all of
#: them: profit is ``asinh`` rather than a log, where 10 is a deviation of
#: roughly 11000 times baseline revenue rather than 22000, and the bound
#: therefore bites at a different place. All five share one value today, which
#: is recorded here rather than fixed, because changing it moves results.
DEFAULT_CLIP_BOUNDS = (-10.0, 10.0)


@dataclass(frozen=True)
class MetricSpec:
    """One scored metric and the names it is known by across the model.

    Attributes
    ----------
    key : str
        Canonical name, and the key in every ``level``/``trend`` dict.
    rate_attr : str
        Farmer attribute holding this year's rate.
    mean_attr, trend_attr : str
        The unified scoring API: the names a tracker, a remembered experience
        and a region aggregate all answer to, so one scoring path serves all
        three.
    higher_is_better : bool
        False for leaching, which is scored inverted.
    weight_key, trend_weight_key : str
        Config keys under ``aftpar.<aft>``. Spelled out rather than derived,
        because soil carbon's weights are ``weight_soil`` and
        ``trend_weight_soil`` while its metric key is ``soilc``.
    negligible_key : str or None
        Config key of the materiality offset added to value and baseline before
        the log. None for profit, which divides by baseline revenue instead.
    physical_attr : str or None
        Farmer attribute carrying the metric in its own physical unit, where
        one exists. None means there is nothing to price.
    physical_baseline_attr : str or None
        Farmer attribute holding the historic mean of ``physical_attr``. A log
        deviation only becomes an absolute quantity against the baseline it was
        taken about, so an instrument that prices this metric needs both.
    physical_unit : str or None
        Unit of ``physical_attr``, for the reader. The rate is always the log
        deviation of ``physical_attr`` itself, which is what makes converting a
        rate back to an absolute change exact rather than approximate.
    economic : bool
        True for yield and profit, which share one slot in the score: a farm
        weighs its income once, either as physical yield or as money, never
        both. ``tpb.use_profit`` picks which.
    clip_bounds : tuple[float, float]
        See ``DEFAULT_CLIP_BOUNDS``.
    """

    key: str
    rate_attr: str
    mean_attr: str
    trend_attr: str
    higher_is_better: bool
    weight_key: str
    trend_weight_key: str
    negligible_key: str | None
    physical_attr: str | None
    physical_baseline_attr: str | None
    physical_unit: str | None
    economic: bool
    clip_bounds: tuple[float, float] = DEFAULT_CLIP_BOUNDS

    @property
    def priceable(self) -> bool:
        """Whether an instrument could charge or pay on this metric."""
        return self.physical_attr is not None

    @property
    def level_scale_key(self) -> str:
        """Key of this metric's level ruler in ``world.reference_scales``."""
        return f"{self.key}_std"

    @property
    def trend_scale_key(self) -> str:
        """Key of this metric's trend ruler in ``world.reference_scales``."""
        return f"{self.key}_trend_std"


def _spec(
    key,
    *,
    rate_attr,
    weight_key,
    trend_weight_key,
    negligible_key=None,
    physical_attr=None,
    physical_baseline_attr=None,
    physical_unit=None,
    higher_is_better=True,
    economic=False,
):
    """Build a spec, filling in the alias names that follow the convention."""
    return MetricSpec(
        key=key,
        rate_attr=rate_attr,
        mean_attr=f"mean_{key}",
        trend_attr=f"{key}_trend",
        higher_is_better=higher_is_better,
        weight_key=weight_key,
        trend_weight_key=trend_weight_key,
        negligible_key=negligible_key,
        physical_attr=physical_attr,
        physical_baseline_attr=physical_baseline_attr,
        physical_unit=physical_unit,
        economic=economic,
    )


#: The registry. Iteration order is the order the score is summed in, and the
#: two economic metrics come first so that dropping the inactive one leaves the
#: same sequence of additions as before this was a loop.
METRICS: dict[str, MetricSpec] = {
    spec.key: spec
    for spec in (
        _spec(
            "yield",
            rate_attr="yield_rate",
            weight_key="weight_yield",
            trend_weight_key="trend_weight_yield",
            negligible_key="negligible_yield",
            physical_attr="cropyield",
            physical_baseline_attr="baseline_yield",
            physical_unit="gC/m2",
            economic=True,
        ),
        _spec(
            "profit",
            rate_attr="profit",
            weight_key="weight_profit",
            trend_weight_key="trend_weight_profit",
            economic=True,
        ),
        _spec(
            "soilc",
            rate_attr="soilc_rate",
            weight_key="weight_soil",
            trend_weight_key="trend_weight_soil",
            negligible_key="negligible_soilc",
            physical_attr="soilc",
            physical_baseline_attr="baseline_soilc",
            physical_unit="gC/m2",
        ),
        _spec(
            "moisture",
            rate_attr="moisture_rate",
            weight_key="weight_moisture",
            trend_weight_key="trend_weight_moisture",
            negligible_key="negligible_moisture",
            physical_attr="root_moisture",
            physical_baseline_attr="baseline_moisture",
            physical_unit="mm",
        ),
        _spec(
            "leaching",
            rate_attr="leaching_rate",
            weight_key="weight_leaching",
            trend_weight_key="trend_weight_leaching",
            negligible_key="negligible_leaching",
            # The nitrogen that left the field, not its concentration in the
            # water that carried it. A charge on losses is owed a mass, and a
            # mass is what adds up over a catchment.
            # ``farmer.leaching_concentration`` reports the concentration for
            # comparison against a nitrate standard, and nothing decides on it.
            physical_attr="leaching",
            physical_baseline_attr="baseline_leaching",
            physical_unit="gN/m2",
            higher_is_better=False,
        ),
    )
}

#: Metric keys in registry order. What the ``level``, ``trend`` and
#: ``level_std`` dicts are keyed on.
METRIC_KEYS: tuple[str, ...] = tuple(METRICS)


def zero_metrics() -> dict[str, float]:
    """A fresh ``{metric: 0.0}``, the neutral state of a level or trend dict."""
    return {key: 0.0 for key in METRIC_KEYS}


def rates_of(farmer) -> dict[str, float]:
    """This year's rate for every metric, read off the farmer.

    One place that knows which farmer attribute carries which metric, so the
    tracker, the region aggregate and a transition reset cannot disagree about
    it.
    """
    return {key: getattr(farmer, spec.rate_attr) for key, spec in METRICS.items()}


def _metric_property(name: str, key: str, getter):
    """One generated property, closed over its metric key."""
    return property(
        lambda self, _key=key: getter(self, _key),
        doc=f"``{key}`` via the generated ``{name}`` alias. See metrics.METRICS.",
    )


def add_metric_properties(cls, patterns):
    """Give ``cls`` one generated property per metric per name pattern.

    ``patterns`` maps a name pattern containing ``{key}`` to a getter taking
    ``(instance, metric_key)``. The trackers, the remembered experiences and
    the region aggregates all have to answer to the same names for one scoring
    path to serve all three, and hand-writing that came to some fifty near
    identical properties which could drift apart and which had to be extended
    by hand for every new metric.

    A name already defined on the class is left alone, so a metric that needs
    something other than the generated form can still say so.
    """
    for pattern, getter in patterns.items():
        for key in METRIC_KEYS:
            name = pattern.format(key=key)
            if name not in cls.__dict__:
                setattr(cls, name, _metric_property(name, key, getter))
    return cls


def economic_metric(use_profit: bool) -> MetricSpec:
    """The metric filling the economic slot: profit if ``use_profit`` else yield."""
    return METRICS["profit"] if use_profit else METRICS["yield"]


def scored_metrics(use_profit: bool):
    """The metrics that actually reach the score, in the order they are summed.

    The economic slot holds one of yield and profit; the other is still tracked
    and still reported, it just carries no weight.
    """
    active = economic_metric(use_profit).key
    return tuple(
        spec
        for spec in METRICS.values()
        if not spec.economic or spec.key == active
    )
