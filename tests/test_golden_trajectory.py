"""Golden trajectory: pins the numbers the CA scoring machinery produces.

Why this exists
---------------
The rest of the suite asserts structure and invariants - that a score is a
float, that capital moves by the year's profit. None of it would notice if a
refactor of the scoring plumbing shifted every farm's score by a percent, which
is exactly the failure mode of moving five hard-coded metrics onto a declared
metric set. This test records the actual numbers of a short run and fails on any
change to them.

It is a scaffold, not a specification. When a step is meant to change behaviour,
re-record with ``INSEEDS_REGOLD=1 pytest tests/test_golden_trajectory.py`` and
say so in the commit message. A silent re-record defeats the purpose.

What it pins, and in what order
-------------------------------
The reference scales are asserted before the per-farm state. They are the
rulers every score is divided by, they are rebuilt from every farmer's tracker
once a year, and ``compute_reference_scales`` reaches into the tracker by
attribute name (``getattr(tracker, f"mean_{metric}")``) and by dict key. A
change in which observations reach the tracker, or in the order they are
accumulated, shows up in the scales first. Failing on them first says *why*
something moved; failing on a farm's bundle sequence only says *that* it did.

The synthetic earth
-------------------
In test mode ``update_lpjml`` neither refreshes ``from_earth`` nor advances the
coupler's simulation year, both of which LPJmL would normally do. An unmodified
run therefore holds every metric at its baseline and, because the tracker skips
an observation it has already recorded for the year, never gets past its first
one: all rates are zero, every scale sits on its 0.01 floor, and no farm ever
decides anything. A trajectory recorded from that pins almost nothing.

So the harness plays the part of LPJmL, advancing the year and writing a
deterministic response into ``from_earth``. It is not meant to be physically
right - only to be reproducible and to move every path the refactor touches:
rates away from zero and apart across farms, scales off their floors, and
enough of a management signal that peers differ by bundle.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from inseeds.components.farming.ca_management import ManagementBundle


#: Written by ``INSEEDS_REGOLD=1``; otherwise read and compared against.
GOLDEN_FILE = Path(__file__).parent / "data" / "golden_trajectory.json"

#: Seeds the model's own draws (AFT assignment, observation offsets).
MODEL_SEED = 20240617

#: Separate stream for the synthetic earth's per-cell wobble. Kept off the
#: global RNG so the harness cannot shift the draws the model itself makes.
EARTH_SEED = 987654321

#: The pickled fixture carries root moisture at 0.3, against a materiality
#: offset of 10 mm in the config. ``log_rate`` adds that offset to both value and
#: baseline, so at this magnitude the offset is the whole quantity and moisture
#: cannot move however hard the harness pushes it - its ruler would stay on its
#: floor and a mix-up between it and another metric would be invisible. Scaling
#: it to a field-plausible few hundred millimetres, in the historic read as well
#: as the runtime one so the baseline moves with it, restores the metric.
MOISTURE_TO_MM = 600.0

#: Bundle pairs the anticipated-change estimator is probed on each year. Chosen
#: to span the three practice dimensions one at a time plus the full switch.
PROBE_PAIRS = (
    ("conventional", "notill"),
    ("conventional", "conservation"),
    ("notill_residue", "conservation"),
    ("conservation", "notill_residue"),
)


def scale_root_moisture(output):
    """Put the fixture's root moisture on a scale where it can carry a signal."""
    output["rootmoist_agr"].values[...] *= MOISTURE_TO_MM
    return output


class SyntheticEarth:
    """Deterministic stand-in for LPJmL's yearly response to management.

    Writes into ``world._from_earth_data`` in place, which the cell views alias,
    exactly as ``update_lpjml`` does outside test mode.

    The response is crude on purpose: soil carbon accumulates faster under the
    conserving practices, the cover crop costs a little yield and raises
    leaching, residue holds moisture. Those are the directions the real coupling
    shows, and having them present means peers with different bundles carry
    different remembered levels, which is what the anticipated-change estimator
    reads.
    """

    def __init__(self, model, farmers, seed=EARTH_SEED):
        self._lpjml = model.lpjml
        self._data = model.world._from_earth_data
        self._farmers = farmers
        self._rng = np.random.default_rng(seed)
        # One fixed offset per cell, so farms start out spread rather than
        # identical. A degenerate spread would leave every level ruler on its
        # floor and hide any change in how the spread is computed.
        self._cell_offset = self._rng.normal(0.0, 0.05, size=len(farmers))
        # Soil carbon and root moisture need a spread of their own. Both respond
        # to management the same way in every cell, so without this they move
        # identically across farms, the cross-farm spread stays zero and their
        # rulers never leave the floor - which is the one case where a change in
        # how the spread is computed would go unseen.
        self._soil_rate = self._rng.uniform(0.2, 2.2, size=len(farmers))
        self._moisture_offset = self._rng.normal(0.0, 0.04, size=len(farmers))

    def advance(self, year, year_index):
        """Move to ``year`` and apply its response to the practices in force."""
        # The coupler advances this off the back of the LPJmL exchange, which
        # test mode skips. Left alone, every tracker refuses the year's
        # observation as one it has already seen and the run stands still.
        self._lpjml._sim_year = year

        wobble = self._rng.normal(0.0, 0.02, size=len(self._farmers))

        for i, farmer in enumerate(self._farmers):
            cell = farmer.cell.cell_index
            notill = 1.0 - float(farmer.tillage)
            cover = 1.0 if float(farmer.cover_crop) > 0 else 0.0
            residue = float(farmer.residue_on_field)
            shake = 1.0 + self._cell_offset[i] + wobble[i]

            # Soil carbon integrates, so it is the metric whose deviation from
            # baseline outgrows its year-to-year noise - the case the level
            # ruler has to handle.
            gain = 1.0 + self._soil_rate[i] * (
                0.010 * (notill + cover + residue) + 0.002 * year_index
            )
            self._data["soilc_agr_layer"].values[cell, 0, -1] *= gain

            # Yield: a standing cover crop costs a little, accumulated carbon
            # pays back slowly.
            self._data["pft_harvestc"].values[cell, :, -1] *= (
                1.0 - 0.02 * cover + 0.006 * year_index * notill
            ) * shake

            self._data["rootmoist_agr"].values[cell, -1] *= (
                1.0 + 0.01 * residue - 0.004 * cover + self._moisture_offset[i]
            ) * shake

            # Leaching rises with the cover crop in this configuration; runoff
            # is left alone so the concentration moves with the flux.
            self._data["leaching"].values[cell, -1] *= (
                1.0 + 0.03 * cover - 0.01 * notill
            ) * shake

            self._data["litcover_agr"].values[cell, -1] = min(
                0.95, 0.2 + 0.6 * residue
            )


def _round_trip(value):
    """Plain Python float, so the JSON comparison is exact rather than dtype-wise."""
    return float(value)


def _farmer_record(farmer):
    tracker = farmer.behaviour.performance_tracker
    level = tracker.level
    trend = tracker.trend
    return {
        "aft": farmer.aft.name,
        "bundle": farmer.behaviour.practice_bundle.key,
        "own_score": _round_trip(farmer.behaviour.own_score()),
        "capital": _round_trip(farmer.capital),
        "n": int(tracker.n),
        "level": {k: _round_trip(v) for k, v in sorted(level.items())},
        "trend": {k: _round_trip(v) for k, v in sorted(trend.items())},
    }


def _region_record(store):
    """Per-bundle aggregates of one country's published store."""
    record = {
        "total_farmers": _round_trip(store.total_farmers),
        "bundles": {},
    }
    for bundle, performance in sorted(store.items(), key=lambda kv: kv[0].key):
        record["bundles"][bundle.key] = {
            "count": int(performance.count),
            "adoption": _round_trip(store.bundle_counts.get(bundle, 0.0)),
            "mean_yield": _round_trip(performance.mean_yield),
            "mean_soilc": _round_trip(performance.mean_soilc),
            "mean_moisture": _round_trip(performance.mean_moisture),
            "mean_profit": _round_trip(performance.mean_profit),
            "mean_leaching": _round_trip(performance.mean_leaching),
            "yield_trend": _round_trip(performance.yield_trend),
            "soilc_trend": _round_trip(performance.soilc_trend),
            "leaching_trend": _round_trip(performance.leaching_trend),
        }
    return record


def _anticipation_record(farmer):
    """Probe the peer-learning estimator on a fixed set of bundle pairs.

    Reads through the country effect cache, so a cache key that stops
    distinguishing what it should shows up here.
    """
    behaviour = farmer.behaviour
    return {
        f"{current}->{target}": _round_trip(
            behaviour.anticipated_relative_yield_change(
                ManagementBundle[current],
                ManagementBundle[target],
            )
        )
        for current, target in PROBE_PAIRS
    }


#: Metric keys the tracker and memory dicts are built on.
METRIC_KEYS = ("soilc", "moisture", "yield", "profit", "leaching")

#: Bundles the peer probe writes made-up records of.
PROBE_MEMORY_BUNDLES = ("conventional", "conservation")


def _seeded_peer_probe(model, farmers):
    """Read the anticipated-change estimator on records we control.

    Eight years is not enough for a farm to run two bundles long enough to
    remember both, so in the run itself no peer ever clears the dual-experience
    gate and the estimator returns zero for every pair. Zero pins nothing. Here
    every farm is given a made-up record of two bundles, differing per farm, and
    the estimator is read on it.

    This is the machinery the metric generalisation and the cache key both
    touch: the dual-experience gate, the within-peer differencing, the
    experience weighting, the weighted median, the self-exclusion at country
    level and the country cache. Run last, because it overwrites memories.
    """
    year = model.lpjml.sim_year

    for i, farmer in enumerate(farmers):
        for bundle_key in PROBE_MEMORY_BUNDLES:
            # Levels differ by farm and by bundle, so the within-peer difference
            # is neither constant nor zero; durations differ so the weighted
            # median is not just the middle element.
            better = 0.04 + 0.002 * i if bundle_key == "conservation" else 0.0
            level = {
                metric: 0.01 * i + better + 0.001 * METRIC_KEYS.index(metric)
                for metric in METRIC_KEYS
            }
            trend = {metric: 0.001 * (i % 4) for metric in METRIC_KEYS}
            farmer.behaviour.performance_memory.record_performance(
                ManagementBundle[bundle_key],
                trend,
                level,
                n=3 + (i % 5),
                year=year,
            )

    # The run left a cache keyed to this same year, and it was built before
    # these memories existed.
    for country in model.world.countries:
        country._effect_cache = {}

    probes = {}
    for position in (0, 5, 11):
        farmer = farmers[position]
        for current, target in (
            PROBE_MEMORY_BUNDLES,
            tuple(reversed(PROBE_MEMORY_BUNDLES)),
        ):
            probes[f"farmer{position}:{current}->{target}"] = _round_trip(
                farmer.behaviour.anticipated_relative_yield_change(
                    ManagementBundle[current], ManagementBundle[target]
                )
            )
    return probes


def record_trajectory(model):
    """Step the model over its sim years, recording the state of each one."""
    farmers = list(model.world.farmers)
    earth = SyntheticEarth(model, farmers)

    trajectory = {"scales": {}, "farmers": {}, "regions": {}, "anticipation": {}}

    for year_index, year in enumerate(list(model.lpjml.get_sim_years())):
        earth.advance(year, year_index)
        model.update(year)

        key = str(year)
        scales = model.world.statistic.get("reference_scales")
        trajectory["scales"][key] = {
            name: _round_trip(value) for name, value in sorted(scales.items())
        }
        trajectory["farmers"][key] = [_farmer_record(f) for f in farmers]

        regions = {}
        for country in sorted(model.world.countries, key=lambda c: c.code):
            store = country.statistic.get("management_performance")
            if store is not None:
                regions[country.code] = _region_record(store)
        trajectory["regions"][key] = regions

        trajectory["anticipation"][key] = _anticipation_record(farmers[0])

    trajectory["peer_probe"] = _seeded_peer_probe(model, farmers)

    return trajectory


def _first_difference(recorded, expected, path=""):
    """Return a readable description of the first mismatch, or None."""
    if isinstance(expected, dict):
        if not isinstance(recorded, dict):
            return f"{path}: expected a mapping, recorded {type(recorded).__name__}"
        for name in expected:
            if name not in recorded:
                return f"{path}/{name}: missing from the recorded run"
            difference = _first_difference(
                recorded[name], expected[name], f"{path}/{name}"
            )
            if difference:
                return difference
        for name in recorded:
            if name not in expected:
                return f"{path}/{name}: not in the golden file"
        return None

    if isinstance(expected, list):
        if len(recorded) != len(expected):
            return (
                f"{path}: {len(recorded)} entries recorded, "
                f"{len(expected)} in the golden file"
            )
        for i, (got, want) in enumerate(zip(recorded, expected)):
            difference = _first_difference(got, want, f"{path}[{i}]")
            if difference:
                return difference
        return None

    if recorded != expected:
        return f"{path}: recorded {recorded!r}, golden {expected!r}"
    return None


@pytest.fixture(scope="module")
def recorded_trajectory(ca_model_builder):
    """One seeded run, shared by the assertions below."""
    model = ca_model_builder(
        seed=MODEL_SEED, output_transform=scale_root_moisture
    )
    return record_trajectory(model)


@pytest.fixture(scope="module")
def golden_trajectory(recorded_trajectory):
    """The stored trajectory, written first if regeneration was asked for."""
    if os.environ.get("INSEEDS_REGOLD"):
        GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN_FILE, "w") as handle:
            json.dump(recorded_trajectory, handle, indent=1, sort_keys=True)

    if not GOLDEN_FILE.exists():
        pytest.fail(
            f"No golden trajectory at {GOLDEN_FILE}. Record one with "
            "INSEEDS_REGOLD=1 pytest tests/test_golden_trajectory.py"
        )

    with open(GOLDEN_FILE) as handle:
        return json.load(handle)


def test_the_run_is_worth_pinning(recorded_trajectory):
    """Guard the harness itself: a degenerate run would pin nothing.

    If the synthetic earth ever stops biting - a renamed output, a changed
    fixture - every rate collapses to zero and every scale to its floor, and the
    comparisons below would pass no matter what the refactor did to the scoring.
    """
    scales = recorded_trajectory["scales"]
    years = sorted(scales, key=int)
    last = scales[years[-1]]

    on_the_floor = sorted(
        name
        for name, value in last.items()
        if name.endswith("_std") and value <= 0.01
    )
    assert not on_the_floor, (
        f"These reference scales sit on their 0.01 floor: {on_the_floor}. A "
        "metric pinned at the floor carries no spread, so mixing it up with "
        "another metric would not show in the comparison."
    )

    # Distinct values per metric, so a registry that maps a metric to the wrong
    # ruler cannot pass by returning a number that happens to be right.
    level_rulers = {
        value for name, value in last.items() if name.endswith("_std")
    }
    assert len(level_rulers) == len(
        [name for name in last if name.endswith("_std")]
    ), "Two reference scales are numerically identical, so a swap between them would not show."

    levels = [
        entry["level"]["soilc"] for entry in recorded_trajectory["farmers"][years[-1]]
    ]
    assert any(abs(level) > 1e-6 for level in levels), (
        "No farm moved off its baseline, so the trackers pin nothing."
    )

    moved = [
        position
        for position in range(len(recorded_trajectory["farmers"][years[0]]))
        if len(
            {recorded_trajectory["farmers"][y][position]["bundle"] for y in years}
        )
        > 1
    ]
    assert moved, (
        "No farm ever changed bundle, so nothing pins the transition path - "
        "the memory write and the tracker reset that follow a switch."
    )


def test_reference_scales_are_unchanged(recorded_trajectory, golden_trajectory):
    """The rulers first: they explain any movement in the scores below."""
    difference = _first_difference(
        recorded_trajectory["scales"], golden_trajectory["scales"], "scales"
    )
    assert difference is None, (
        f"Reference scales moved. {difference}\n"
        "The scales are rebuilt each year from every farmer's tracker, so this "
        "usually means an accumulator changed, not the scoring itself."
    )


def test_region_aggregates_are_unchanged(recorded_trajectory, golden_trajectory):
    """Country stores next: they sit between the trackers and the scores."""
    difference = _first_difference(
        recorded_trajectory["regions"], golden_trajectory["regions"], "regions"
    )
    assert difference is None, f"Country aggregates moved. {difference}"


def test_anticipated_change_is_unchanged(recorded_trajectory, golden_trajectory):
    """The peer-learning estimator and the cache it reads through."""
    difference = _first_difference(
        recorded_trajectory["anticipation"],
        golden_trajectory["anticipation"],
        "anticipation",
    )
    assert difference is None, f"Anticipated change moved. {difference}"

    difference = _first_difference(
        recorded_trajectory["peer_probe"], golden_trajectory["peer_probe"], "peer_probe"
    )
    assert difference is None, (
        f"The estimator read differently on controlled records. {difference}"
    )


def test_the_peer_probe_reaches_the_estimator(recorded_trajectory):
    """Guard the probe: all-zero answers would mean the gate is still shutting it out."""
    probes = recorded_trajectory["peer_probe"]
    assert any(value != 0.0 for value in probes.values()), (
        "Every probe returned zero, so no peer cleared the dual-experience "
        "gate and the estimator is not being exercised."
    )


def test_farmer_trajectory_is_unchanged(recorded_trajectory, golden_trajectory):
    """Finally the farms: scores, capital and the bundles they chose."""
    difference = _first_difference(
        recorded_trajectory["farmers"], golden_trajectory["farmers"], "farmers"
    )
    assert difference is None, f"Farm trajectory moved. {difference}"
