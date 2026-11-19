"""Weighting a peer's remembered effect by how well its land stands in.

The estimator differences each peer within itself, which takes that peer's own
site out of its record, and then pools across the neighbourhood or country. What
it never asked is whether those sites resemble the farm doing the reading. On
the published runs that shows up as unaided legume adopters sitting 0.22 SD
below the cropland mean on the very difference the legume is chosen for.

``tpb.land_similarity`` is the correction, and it is all or nothing: the
objection is to the estimator rather than to cover crops, so switching it on
reaches every comparison the estimator serves. These tests pin both states, off
having to reproduce the old estimator exactly and on having to move the estimate
toward the peers that resemble the asker.
"""

import numpy as np
import pytest

from inseeds.components.farming.ca_management import ManagementBundle
from inseeds.components.farming.ca_world import LAND_COVARIATES
from inseeds.components.farming.metrics import zero_metrics


FROM_BUNDLE = ManagementBundle.conventional
TO_BUNDLE = ManagementBundle.conservation

#: The near group's remembered gain, and the far group's. The far group is both
#: larger and more numerous, so an unweighted median lands on it and any move to
#: the near value can only have come from the weighting.
NEAR_GAIN = 0.02
FAR_GAIN = 0.20

#: Two near peers against four far ones. Deliberately unbalanced: with equal
#: counts the weighted median sits on the near value either way and the test
#: would pass without the weighting doing anything.
N_NEAR = 2
N_FAR = 4

#: Nitrogen mineralisation, gN/m2/yr, and the ruler the kernel divides by.
#: The scale is imposed rather than measured off the fixture so the distance
#: between the groups is exact: there is no bandwidth left to turn, so a test
#: that wants the kernel to discriminate has to place the groups far enough
#: apart in spreads, and that is only arithmetic if the spread is known.
NEAR_NMIN = 5.0
FAR_NMIN = 20.0
NMIN_SCALE = 3.0
MOISTURE_SCALE = 50.0
YIELD_SCALE = 80.0


def _set_scales(model):
    """Impose the rulers, so peer distance in spreads is known exactly."""
    model.world.statistic.set(
        "land_scales",
        {
            "baseline_nmineralization": NMIN_SCALE,
            "baseline_moisture": MOISTURE_SCALE,
            "baseline_yield": YIELD_SCALE,
        },
    )


def _flatten_land(farmer, nmin, moisture=500.0, cropyield=400.0):
    """Put a farm's baseline covariates at known values.

    All three are set on every farm in these tests, not only the one being
    varied, because the kernel averages over the covariates: a difference left
    to the fixture in moisture or yield would dilute the nitrogen contrast by
    an amount that depends on which cells the fixture happens to hold.
    """
    farmer.baseline_nmineralization = nmin
    farmer.baseline_moisture = moisture
    farmer.baseline_yield = cropyield


def _remember(farmer, year, gain):
    """Give one farm a record of both bundles, differing by ``gain`` in yield."""
    memory = farmer.behaviour.performance_memory
    for bundle, weight in ((FROM_BUNDLE, 0.0), (TO_BUNDLE, 1.0)):
        level = zero_metrics()
        level["yield"] = weight * gain
        memory.record_performance(bundle, zero_metrics(), level, n=5, year=year)


@pytest.fixture
def split_neighbourhood(ca_model_builder):
    """An asker whose neighbours fall into a near group and a far group.

    The near group shares the asker's nitrogen mineralisation and remembers a
    small gain; the far group sits five population spreads away and remembers a
    large one. Every other covariate is held equal, so distance in the kernel is
    nitrogen alone.
    """
    model = ca_model_builder(seed=20261005)
    farmers = list(model.world.farmers)
    if len(farmers) < 1 + N_NEAR + N_FAR:
        pytest.skip("fixture has too few farmers to split a neighbourhood")

    year = model.lpjml.sim_year
    asker, rest = farmers[0], farmers[1:]
    near = rest[:N_NEAR]
    far = rest[N_NEAR : N_NEAR + N_FAR]

    _flatten_land(asker, nmin=NEAR_NMIN)
    for peer in near:
        _flatten_land(peer, nmin=NEAR_NMIN)
        _remember(peer, year, NEAR_GAIN)
    for peer in far:
        _flatten_land(peer, nmin=FAR_NMIN)
        _remember(peer, year, FAR_GAIN)

    # Everyone else is out of the pool: no record of either bundle, and out of
    # the neighbourhood, so the only peers are the ones placed above.
    asker.neighbourhood = list(near) + list(far)
    _set_scales(model)

    return model, asker


def _settings(model):
    return model.config.coupled_config.tpb.land_similarity


def _ask(asker):
    return asker.behaviour.anticipated_change("yield", FROM_BUNDLE, TO_BUNDLE)


def _bundle_cache_keys(asker):
    """Cache keys for the bundle pair under test, as the asker resolves them.

    The cache cannot simply be read whole. Resolving a memory key asks
    ``indicate_cover_crop_type``, which runs an estimate of its own between the
    two cover crops, so any call here leaves a second, unrelated entry behind.
    """
    memory = asker.behaviour.performance_memory
    current, target = memory.key(FROM_BUNDLE), memory.key(TO_BUNDLE)
    cache = getattr(asker.cell.country, "_effect_cache", {})
    return [
        key
        for key in cache
        if key[1] == current and key[2] == target
    ]


class TestTheSetting:
    """What the flag is, before what the kernel does."""

    def test_the_shipped_config_has_it_on(self, ca_model_builder):
        """Runs from this tree carry the correction; earlier ones do not."""
        model = ca_model_builder(seed=1)

        assert _settings(model).enabled is True

    def test_off_means_no_similarity_vector_is_built(self, split_neighbourhood):
        """Off takes the unweighted path rather than a path of uniform weights.

        A vector of ones would reduce to the same number here, but not in every
        case the weighted median can see, so the two are kept distinct.
        """
        model, asker = split_neighbourhood
        _settings(model).enabled = False

        assert asker.behaviour._land_similarity_config() is None
        assert asker.behaviour._similarity_weights([asker], None) is None

    def test_a_config_without_the_block_still_runs(self, split_neighbourhood):
        """An older configuration must keep working and keep its numbers."""
        model, asker = split_neighbourhood
        tpb = model.config.coupled_config.tpb
        saved = tpb.land_similarity
        del tpb.land_similarity
        try:
            assert asker.behaviour._land_similarity_config() is None
            assert _ask(asker) == pytest.approx(np.expm1(FAR_GAIN))
        finally:
            tpb.land_similarity = saved


class TestTheKernel:
    """``land_similarity`` on its own, away from the estimator."""

    def test_an_identical_farm_scores_one(self, split_neighbourhood):
        model, asker = split_neighbourhood
        settings = _settings(model)

        assert asker.behaviour.land_similarity(asker, settings) == pytest.approx(1.0)

    def test_it_falls_as_the_land_diverges(self, split_neighbourhood):
        """Monotone in distance, which is the whole content of the weighting."""
        model, asker = split_neighbourhood
        settings = _settings(model)
        peer = list(model.world.farmers)[1]

        scores = []
        for nmin in (NEAR_NMIN, 10.0, 20.0, 40.0):
            _flatten_land(peer, nmin=nmin)
            scores.append(asker.behaviour.land_similarity(peer, settings))

        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[-1]

    def test_one_spread_away_is_halfway_discounted(self, split_neighbourhood):
        """Where the width comes from now that there is no width parameter.

        Distances are in units of the population's own robust spread, so a peer
        one spread away in every covariate sits at half weight before the floor
        is applied. That is the self-calibration: nothing was tuned to put it
        there.
        """
        model, asker = split_neighbourhood
        settings = _settings(model)
        settings.floor = 0.0
        peer = list(model.world.farmers)[1]
        _flatten_land(
            peer,
            nmin=NEAR_NMIN + NMIN_SCALE,
            moisture=500.0 + MOISTURE_SCALE,
            cropyield=400.0 + YIELD_SCALE,
        )

        assert asker.behaviour.land_similarity(peer, settings) == pytest.approx(0.5)

    def test_the_floor_is_a_floor(self, split_neighbourhood):
        """A remote peer is downweighted, never discarded."""
        model, asker = split_neighbourhood
        settings = _settings(model)
        peer = list(model.world.farmers)[1]
        _flatten_land(peer, nmin=1e9)

        score = asker.behaviour.land_similarity(peer, settings)

        assert score == pytest.approx(settings.floor, abs=1e-6)
        assert score > 0.0

    def test_it_is_uniform_before_the_scales_exist(self, ca_model_builder):
        """Mid-initialisation there is no ruler, and guessing one is worse.

        Farmers are built before the first ``compute_land_scales``, so anything
        asking this question in between has to get the unweighted answer.
        """
        model = ca_model_builder(seed=3)
        asker, peer = list(model.world.farmers)[:2]
        model.world.statistic.set("land_scales", {})

        assert asker.behaviour.land_similarity(peer, _settings(model)) == 1.0

    def test_a_covariate_the_farm_lacks_is_skipped(self, split_neighbourhood):
        """Output written without nitrogen mineralisation still has to run."""
        model, asker = split_neighbourhood
        settings = _settings(model)
        peer = list(model.world.farmers)[1]
        _flatten_land(peer, nmin=FAR_NMIN)
        for name in LAND_COVARIATES:
            delattr(peer, name)

        assert asker.behaviour.land_similarity(peer, settings) == 1.0

    def test_an_absent_covariate_reads_as_nan_and_not_as_distance(
        self, split_neighbourhood
    ):
        """A missing output must not look like land that differs.

        ``n_mineralization`` returns NaN where the variable is absent, and the
        kernel has to drop that covariate rather than let NaN propagate into a
        weight. Here the remaining two covariates still match, so an identical
        farm has to keep scoring one.
        """
        model, asker = split_neighbourhood
        settings = _settings(model)
        peer = list(model.world.farmers)[1]
        _flatten_land(peer, nmin=float("nan"))
        asker.baseline_nmineralization = float("nan")

        assert asker.behaviour.land_similarity(peer, settings) == pytest.approx(1.0)


class TestTheEstimate:
    """The point of the exercise: which peers the answer comes from."""

    def test_off_reads_the_larger_distant_gain(self, split_neighbourhood):
        """Four distant farms outvote two near ones, which is the bug."""
        model, asker = split_neighbourhood
        _settings(model).enabled = False

        assert _ask(asker) == pytest.approx(np.expm1(FAR_GAIN))

    def test_on_reads_the_land_that_resembles_the_asker(self, split_neighbourhood):
        """Two near farms outweigh four distant ones once distance counts."""
        model, asker = split_neighbourhood

        assert _ask(asker) == pytest.approx(np.expm1(NEAR_GAIN))

    def test_a_floor_of_one_stops_discriminating(self, split_neighbourhood):
        """The setting has to be able to do nothing, or it cannot be calibrated.

        With the bandwidth gone, ``floor`` is the only knob and it has to span
        the whole range on its own: at 1 every peer scores the same and the
        unweighted estimate comes back.
        """
        model, asker = split_neighbourhood
        _settings(model).floor = 1.0

        assert _ask(asker) == pytest.approx(np.expm1(FAR_GAIN))

    def test_land_a_little_different_is_left_alone(self, split_neighbourhood):
        """The default floor is meant to bite on distant land and nothing else.

        A peer under a spread away should not lose a majority it would have
        held, or the correction would be rewriting ordinary comparisons rather
        than the ones it was built for.
        """
        model, asker = split_neighbourhood
        for peer in asker.neighbourhood:
            if peer.baseline_nmineralization == FAR_NMIN:
                _flatten_land(peer, nmin=NEAR_NMIN + 0.5 * NMIN_SCALE)

        assert _ask(asker) == pytest.approx(np.expm1(FAR_GAIN))

    def test_it_does_not_change_which_level_answers(self, split_neighbourhood):
        """Weighting reweights the pool; it must not shrink it.

        ``min_peers`` is counted on the collected peers, before similarity is
        applied, so switching the flag on cannot push a comparison from the
        local level down to the country one and change what it reads for
        reasons that have nothing to do with land.
        """
        model, asker = split_neighbourhood
        country = asker.cell.country
        _settings(model).floor = 0.0  # every distant peer free to go to zero
        country._effect_cache = {}

        answer = _ask(asker)

        assert answer == pytest.approx(np.expm1(NEAR_GAIN))
        assert not _bundle_cache_keys(asker), (
            "the local level answered, so the country scan should never have "
            "run for this bundle pair"
        )
        assert country._effect_cache, (
            "sanity: the cover-crop comparison inside the call does cache, so "
            "an empty cache would mean this test is not exercising the path"
        )


class TestTheCountryLevel:
    """The cached path, which is the one that has to stay cheap."""

    @staticmethod
    def _go_country(asker):
        asker.neighbourhood = []
        asker.cell.country._effect_cache = {}

    def test_the_scan_is_still_shared(self, split_neighbourhood):
        """What is cached is the peers, not the answer, so the scan runs once."""
        model, asker = split_neighbourhood
        self._go_country(asker)
        country = asker.cell.country

        first = _ask(asker)
        entries_after_first = dict(country._effect_cache)
        second = _ask(asker)

        assert first == second
        assert list(country._effect_cache) == list(entries_after_first), (
            "a second read re-scanned the country instead of reusing the pool"
        )

    def test_the_cached_pool_carries_its_peers(self, split_neighbourhood):
        """The entry has to hold the peers, or similarity cannot be applied."""
        model, asker = split_neighbourhood
        self._go_country(asker)

        _ask(asker)
        (key,) = _bundle_cache_keys(asker)
        index, deltas, weights, peers, _reduced = (
            asker.cell.country._effect_cache[key]
        )

        assert len(peers) == len(deltas) == len(weights) == len(index)
        assert len(peers) == N_NEAR + N_FAR
        assert all(hasattr(peer, "behaviour") for peer in peers)

    def test_two_askers_on_one_pool_can_differ(self, split_neighbourhood):
        """The shared scan must not hand both farms the same answer.

        An asker sitting in the far group should read the far gain from the very
        pool that gives the near gain to an asker in the near group. If the
        cached median were reused, both would read the same number and the
        weighting would be silently inert at the country level.
        """
        model, asker = split_neighbourhood
        _settings(model).floor = 0.0

        farmers = list(model.world.farmers)
        other = farmers[1 + N_NEAR + N_FAR]
        _flatten_land(other, nmin=FAR_NMIN)  # stands with the far group
        other.neighbourhood = []
        self._go_country(asker)

        near_view = _ask(asker)
        far_view = _ask(other)

        assert near_view == pytest.approx(np.expm1(NEAR_GAIN))
        assert far_view == pytest.approx(np.expm1(FAR_GAIN))
