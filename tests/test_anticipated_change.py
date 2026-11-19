"""The peer estimator, once it answers for more than one metric.

Anticipated change used to exist only for yield, and the country-level scan was
cached on the bundle pair alone. Now that the same estimator has to serve a
carbon payment and a nitrogen tax as well, that key is a trap: the second metric
asked for on a pair would be handed the first one's answer, silently and
plausibly.
"""

import numpy as np
import pytest

from inseeds.components.farming.ca_management import ManagementBundle
from inseeds.components.farming.farmer import agricultural_leaching, log_rate
from inseeds.components.farming.metrics import METRIC_KEYS, METRICS, zero_metrics


#: Two bundles that differ in every practice, so nothing about the pair is
#: degenerate.
FROM_BUNDLE = ManagementBundle.conventional
TO_BUNDLE = ManagementBundle.conservation


def seed_memories(farmers, year, gains):
    """Give every farm a record of both bundles, differing by ``gains``.

    In a short run no farm accumulates two bundles of its own, so the estimator
    is never reached with real memories. Written records are the only way to
    read it.
    """
    for farmer in farmers:
        memory = farmer.behaviour.performance_memory
        for bundle, weight in ((FROM_BUNDLE, 0.0), (TO_BUNDLE, 1.0)):
            level = zero_metrics()
            for metric, gain in gains.items():
                level[metric] = weight * gain
            memory.record_performance(
                bundle, zero_metrics(), level, n=5, year=year
            )


@pytest.fixture
def seeded_model(ca_model_builder):
    """A private model whose farms all remember both bundles.

    Yield improves on the switch and soil carbon worsens, by different amounts.
    Opposite signs, so an answer that came from the wrong metric cannot be
    mistaken for the right one.
    """
    model = ca_model_builder(seed=4242)
    farmers = list(model.world.farmers)
    seed_memories(
        farmers,
        model.lpjml.sim_year,
        {"yield": 0.10, "soilc": -0.25, "leaching": 0.4},
    )
    return model, farmers


def test_each_metric_gets_its_own_answer(seeded_model):
    """The headline: two metrics on one bundle pair do not share a value."""
    model, farmers = seeded_model
    behaviour = farmers[0].behaviour

    yield_change = behaviour.anticipated_change("yield", FROM_BUNDLE, TO_BUNDLE)
    soilc_change = behaviour.anticipated_change("soilc", FROM_BUNDLE, TO_BUNDLE)
    leaching_change = behaviour.anticipated_change("leaching", FROM_BUNDLE, TO_BUNDLE)

    assert yield_change == pytest.approx(np.expm1(0.10))
    assert soilc_change == pytest.approx(np.expm1(-0.25))
    assert leaching_change == pytest.approx(np.expm1(0.4))


def test_the_country_cache_is_keyed_on_the_metric(seeded_model):
    """Same pair, three metrics, three cache entries rather than one."""
    model, farmers = seeded_model
    farmer = farmers[0]
    country = farmer.cell.country
    country._effect_cache = {}

    # Emptying the neighbourhood is what sends the estimate to the country
    # level, which is the only level that caches. With local peers present it
    # answers there and the cache is never consulted.
    farmer.neighbourhood = []

    answers = {
        metric: farmer.behaviour.anticipated_change(metric, FROM_BUNDLE, TO_BUNDLE)
        for metric in ("yield", "soilc", "leaching")
    }

    cached_metrics = {key[0] for key in country._effect_cache}
    assert cached_metrics == {"yield", "soilc", "leaching"}, (
        "The country scan is shared between metrics, so whichever was asked "
        "for first would answer for all of them."
    )
    assert answers["yield"] == pytest.approx(np.expm1(0.10))
    assert answers["soilc"] == pytest.approx(np.expm1(-0.25))


def test_a_second_read_returns_the_cached_value(seeded_model):
    """The cache still caches: asking twice does not rescan or drift."""
    model, farmers = seeded_model
    behaviour = farmers[0].behaviour

    first = behaviour.anticipated_change("soilc", FROM_BUNDLE, TO_BUNDLE)
    second = behaviour.anticipated_change("soilc", FROM_BUNDLE, TO_BUNDLE)

    assert first == second


def test_the_yield_wrapper_is_the_generic_call(seeded_model):
    """``anticipated_relative_yield_change`` is the yield case, nothing more."""
    model, farmers = seeded_model
    behaviour = farmers[0].behaviour

    assert behaviour.anticipated_relative_yield_change(
        FROM_BUNDLE, TO_BUNDLE
    ) == behaviour.anticipated_change("yield", FROM_BUNDLE, TO_BUNDLE)


def test_switching_back_reverses_the_sign(seeded_model):
    """A gain one way is a loss the other, since the estimate is a difference."""
    model, farmers = seeded_model
    behaviour = farmers[0].behaviour

    there = behaviour.anticipated_change("yield", FROM_BUNDLE, TO_BUNDLE)
    back = behaviour.anticipated_change("yield", TO_BUNDLE, FROM_BUNDLE)

    # Fractions, not log ratios, so they are reciprocal rather than opposite.
    assert (1.0 + there) * (1.0 + back) == pytest.approx(1.0)


def test_a_metric_nobody_differs_on_reads_zero(seeded_model):
    """Metrics left flat in the seeded records give no signal, not a stale one."""
    model, farmers = seeded_model
    behaviour = farmers[0].behaviour

    untouched = set(METRIC_KEYS) - {"yield", "soilc", "leaching"}
    for metric in untouched:
        assert behaviour.anticipated_change(metric, FROM_BUNDLE, TO_BUNDLE) == 0.0


def test_no_peers_means_no_estimate(ca_model_builder):
    """With nothing remembered the estimator declines rather than inventing."""
    model = ca_model_builder(seed=99)
    farmer = list(model.world.farmers)[0]

    assert farmer.behaviour.anticipated_change(
        "soilc", FROM_BUNDLE, TO_BUNDLE
    ) == 0.0


class TestCoverCropTypeKeying:
    """Two cover crops are two practices, and the memory has to say which.

    ``indicate_cover_crop_type`` returns a constant today, so nothing in a plain
    run separates the two. These tests drive it from the outside, which is how
    it will behave once the type is chosen on its economics rather than fixed.
    """

    @staticmethod
    def _set_type(farmer, value):
        farmer.indicate_cover_crop_type = lambda: value

    def test_the_two_types_are_remembered_apart(self, ca_model_builder):
        model = ca_model_builder(seed=7)
        farmer = list(model.world.farmers)[0]
        memory = farmer.behaviour.performance_memory
        year = model.lpjml.sim_year

        legume_level = zero_metrics()
        legume_level["yield"] = 0.3
        catch_level = zero_metrics()
        catch_level["yield"] = -0.3

        self._set_type(farmer, 2)
        memory.record_performance(
            TO_BUNDLE, zero_metrics(), legume_level, n=6, year=year
        )
        self._set_type(farmer, 1)
        memory.record_performance(
            TO_BUNDLE, zero_metrics(), catch_level, n=6, year=year
        )

        assert memory.get((TO_BUNDLE, 2)).level["yield"] == pytest.approx(0.3)
        assert memory.get((TO_BUNDLE, 1)).level["yield"] == pytest.approx(-0.3)

    def test_a_bundle_without_a_cover_crop_keys_on_none(self, ca_model_builder):
        """The type must not split slots where there is no cover crop to type."""
        model = ca_model_builder(seed=7)
        farmer = list(model.world.farmers)[0]
        memory = farmer.behaviour.performance_memory

        self._set_type(farmer, 2)
        first = memory.key(FROM_BUNDLE)
        self._set_type(farmer, 1)

        assert first == memory.key(FROM_BUNDLE) == (FROM_BUNDLE, None)

    def test_a_peer_growing_the_other_crop_does_not_count(self, ca_model_builder):
        """Peer matching is on the practice asked about, not the practice held."""
        model = ca_model_builder(seed=7)
        farmers = list(model.world.farmers)
        asker = farmers[0]

        gains = {"yield": 0.2}
        for peer in farmers[1:]:
            self._set_type(peer, 1)  # every peer grows a catch crop
        seed_memories(farmers[1:], model.lpjml.sim_year, gains)

        self._set_type(asker, 2)  # the asker is weighing a legume
        assert asker.behaviour.anticipated_change(
            "yield", FROM_BUNDLE, TO_BUNDLE
        ) == 0.0, (
            "A catch crop's record answered a question about a legume, so the "
            "peer scan is not matching on the cover crop."
        )

        self._set_type(asker, 1)  # now asking about the crop they actually ran
        asker.cell.country._effect_cache = {}
        assert asker.behaviour.anticipated_change(
            "yield", FROM_BUNDLE, TO_BUNDLE
        ) == pytest.approx(np.expm1(0.2))

    def test_the_crop_in_the_ground_outranks_the_preference(self, ca_model_builder):
        """An episode is one crop throughout, so the key cannot drift under it.

        The preference moves with prices and peer records every year. What the
        slot is keyed on is the crop actually sown, which only ``apply_bundle``
        changes, so the record a farm builds up stays experience of one crop.
        """
        model = ca_model_builder(seed=7)
        farmer = list(model.world.farmers)[0]
        behaviour = farmer.behaviour

        behaviour.practice_bundle = TO_BUNDLE
        farmer.cover_crop = 2
        self._set_type(farmer, 1)  # the farm would now rather sow a catch crop

        assert behaviour.performance_memory.key(TO_BUNDLE) == (TO_BUNDLE, 2)


class TestAbsoluteChange:
    """The physical quantity an instrument would be owed, not the ratio."""

    def test_it_is_the_relative_change_on_the_farms_own_baseline(self, seeded_model):
        model, farmers = seeded_model
        farmer = farmers[0]

        relative = farmer.behaviour.anticipated_change("soilc", FROM_BUNDLE, TO_BUNDLE)
        absolute = farmer.behaviour.anticipated_absolute_change(
            "soilc", FROM_BUNDLE, TO_BUNDLE
        )

        assert absolute == pytest.approx(farmer.baseline_soilc * relative)
        assert absolute != pytest.approx(relative), (
            "Soil carbon baselines are hundreds of gC/m2, so an absolute change "
            "that equals the fraction means the baseline never got applied."
        )

    def test_two_farms_with_the_same_ratio_are_owed_different_amounts(
        self, seeded_model
    ):
        """The point of converting: a percentage is not a quantity."""
        model, farmers = seeded_model
        rich, poor = max(
            farmers, key=lambda f: f.baseline_soilc
        ), min(farmers, key=lambda f: f.baseline_soilc)
        if rich.baseline_soilc == poor.baseline_soilc:
            pytest.skip("Fixture cells share a soil carbon baseline.")

        assert rich.behaviour.anticipated_change(
            "soilc", FROM_BUNDLE, TO_BUNDLE
        ) == pytest.approx(
            poor.behaviour.anticipated_change("soilc", FROM_BUNDLE, TO_BUNDLE)
        )
        assert abs(
            rich.behaviour.anticipated_absolute_change("soilc", FROM_BUNDLE, TO_BUNDLE)
        ) > abs(
            poor.behaviour.anticipated_absolute_change("soilc", FROM_BUNDLE, TO_BUNDLE)
        )

    def test_profit_has_nothing_to_be_absolute_in(self, seeded_model):
        """Profit is already an economic score; asking for its physics is a mistake."""
        model, farmers = seeded_model

        with pytest.raises(ValueError, match="no physical counterpart"):
            farmers[0].behaviour.anticipated_absolute_change(
                "profit", FROM_BUNDLE, TO_BUNDLE
            )


class TestSoilCarbonFlow:
    """The quantity a sequestration payment would be written on.

    Soil carbon is scored as a stock, but a payment is owed on the flow, and the
    two are not interchangeable: a farm sitting on a large stock it built
    decades ago is sequestering nothing this year.
    """

    def test_a_building_stock_reads_as_a_positive_flow(self, ca_model_builder):
        model = ca_model_builder(seed=11)
        farmer = list(model.world.farmers)[0]
        tracker = farmer.behaviour.performance_tracker

        # A fitted slope needs two points that differ; write them directly so
        # the sign under test is the one intended. Both series get the same
        # numbers, which is what a run without perception noise produces: the
        # payment is billed from the clean one and the decision taken on the
        # perceived one, and here there is nothing to tell them apart.
        tracker.n = 2
        tracker.sum_t, tracker.sum_tt = 1.0, 1.0
        tracker.sums["soilc"], tracker.time_sums["soilc"] = 0.02, 0.02
        tracker.clean_sums["soilc"] = 0.02
        tracker.clean_time_sums["soilc"] = 0.02

        assert tracker.trend["soilc"] == pytest.approx(0.02)
        assert tracker.clean_trend["soilc"] == pytest.approx(0.02)
        assert farmer.soilc_accumulation == pytest.approx(
            farmer.baseline_soilc * np.expm1(0.02)
        )
        assert farmer.soilc_accumulation > 0.0

    def test_a_falling_stock_is_a_loss_but_not_an_invoice(self, ca_model_builder):
        """Floor at zero: the scheme pays for carbon in, it does not bill for carbon out."""
        model = ca_model_builder(seed=11)
        farmer = list(model.world.farmers)[0]
        tracker = farmer.behaviour.performance_tracker

        tracker.n = 2
        tracker.sum_t, tracker.sum_tt = 1.0, 1.0
        tracker.sums["soilc"], tracker.time_sums["soilc"] = -0.02, -0.02
        tracker.clean_sums["soilc"] = -0.02
        tracker.clean_time_sums["soilc"] = -0.02

        assert farmer.soilc_accumulation < 0.0, (
            "The physical flow should report the decline as a decline."
        )
        assert farmer.creditable_soilc_accumulation == 0.0, (
            "A farm losing carbon should be paid nothing, not charged."
        )

    def test_a_flat_stock_earns_nothing(self, ca_model_builder):
        """Holding a large stock is not sequestering, and is not paid as if it were."""
        model = ca_model_builder(seed=11)
        farmer = list(model.world.farmers)[0]

        farmer.behaviour.performance_tracker.n = 1  # no slope to fit
        assert farmer.creditable_soilc_accumulation == 0.0

    def test_a_misread_stock_is_still_paid_on_what_the_soil_did(
        self, ca_model_builder
    ):
        """Perception noise must not reach the cheque.

        A scheme measures the soil; it does not pay a farm's opinion of it. If
        the payment followed the perceived trend, an observation error would
        become money, and the misperception these runs exist to slow adoption
        down with would turn up instead as a mispriced subsidy.
        """
        model = ca_model_builder(seed=11)
        farmer = list(model.world.farmers)[0]
        tracker = farmer.behaviour.performance_tracker

        tracker.n = 2
        tracker.sum_t, tracker.sum_tt = 1.0, 1.0
        # The soil gained; the farm believes it lost.
        tracker.clean_sums["soilc"] = 0.02
        tracker.clean_time_sums["soilc"] = 0.02
        tracker.sums["soilc"], tracker.time_sums["soilc"] = -0.05, -0.05

        assert tracker.trend["soilc"] == pytest.approx(-0.05)
        assert farmer.soilc_accumulation == pytest.approx(
            farmer.baseline_soilc * np.expm1(0.02)
        )
        assert farmer.creditable_soilc_accumulation > 0.0


class TestPerceptionNoise:
    """What a farm gets wrong reading its own fields."""

    def _tracker(self, noise, weather=0.03, trend=0.02, years=40, seed=0):
        from inseeds.components.farming.ca_management import (
            ManagementPerformanceTracker,
        )
        from inseeds.components.farming.metrics import METRIC_KEYS

        draw = np.random.RandomState(1)
        series = [
            {
                key: (trend * t + weather * draw.randn() if key == "soilc" else 0.0)
                for key in METRIC_KEYS
            }
            for t in range(years)
        ]
        np.random.seed(seed)
        tracker = ManagementPerformanceTracker(t_start=0)
        for year, rates in enumerate(series):
            tracker.add_observation(rates, year, noise=noise)
        return tracker

    def test_off_leaves_the_two_series_identical(self):
        """The default has to be the model as it was, to the bit."""
        tracker = self._tracker(None)
        assert tracker.trend == tracker.clean_trend
        assert tracker.sums == tracker.clean_sums
        assert tracker.time_sums == tracker.clean_time_sums

    def test_the_ruler_is_the_wobble_and_not_the_climb(self):
        """A strong trend must not be read as a farm that cannot see straight.

        ``level_std`` grows with the trend because it is the spread of the
        whole series. Scaling the error on that would hand the most noise to
        the farms where the practice works best.
        """
        from inseeds.components.farming.ca_management import PerceptionNoise

        for trend in (0.005, 0.02, 0.08):
            tracker = self._tracker(PerceptionNoise(sigma=0.25), trend=trend)
            assert tracker.clean_interannual_std["soilc"] == pytest.approx(
                0.03, abs=0.01
            ), "the ruler should recover the weather, whatever the trend"

    def test_the_realised_error_matches_the_nominal_sigma(self):
        """sigma is the swept number, so it has to mean what it says."""
        from inseeds.components.farming.ca_management import PerceptionNoise

        for sigma in (0.1, 0.25, 0.5):
            totals = []
            for seed in range(300):
                tracker = self._tracker(PerceptionNoise(sigma=sigma), seed=seed)
                totals.append(
                    tracker.sums["soilc"] - tracker.clean_sums["soilc"]
                )
            # Sum of ~38 independent per-year errors.
            realised = float(np.std(totals) / np.sqrt(38))
            assert realised == pytest.approx(sigma * 0.03, rel=0.2)

    def test_a_metric_can_be_made_harder_to_read_than_another(self):
        """Soil carbon is not as legible as yield, and the config can say so."""
        from inseeds.components.farming.ca_management import PerceptionNoise

        plain = self._tracker(PerceptionNoise(sigma=0.25), seed=3)
        harder = self._tracker(
            PerceptionNoise(sigma=0.25, observability={"soilc": 20.0}), seed=3
        )
        plain_error = abs(plain.sums["soilc"] - plain.clean_sums["soilc"])
        harder_error = abs(harder.sums["soilc"] - harder.clean_sums["soilc"])
        assert harder_error > plain_error
        assert plain.clean_trend == harder.clean_trend, (
            "however badly it is read, the soil did the same thing"
        )


class TestNoStacking:
    """Paying for the practice and for the carbon the practice produced."""

    @staticmethod
    def _settings(no_stacking):
        from inseeds.components.farming.ca_management import PolicySettings

        return PolicySettings.from_config(
            {
                "carbon_price": 185.0,
                "practice_subsidies": {
                    "cover_crop": [60.0, 60.0],
                    "residue_on_field": [20.0, 20.0],
                },
                "no_stacking": no_stacking,
            },
            gdp_cost_ratio=1.0,
            default_span=(2025, 2100),
        )

    def test_off_pays_both(self):
        """The headline scenario, and the thing a referee calls double payment."""
        assert self._settings(False).unstacked(70.0, 80.0) == (70.0, 80.0)

    def test_on_pays_the_larger_and_not_the_sum(self):
        assert self._settings(True).unstacked(70.0, 80.0) == (0.0, 80.0)
        assert self._settings(True).unstacked(90.0, 80.0) == (90.0, 0.0)

    def test_on_never_pays_more_than_off(self):
        """The whole point: it can only bound the stacked result from below."""
        settings_on = self._settings(True)
        settings_off = self._settings(False)
        for carbon, practice in [(0.0, 0.0), (70.0, 80.0), (90.0, 0.0), (0.0, 55.0)]:
            assert sum(settings_on.unstacked(carbon, practice)) <= sum(
                settings_off.unstacked(carbon, practice)
            )

    def test_a_farm_earning_only_one_is_untouched(self):
        """It bites where both are on, so single-instrument runs are unchanged."""
        settings = self._settings(True)
        assert sum(settings.unstacked(90.0, 0.0)) == 90.0
        assert sum(settings.unstacked(0.0, 55.0)) == 55.0

    def test_the_rule_is_not_itself_an_instrument(self):
        """A run with no_stacking and no rates still resolves to no policy."""
        from inseeds.components.farming.ca_management import PolicySettings

        settings = PolicySettings.from_config(
            {"no_stacking": True},
            gdp_cost_ratio=1.0,
            default_span=(2025, 2100),
        )
        assert settings.any_in_force is False


class TestReportingNoise:
    """What a farm tells its neighbours, as against what its fields did."""

    @staticmethod
    def _give_a_record(behaviour, wobble=0.1):
        """A real series on the tracker, so there is a spread to scale against."""
        from inseeds.components.farming.metrics import METRIC_KEYS

        tracker = behaviour.performance_tracker
        draw = np.random.RandomState(4)
        for t in range(12):
            tracker._add_point(
                t, {key: wobble * draw.randn() for key in METRIC_KEYS}
            )
        assert tracker.clean_interannual_std["yield"] > 0.0
        return tracker

    def test_off_is_absent_rather_than_zeroed(self):
        """So a run without it takes no draws and disturbs no random stream."""
        from inseeds.components.farming.ca_management import ReportingNoise

        class Block:
            enabled = False
            sigma = 1.0

        class Tpb:
            reporting_noise = Block()

        assert ReportingNoise.from_config(Tpb()) is None

    def test_a_farm_tells_every_neighbour_the_same_story(self, ca_model_builder):
        """Drawn once per comparison and kept.

        Redrawing per listener would let the error average out across a
        neighbourhood, which is the one thing the reporting literature says it
        does not do.
        """
        from inseeds.components.farming.ca_management import ReportingNoise

        model = ca_model_builder(seed=11)
        farmer = list(model.world.farmers)[0]
        behaviour = farmer.behaviour
        self._give_a_record(behaviour)

        noise = ReportingNoise(sigma=1.0)
        first = behaviour.report_error("yield", ("a", 0), ("b", 0), noise)
        again = behaviour.report_error("yield", ("a", 0), ("b", 0), noise)
        assert first == again

    def test_a_farm_talks_up_the_practice_it_runs(self, ca_model_builder):
        """The self-justification term, and the half that does not cancel."""
        from inseeds.components.farming.ca_management import ReportingNoise

        model = ca_model_builder(seed=11)
        farmer = list(model.world.farmers)[0]
        behaviour = farmer.behaviour
        self._give_a_record(behaviour)

        mine = behaviour.practice_bundle
        other = object()
        noise = ReportingNoise(sigma=0.0, bias=0.5)

        # Reporting on a move *to* the practice it runs: talked up.
        towards = behaviour.report_error("yield", (other, 0), (mine, 0), noise)
        # Reporting on a move *away* from it: talked down.
        away = behaviour.report_error("yield", (mine, 0), (other, 0), noise)

        assert towards > 0.0
        assert away < 0.0
        assert towards == pytest.approx(-away)

    def test_a_farm_with_no_record_yet_says_nothing_wrong(self, ca_model_builder):
        """No spread to scale against means no invented error."""
        from inseeds.components.farming.ca_management import ReportingNoise

        model = ca_model_builder(seed=11)
        behaviour = list(model.world.farmers)[0].behaviour
        behaviour.performance_tracker.n = 1

        assert behaviour.report_error(
            "yield", ("a", 0), ("b", 0), ReportingNoise(sigma=5.0)
        ) == 0.0


class TestPhysicalCounterparts:
    """What the registry promises an instrument can be written against."""

    def test_every_priceable_metric_can_be_read_off_a_farm(self, seeded_model):
        model, farmers = seeded_model
        farmer = farmers[0]

        for spec in METRICS.values():
            if not spec.priceable:
                continue
            assert hasattr(farmer, spec.physical_attr), (
                f"{spec.key} claims a physical counterpart "
                f"{spec.physical_attr!r} the farm does not carry."
            )
            assert hasattr(farmer, spec.physical_baseline_attr), (
                f"{spec.key} claims a physical baseline "
                f"{spec.physical_baseline_attr!r} the farm does not carry."
            )

    def test_every_metric_is_priced_on_what_it_is_scored_on(self, seeded_model):
        """The rate must be the log deviation of the priceable quantity itself.

        Otherwise converting a remembered rate into kilograms carries a ratio
        across from one quantity to another, and the conversion holds only if
        whatever separates them happens to be equal between the two practices
        being compared. Leaching was scored as a concentration and priceable
        only as a flux, which assumed the two bundles drained alike - the very
        thing a cover crop changes.
        """
        model, farmers = seeded_model
        farmer = farmers[0]

        for spec in METRICS.values():
            if not spec.priceable:
                continue
            rate = getattr(farmer, spec.rate_attr)
            expected = log_rate(
                getattr(farmer, spec.physical_attr),
                getattr(farmer, spec.physical_baseline_attr),
                getattr(farmer, spec.negligible_key),
            )
            assert rate == pytest.approx(expected), (
                f"{spec.key} is scored on something other than "
                f"{spec.physical_attr!r}, so a price on it would be a guess."
            )

    def test_the_leaching_baseline_counts_dry_years(self, seeded_model):
        """A year that drained nothing lost nothing, which is a real zero.

        A concentration baseline has to skip those years, because a
        concentration without drainage is undefined - which reads the record
        as if the dry years had never happened. The flux has no such gap, and
        the baseline is the plain mean over every year on file.
        """
        model, farmers = seeded_model

        for farmer in farmers:
            n_years = len(farmer.cell.from_earth.time)
            record = [
                agricultural_leaching(farmer, time_idx=i)
                for i in range(n_years)
            ]

            assert farmer.baseline_leaching == pytest.approx(float(np.mean(record)))
            assert farmer.baseline_leaching >= 0.0
