"""What a carbon payment is written against.

The scheme as first specified pays on the rise in a farm's own soil carbon
stock, floored at zero, which is how soil carbon protocols are written. On land
that is losing carbon under every available management - most cropland in these
runs - that floor binds whatever the farm does, so a practice that turns a loss
of 8 gC/m2/yr into a loss of 2 is paid nothing for the 6 it kept out of the air.

``policy.carbon_price_basis`` is the alternative, and it is a separate option
rather than a correction: ``stock_gain`` has to keep settling exactly what it
settled before, because the published runs are on it. These tests pin all three
bases - the old one against drift, and the two avoided-loss ones against the
arithmetic they are supposed to do and the gaming they are supposed to refuse.
"""

import numpy as np
import pytest

from inseeds.components.farming.ca_management import (
    COVER_CROP_LEGUME,
    COVER_CROP_NONLEGUME,
    ManagementBundle,
    ManagementPerformance,
    PolicySettings,
)
from inseeds.components.farming.ca_world import (
    GLOBAL_REFERENCE,
    MIN_REFERENCE_YEARS,
)
from inseeds.components.farming.metrics import zero_metrics


#: A price high enough that nothing rounds to zero, in $/tCO2e. The rate is not
#: what is under test here - which slope it multiplies is.
PRICE = 100.0

#: Log-deviation slopes per year. Losing, losing less, and gaining, so that a
#: basis can be shown to separate them. Small enough that expm1 stays close to
#: linear, which keeps the expected values readable.
LOSING = -0.02
LOSING_LESS = -0.005
GAINING = 0.01


class _Tracker:
    """A performance tracker stub whose slope is imposed rather than fitted.

    The regression is tested elsewhere. What these tests are about is which
    slope the payment is written against, so the slope is set by hand: fitting
    one would make every expected value depend on the fixture's weather.
    """

    def __init__(self, slope, n=MIN_REFERENCE_YEARS):
        self.n = n
        self.slope = slope

    @property
    def clean_trend(self):
        trend = zero_metrics()
        trend["soilc"] = self.slope
        return trend

    @property
    def trend(self):
        return self.clean_trend


def _policy(model):
    return model.config.coupled_config.policy


def _set_basis(model, basis, price=PRICE):
    """Put an instrument in force on ``basis``."""
    _policy(model).carbon_price = price
    _policy(model).carbon_price_basis = basis


def _set_reference(model, slope, code=None):
    """Impose the conventional drift, globally and optionally for one country."""
    reference = {GLOBAL_REFERENCE: slope}
    if code is not None:
        reference[code] = slope
    model.world.statistic.set("conventional_soilc_trend", reference)


def _hold(farmer, bundle):
    """Put a farm on ``bundle``, so that ``committed_bundle`` agrees.

    Both flags, because the committed bundle follows ``residue_committed`` and
    not the practice triple when the two disagree - an unenrolled farm above
    the litter threshold is not treated as having retained.
    """
    farmer.behaviour.practice_bundle = bundle
    farmer.behaviour.residue_committed = bool(bundle.residue_on_field)


def _observe(farmer, slope, n=MIN_REFERENCE_YEARS):
    farmer.behaviour.performance_tracker = _Tracker(slope, n=n)


def _accumulation(farmer, slope):
    """The gC/m2/yr a slope means on this farm's stock, as the model converts."""
    return float(farmer.baseline_soilc * np.expm1(slope))


@pytest.fixture
def adopter(ca_model_builder):
    """One farm that has moved to full CA, on land that is losing carbon.

    The conventional reference is put at the same slope the farm itself had
    before moving, so that any credit under an avoided-loss basis can only have
    come from the farm's own departure and not from a drift left lying around.
    """
    model = ca_model_builder(seed=20261006)
    farmer = next(iter(model.world.farmers))

    _set_reference(model, LOSING)
    _observe(farmer, LOSING)
    _hold(farmer, ManagementBundle.conventional)
    farmer._update_soilc_anchor()  # freeze the anchor while still conventional

    _hold(farmer, ManagementBundle.conservation)
    farmer._update_soilc_anchor()  # and close it on departure

    return model, farmer


class TestTheSetting:
    """What the option is, before what it pays."""

    def test_the_shipped_config_is_on_the_old_basis(self, ca_model_builder):
        """The default has to be the basis the published runs used."""
        model = ca_model_builder(seed=1)

        assert _policy(model).carbon_price_basis == "stock_gain"

    def test_a_config_without_the_key_reads_as_stock_gain(self):
        """An older configuration must keep working and keep its numbers."""
        settings = PolicySettings.from_config({"carbon_price": PRICE})

        assert settings.carbon_price_basis == "stock_gain"

    def test_an_empty_value_reads_as_stock_gain(self):
        """A key left blank in yaml arrives as None, not as a missing key."""
        settings = PolicySettings.from_config(
            {"carbon_price": PRICE, "carbon_price_basis": None}
        )

        assert settings.carbon_price_basis == "stock_gain"

    def test_an_unknown_basis_is_refused(self):
        """Silently falling back would make a typo look like a result."""
        with pytest.raises(ValueError, match="carbon_price_basis"):
            PolicySettings.from_config(
                {"carbon_price": PRICE, "carbon_price_basis": "avoided-loss"}
            )

    def test_every_listed_basis_parses(self):
        for basis in PolicySettings.CARBON_PRICE_BASES:
            settings = PolicySettings.from_config(
                {"carbon_price": PRICE, "carbon_price_basis": basis}
            )
            assert settings.carbon_price_basis == basis

    def test_a_farm_without_an_instrument_still_reports_a_basis(self, adopter):
        """``policy`` resolves to None with no instrument in force.

        The farm must not raise on the way to reporting its accumulation, and
        must report the old quantity, which is what a run with no carbon price
        has always written out.
        """
        model, farmer = adopter
        _policy(model).carbon_price = 0.0
        _policy(model).carbon_price_basis = "avoided_loss"

        assert farmer.policy is None
        assert farmer.carbon_price_basis == "stock_gain"


class TestStockGain:
    """The old basis, which has to keep doing exactly what it did."""

    def test_a_falling_stock_is_paid_nothing(self, adopter):
        """The floor binding on declining land is the whole of the problem."""
        model, farmer = adopter
        _set_basis(model, "stock_gain")

        assert farmer.soilc_accumulation < 0
        assert farmer.creditable_soilc_accumulation == 0.0
        assert farmer.carbon_payment_for(
            farmer.creditable_soilc_accumulation
        ) == 0.0

    def test_losing_less_is_still_paid_nothing(self, adopter):
        """The case the avoided-loss bases exist for, pinned on the old one."""
        model, farmer = adopter
        _set_basis(model, "stock_gain")
        _observe(farmer, LOSING_LESS)

        assert farmer.creditable_soilc_accumulation == 0.0

    def test_a_rising_stock_is_paid_on_the_rise(self, adopter):
        model, farmer = adopter
        _set_basis(model, "stock_gain")
        _observe(farmer, GAINING)

        assert farmer.creditable_soilc_accumulation == pytest.approx(
            _accumulation(farmer, GAINING)
        )

    def test_it_ignores_the_drift(self, adopter):
        """No reference is read, so the statistic cannot reach the payment."""
        model, farmer = adopter
        _set_basis(model, "stock_gain")
        _observe(farmer, GAINING)
        before = farmer.creditable_soilc_accumulation

        _set_reference(model, LOSING * 10)

        assert farmer.creditable_soilc_accumulation == before

    def test_a_conventional_farm_that_gained_is_still_paid(self, adopter):
        """No additionality gate on this basis: the carbon really went in."""
        model, farmer = adopter
        _set_basis(model, "stock_gain")
        _observe(farmer, GAINING)
        _hold(farmer, ManagementBundle.conventional)

        assert farmer.creditable_soilc_accumulation > 0.0


class TestTheAnchor:
    """Which year's slope an ``avoided_loss`` credit is measured from."""

    def test_it_follows_a_farm_that_has_not_moved(self, ca_model_builder):
        model = ca_model_builder(seed=2)
        farmer = next(iter(model.world.farmers))
        _set_reference(model, LOSING)
        _hold(farmer, ManagementBundle.conventional)

        _observe(farmer, LOSING)
        farmer._update_soilc_anchor()
        assert farmer._soilc_anchor == (pytest.approx(LOSING), pytest.approx(LOSING))

        _observe(farmer, LOSING_LESS)
        farmer._update_soilc_anchor()
        assert farmer._soilc_anchor[0] == pytest.approx(LOSING_LESS)

    def test_it_freezes_when_the_farm_moves(self, adopter):
        """The anchor is the last conventional year, not the first CA one."""
        model, farmer = adopter

        assert farmer._soilc_anchor_frozen is True
        assert farmer._soilc_anchor[0] == pytest.approx(LOSING)

        _observe(farmer, GAINING)
        farmer._update_soilc_anchor()

        assert farmer._soilc_anchor[0] == pytest.approx(LOSING)

    def test_abandoning_and_re_adopting_cannot_re_anchor(self, adopter):
        """The gaming the freeze exists to refuse.

        Let the slope fall back while conventional, re-adopt, and a farm allowed
        to re-anchor would be paid a second time for recovering ground it chose
        to lose.
        """
        model, farmer = adopter
        frozen = farmer._soilc_anchor

        _hold(farmer, ManagementBundle.conventional)
        _observe(farmer, LOSING * 2)
        farmer._update_soilc_anchor()
        _hold(farmer, ManagementBundle.conservation)
        farmer._update_soilc_anchor()

        assert farmer._soilc_anchor == frozen

    def test_a_farm_that_moved_too_early_has_none(self, ca_model_builder):
        """A slope over two points is a difference and carries one year's weather."""
        model = ca_model_builder(seed=3)
        farmer = next(iter(model.world.farmers))
        _set_reference(model, LOSING)
        _observe(farmer, LOSING, n=MIN_REFERENCE_YEARS - 1)
        _hold(farmer, ManagementBundle.conventional)
        farmer._update_soilc_anchor()

        _hold(farmer, ManagementBundle.conservation)
        farmer._update_soilc_anchor()

        assert farmer._soilc_anchor is None
        assert farmer._soilc_anchor_frozen is True

    def test_a_farm_without_an_anchor_falls_back_to_the_benchmark(
        self, ca_model_builder
    ):
        """Most pioneers, by construction: they move before they have a record."""
        model = ca_model_builder(seed=4)
        farmer = next(iter(model.world.farmers))
        _set_basis(model, "avoided_loss")
        _set_reference(model, LOSING)
        _observe(farmer, LOSING_LESS, n=MIN_REFERENCE_YEARS - 1)
        _hold(farmer, ManagementBundle.conservation)
        farmer._update_soilc_anchor()

        assert farmer._soilc_anchor is None
        assert farmer.baseline_soilc_trend == pytest.approx(LOSING)


class TestTheBaseline:
    """The line the credit is measured against, and how the climate moves it."""

    def test_it_is_the_anchor_where_nothing_has_drifted(self, adopter):
        model, farmer = adopter
        _set_basis(model, "avoided_loss")

        assert farmer.baseline_soilc_trend == pytest.approx(LOSING)

    def test_a_worsening_climate_steepens_it(self, adopter):
        """s_base = s_own(t0) + [s_conv(t) - s_conv(t0)], and the drift is negative."""
        model, farmer = adopter
        _set_basis(model, "avoided_loss")
        drift = -0.004
        _set_reference(model, LOSING + drift)

        assert farmer.baseline_soilc_trend == pytest.approx(LOSING + drift)

    def test_an_improving_climate_lifts_it(self, adopter):
        """Symmetric: the farmer is not paid for the climate turning kind either."""
        model, farmer = adopter
        _set_basis(model, "avoided_loss")
        drift = 0.004
        _set_reference(model, LOSING + drift)

        assert farmer.baseline_soilc_trend == pytest.approx(LOSING + drift)

    def test_the_benchmark_basis_ignores_the_anchor(self, adopter):
        model, farmer = adopter
        _set_basis(model, "avoided_loss_benchmark")
        _set_reference(model, LOSING_LESS)

        assert farmer._soilc_anchor is not None
        assert farmer.baseline_soilc_trend == pytest.approx(LOSING_LESS)

    def test_a_country_pool_is_preferred_to_the_global_one(self, adopter):
        model, farmer = adopter
        _set_basis(model, "avoided_loss_benchmark")
        code = farmer.cell.country.code
        model.world.statistic.set(
            "conventional_soilc_trend",
            {GLOBAL_REFERENCE: LOSING, code: LOSING_LESS},
        )

        assert farmer.baseline_soilc_trend == pytest.approx(LOSING_LESS)

    def test_an_unset_statistic_reads_as_no_drift(self, adopter):
        """The first years of a run, before any tracker has three observations."""
        model, farmer = adopter
        _set_basis(model, "avoided_loss_benchmark")
        model.world.statistic.set("conventional_soilc_trend", {})

        assert farmer.conventional_soilc_trend == 0.0


class TestTheCredit:
    """What the farm is actually paid, which is the point of the exercise."""

    def test_losing_less_is_paid_where_stock_gain_pays_nothing(self, adopter):
        """The one result this option exists to produce."""
        model, farmer = adopter
        _observe(farmer, LOSING_LESS)

        _set_basis(model, "stock_gain")
        assert farmer.creditable_soilc_accumulation == 0.0

        _set_basis(model, "avoided_loss")
        credit = farmer.creditable_soilc_accumulation

        assert credit > 0.0
        assert credit == pytest.approx(
            _accumulation(farmer, LOSING_LESS) - _accumulation(farmer, LOSING)
        )

    def test_holding_flat_against_a_worsening_climate_widens_the_credit(
        self, adopter
    ):
        """The property the stock basis cannot have.

        The farm does not improve at all between the two reads. Everything that
        changes is the line beneath it.
        """
        model, farmer = adopter
        _set_basis(model, "avoided_loss")
        _observe(farmer, LOSING)

        assert farmer.creditable_soilc_accumulation == pytest.approx(0.0)

        _set_reference(model, LOSING - 0.004)

        assert farmer.creditable_soilc_accumulation > 0.0

    def test_crossing_into_gain_is_credited_whole(self, adopter):
        """No discontinuity at zero: it is one difference of slopes throughout."""
        model, farmer = adopter
        _set_basis(model, "avoided_loss")
        _observe(farmer, GAINING)

        assert farmer.creditable_soilc_accumulation == pytest.approx(
            _accumulation(farmer, GAINING) - _accumulation(farmer, LOSING)
        )

    def test_the_credit_is_monotone_in_the_farm_s_own_slope(self, adopter):
        model, farmer = adopter
        _set_basis(model, "avoided_loss")

        credits = []
        for slope in (LOSING, LOSING_LESS, 0.0, GAINING):
            _observe(farmer, slope)
            credits.append(farmer.creditable_soilc_accumulation)

        assert credits == sorted(credits)

    def test_doing_worse_than_the_baseline_is_not_invoiced(self, adopter):
        """Still floored, so the scheme stays something a cautious farm enrols in."""
        model, farmer = adopter
        _set_basis(model, "avoided_loss")
        _observe(farmer, LOSING * 2)

        assert farmer.soilc_accumulation < farmer.counterfactual_soilc_accumulation
        assert farmer.creditable_soilc_accumulation == 0.0

    def test_a_conventional_farm_is_never_credited(self, adopter):
        """Additionality: a farm that changed nothing has nothing to be paid for."""
        model, farmer = adopter
        _set_basis(model, "avoided_loss")
        _observe(farmer, GAINING)
        _hold(farmer, ManagementBundle.conventional)

        assert farmer.creditable_soilc_accumulation == 0.0

    def test_the_payment_follows_the_credit(self, adopter):
        """The basis has to reach money, not just the reported accumulation."""
        model, farmer = adopter
        _set_basis(model, "avoided_loss")
        _observe(farmer, LOSING_LESS)

        payment = farmer.carbon_payment_for(farmer.creditable_soilc_accumulation)

        assert payment > 0.0
        assert farmer.carbon_payment_for(0.0) == 0.0


def _remember_slopes(model, before, after, level_after=0.0):
    """Every farm remembers conventional at ``before`` and CA at ``after``.

    Written under every cover crop key, so which crop a farm prefers cannot
    decide whether the estimator finds the record.
    """
    year = model.lpjml.sim_year
    for farm in model.world.farmers:
        slots = farm.behaviour.performance_memory._slots
        for bundle, slope, level in (
            (ManagementBundle.conventional, before, 0.0),
            (ManagementBundle.conservation, after, level_after),
        ):
            trend, mean = zero_metrics(), zero_metrics()
            trend["soilc"], mean["soilc"] = slope, level
            for crop in (None, COVER_CROP_NONLEGUME, COVER_CROP_LEGUME):
                slots[(bundle, crop)] = ManagementPerformance.from_tracker(
                    trend, mean, 10, year
                )


@pytest.fixture
def deciding(ca_model_builder):
    """A conventional farm on land losing carbon, weighing a move to CA."""
    model = ca_model_builder(seed=20261007)
    farmer = next(iter(model.world.farmers))

    _set_reference(model, LOSING)
    _observe(farmer, LOSING)
    _hold(farmer, ManagementBundle.conventional)
    farmer._update_soilc_anchor()

    return model, farmer


class TestTheExpectation:
    """What a farm expects to be paid is what the basis would pay it.

    The worked case: land losing carbon, peers whose switch slowed the loss
    (or turned it into a gain). The expectation reads the change in peers'
    slopes and settles it as the payment would be settled.
    """

    @staticmethod
    def _expected(farmer):
        return farmer.behaviour.anticipated_carbon_credit_change(
            ManagementBundle.conservation
        )

    def test_losing_less_is_expected_under_avoided_loss(self, deciding):
        model, farmer = deciding
        _set_basis(model, "avoided_loss")
        _remember_slopes(model, LOSING, LOSING_LESS)

        assert self._expected(farmer) == pytest.approx(
            _accumulation(farmer, LOSING_LESS) - _accumulation(farmer, LOSING)
        )

    def test_losing_less_is_not_expected_under_stock_gain(self, deciding):
        model, farmer = deciding
        _set_basis(model, "stock_gain")
        _remember_slopes(model, LOSING, LOSING_LESS)

        assert self._expected(farmer) == pytest.approx(0.0)

    def test_crossing_into_gain_is_expected_on_either_basis(self, deciding):
        model, farmer = deciding
        _remember_slopes(model, LOSING, GAINING)
        shift = GAINING - LOSING

        _set_basis(model, "stock_gain")
        assert self._expected(farmer) == pytest.approx(
            _accumulation(farmer, LOSING + shift)
        )

        _set_basis(model, "avoided_loss")
        assert self._expected(farmer) == pytest.approx(
            _accumulation(farmer, LOSING + shift) - _accumulation(farmer, LOSING)
        )

    def test_a_lower_level_does_not_hide_a_better_slope(self, deciding):
        """The case the level comparison got wrong.

        Peers' second episode sits below their first, because the stock was
        still falling, though more slowly. A level difference reads that as a
        loss; the payment is on the slope.
        """
        model, farmer = deciding
        _set_basis(model, "avoided_loss")
        _remember_slopes(model, LOSING, LOSING_LESS, level_after=-0.05)

        assert farmer.behaviour.anticipated_change(
            "soilc", ManagementBundle.conventional, ManagementBundle.conservation
        ) < 0.0
        assert self._expected(farmer) > 0.0

    def test_what_is_expected_is_what_is_then_paid(self, deciding):
        model, farmer = deciding
        _set_basis(model, "avoided_loss")
        _remember_slopes(model, LOSING, LOSING_LESS)
        expected = self._expected(farmer)

        _hold(farmer, ManagementBundle.conservation)
        farmer._update_soilc_anchor()
        _observe(farmer, LOSING_LESS)

        assert farmer.creditable_soilc_accumulation == pytest.approx(expected)

    def test_it_is_paid_every_year_of_the_horizon(self, deciding):
        model, farmer = deciding
        _set_basis(model, "avoided_loss")
        _remember_slopes(model, LOSING, LOSING_LESS)

        transfer = farmer.behaviour.anticipated_policy_transfer(
            ManagementBundle.conservation, benefit_years=6.0
        )

        assert transfer == pytest.approx(
            farmer.carbon_payment_for(self._expected(farmer)) * 6.0
        )
