"""Carbon payments, nitrogen charges, and the cover crop they pay for.

None of this is reachable from the golden trajectory: its farms never adopt a
cover crop bundle and its instruments are off, which is exactly the point of
the golden - it pins the counterfactual. So the instruments need testing here,
against the two things that are easy to get wrong and impossible to see in an
aggregate. First, both payments are on flows - soil carbon on the slope of the
stock, nitrogen on the yearly loss - so each is anticipated per year and
repeated over the planning horizon. Second, the two cover crops
are alternatives, so a farm must be charged for the seed it sowed and not for
the sum of both.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from inseeds.components.farming.ca_management import (
    COVER_CROP_LEGUME,
    COVER_CROP_NONLEGUME,
    ManagementBundle,
    ManagementPerformance,
    PolicySettings,
    default_policy_span,
)
from inseeds.components.farming.farmer import (
    GC_M2_TO_TCO2_HA,
    GN_M2_TO_KGN_HA,
    NON_CROPS,
    agricultural_leaching,
)
from inseeds.components.farming.metrics import zero_metrics


COVER_CROP_BUNDLE = ManagementBundle.from_practices(0, 1, 0)


@pytest.fixture
def farm(ca_model_builder):
    """One farm out of a private model, with the instruments still off."""
    model = ca_model_builder(seed=909)
    return model, list(model.world.farmers)[0]


def set_policy(
    model,
    carbon_price=0.0,
    nitrogen_charge=0.0,
    levy_free_concentration=0.0,
    cover_crop_subsidy=0.0,
    cover_crop_legume_subsidy=0.0,
    residue_subsidy=0.0,
    residue_subsidy_share=0.0,
    start_year=None,
    end_year=9999,
    windows=None,
):
    """Turn instruments on for a model, from this year unless told otherwise."""
    policy = model.config.coupled_config.policy
    policy.carbon_price = carbon_price
    policy.nitrogen_charge = nitrogen_charge
    policy.levy_free_concentration = levy_free_concentration
    policy.practice_subsidies = {
        "cover_crop": cover_crop_subsidy,
        "cover_crop_legume": cover_crop_legume_subsidy,
        "residue_on_field": residue_subsidy,
        "residue_on_field_share": residue_subsidy_share,
    }
    policy.start_year = (
        model.lpjml.sim_year if start_year is None else start_year
    )
    policy.end_year = end_year
    policy.windows = windows or {}
    return policy


def credit(farmer):
    """The sequestration payment the farm is due as of now.

    The ``carbon_payment`` property reports what ``update_capital`` last
    settled on, so a test that has not run a step reads the live computation
    the same way ``update_capital`` does.
    """
    return farmer.carbon_payment_for(farmer.creditable_soilc_accumulation)


def bill(farmer):
    """The nitrogen charge the farm owes as of now; see ``credit``."""
    return farmer.nitrogen_charge_for(farmer.leaching)


def subsidy(farmer):
    """The practice payment the farm is due as of now; see ``credit``."""
    return farmer.practice_subsidy_for(farmer.behaviour.committed_bundle)


def remember(farmer, bundle, cover_crop_type, levels, year, n=5, trends=None):
    """Write a performance record under one exact ``(bundle, type)`` key.

    Goes at the slot directly rather than through ``record_performance``, which
    resolves the type from the farm asking. Here the type is the variable under
    test, so it has to be stated.
    """
    level = zero_metrics()
    level.update(levels)
    trend = zero_metrics()
    trend.update(trends or {})
    farmer.behaviour.performance_memory._slots[(bundle, cover_crop_type)] = (
        ManagementPerformance.from_tracker(trend, level, n, year)
    )


def remember_either_type(farmer, bundle, levels, year, trends=None):
    """Record the same experience under both cover crops.

    For tests about something other than the type, where which crop the farm
    happens to prefer this year would otherwise decide whether the estimator
    finds a record at all.
    """
    for cover_crop_type in (COVER_CROP_NONLEGUME, COVER_CROP_LEGUME):
        remember(farmer, bundle, cover_crop_type, levels, year, trends=trends)


def set_accumulation(monkeypatch, farmer, gc_per_m2_per_year):
    """Pin this year's soil carbon flow, whatever the tracker's slope says."""
    # raising=False because the property is inherited from the mixin, not
    # declared on the class pycopancore composes for the realisation.
    monkeypatch.setattr(
        type(farmer),
        "soilc_accumulation",
        property(lambda self: gc_per_m2_per_year),
        raising=False,
    )


class TestTheDefaultIsNoInstrument:
    """A run that sets nothing must be arithmetically untouched by all this."""

    def test_no_policy_is_in_force(self, farm):
        model, farmer = farm

        assert farmer.policy is None
        assert credit(farmer) == 0.0
        assert bill(farmer) == 0.0

    def test_nothing_is_anticipated_either(self, farm):
        model, farmer = farm

        transfer = farmer.behaviour.anticipated_policy_transfer(
            COVER_CROP_BUNDLE, benefit_years=10.0
        )

        assert transfer == 0.0

    def test_an_instrument_not_yet_started_is_not_in_force(self, farm):
        """Set for a later year, it must not act now."""
        model, farmer = farm
        set_policy(
            model,
            carbon_price=100.0,
            nitrogen_charge=5.0,
            start_year=model.lpjml.sim_year + 5,
        )

        assert farmer.policy is None
        assert credit(farmer) == 0.0
        assert bill(farmer) == 0.0


class TestCarbonPayment:
    """Paid on carbon put in, never charged on carbon coming out."""

    def test_a_gain_is_paid_at_the_posted_price(self, farm, monkeypatch):
        model, farmer = farm
        set_policy(model, carbon_price=50.0)
        set_accumulation(monkeypatch, farmer, 20.0)

        expected = 20.0 * GC_M2_TO_TCO2_HA * 50.0 * farmer.farm_size

        assert farmer.creditable_soilc_accumulation == 20.0
        assert credit(farmer) == pytest.approx(expected)

    def test_a_loss_draws_no_invoice(self, farm, monkeypatch):
        """The asymmetry, and the reason shedding the practice stays cheap."""
        model, farmer = farm
        set_policy(model, carbon_price=50.0)
        set_accumulation(monkeypatch, farmer, -20.0)

        assert farmer.creditable_soilc_accumulation == 0.0
        assert credit(farmer) == 0.0

    def test_the_unit_conversion_is_carbon_to_carbon_dioxide(self):
        """gC/m2 to tCO2/ha: a hundredth for the area, 44/12 for the oxygen."""
        assert GC_M2_TO_TCO2_HA == pytest.approx(0.01 * 44.0 / 12.0)
        assert GC_M2_TO_TCO2_HA == pytest.approx(0.036667, abs=1e-6)


class TestNitrogenCharge:
    """Levied on the nitrogen that actually left, as a real tax would be."""

    def test_the_bill_is_on_the_loss_itself(self, farm):
        model, farmer = farm
        set_policy(model, nitrogen_charge=2.0)

        expected = farmer.leaching * GN_M2_TO_KGN_HA * 2.0 * farmer.farm_size

        assert bill(farmer) == pytest.approx(expected)

    def test_it_is_not_the_change_against_baseline(self, farm):
        """A farm leaching exactly what it always did still owes the tax.

        The distinction is the whole difference between a tax and a credit
        scheme, and both are defensible, so it is worth a test that says which
        one this is.
        """
        model, farmer = farm
        set_policy(model, nitrogen_charge=2.0)
        farmer.leaching = farmer.baseline_leaching

        assert farmer.baseline_leaching > 0.0, "Need a farm that leaches."
        assert bill(farmer) > 0.0

    def test_a_farm_that_loses_nothing_owes_nothing(self, farm):
        model, farmer = farm
        set_policy(model, nitrogen_charge=2.0)
        farmer.leaching = 0.0

        assert bill(farmer) == 0.0


class TestTheLevyFreeAllowance:
    """The charge can be written on the excess over a standard, as MINAS was.

    A concentration, because that is the form a nitrate limit takes and because
    it is the strength reaching the aquifer that does the harm: the same
    kilogram is harmless where there is water to dilute it. So the allowance in
    mass terms belongs to the farm and its drainage, which is the one place the
    model still consults a concentration - as a threshold read once, and not as
    a quantity carried through baselines and peer memory.
    """

    def test_zero_charges_every_kilogram(self, farm):
        model, farmer = farm
        set_policy(model, nitrogen_charge=2.0)

        assert farmer.levy_free_leaching == 0.0
        assert farmer.taxable_leaching == pytest.approx(farmer.leaching)

    def test_the_shipped_config_sets_a_standard(self, farm):
        """Not zero, so turning the charge on does not bill every kilogram.

        Harmless while the charge is off, which is why it can be set here
        without the counterfactual moving.
        """
        model, farmer = farm

        assert (
            model.config.coupled_config.policy.levy_free_concentration == 5.0
        )
        assert farmer.policy is None, "An allowance alone is not an instrument."

    def test_the_allowance_is_the_standard_times_the_drainage(self, farm):
        """11.3 mgN/l over 300 mm is 3.39 gN/m2, which is 34 kgN/ha."""
        model, farmer = farm
        set_policy(
            model, nitrogen_charge=2.0, levy_free_concentration=11.3
        )
        farmer.runoff = 300.0

        assert farmer.levy_free_leaching == pytest.approx(3.39)
        assert farmer.levy_free_leaching * GN_M2_TO_KGN_HA == pytest.approx(
            33.9
        )

    def test_a_wetter_field_may_lose_more_before_it_pays(self, farm):
        """Not a loophole: more water is the same nitrogen more dilute."""
        model, farmer = farm
        set_policy(
            model, nitrogen_charge=2.0, levy_free_concentration=11.3
        )

        farmer.runoff = 150.0
        dry_allowance = farmer.levy_free_leaching
        farmer.runoff = 500.0
        wet_allowance = farmer.levy_free_leaching

        assert wet_allowance > dry_allowance
        assert wet_allowance / dry_allowance == pytest.approx(500.0 / 150.0)

    def test_a_farm_under_the_standard_owes_nothing(self, farm):
        model, farmer = farm
        set_policy(
            model, nitrogen_charge=2.0, levy_free_concentration=11.3
        )
        farmer.runoff = 300.0
        farmer.leaching = 3.0

        assert farmer.leaching < farmer.levy_free_leaching
        assert farmer.taxable_leaching == 0.0
        assert bill(farmer) == 0.0

    def test_only_the_excess_is_charged(self, farm):
        model, farmer = farm
        set_policy(
            model, nitrogen_charge=2.0, levy_free_concentration=11.3
        )
        farmer.runoff = 300.0
        farmer.leaching = 5.39

        assert farmer.taxable_leaching == pytest.approx(2.0)
        expected = 2.0 * GN_M2_TO_KGN_HA * 2.0 * farmer.farm_size
        assert bill(farmer) == pytest.approx(expected)

    def test_a_farm_already_exempt_expects_nothing_from_leaching_less(
        self, farm, monkeypatch
    ):
        """The kink, and the reason the three uses share one function.

        Reading the rate straight off the change would promise this farm money
        the tax office would never hand over, because it is not paying any.
        """
        model, farmer = farm
        set_policy(
            model, nitrogen_charge=5.0, levy_free_concentration=11.3
        )
        farmer.runoff = 300.0
        farmer.leaching = 1.0
        monkeypatch.setattr(
            type(farmer.behaviour),
            "anticipated_absolute_change",
            lambda self, metric, old, new: (
                -0.5 if metric == "leaching" else 0.0
            ),
            raising=False,
        )

        transfer = farmer.behaviour.anticipated_policy_transfer(
            COVER_CROP_BUNDLE, benefit_years=10.0
        )

        assert transfer == pytest.approx(0.0)

    def test_a_farm_above_it_saves_only_down_to_the_line(
        self, farm, monkeypatch
    ):
        """Halving a loss that is barely over the line saves less than half."""
        model, farmer = farm
        set_policy(
            model, nitrogen_charge=5.0, levy_free_concentration=11.3
        )
        farmer.runoff = 300.0
        farmer.leaching = 4.39
        monkeypatch.setattr(
            type(farmer.behaviour),
            "anticipated_absolute_change",
            lambda self, metric, old, new: (
                -2.0 if metric == "leaching" else 0.0
            ),
            raising=False,
        )

        # Taxable falls from 1.0 to 0, so the saving is 1.0 gN/m2 and not 2.0.
        expected = 1.0 * GN_M2_TO_KGN_HA * 5.0 * farmer.farm_size * 10.0
        transfer = farmer.behaviour.anticipated_policy_transfer(
            COVER_CROP_BUNDLE, benefit_years=10.0
        )

        assert transfer == pytest.approx(expected)


class TestTheYearsAnInstrumentActs:
    """A scheme runs over a span, and can be withdrawn as well as introduced.

    Withdrawal is the more interesting half. The carbon payment and the
    practice subsidies are absolute transfers, so left on forever they simply
    accumulate in the capital stock; ending one asks whether the practice
    survives without it or whether the adoption was only ever rented.
    """

    def test_both_ends_are_inclusive(self, farm):
        model, farmer = farm
        year = model.lpjml.sim_year
        set_policy(
            model, nitrogen_charge=2.0, start_year=year, end_year=year
        )

        assert farmer.policy is not None
        assert bill(farmer) > 0.0

    def test_nothing_acts_after_the_end_year(self, farm):
        model, farmer = farm
        year = model.lpjml.sim_year
        set_policy(
            model, nitrogen_charge=2.0, start_year=year - 5, end_year=year - 1
        )

        assert farmer.policy is None
        assert bill(farmer) == 0.0

    def test_the_span_defaults_to_the_coupled_run(self, farm):
        """Said nowhere in the policy block, so it cannot go out of step.

        The fixture couples in 2023 and stops in 2030, and those are the years
        an instrument takes when the block gives none.
        """
        model, farmer = farm
        policy = set_policy(model, nitrogen_charge=2.0)
        del policy.start_year
        del policy.end_year

        settings = farmer.policy

        assert (settings.start_year, settings.end_year) == (2023, 2030)
        assert settings.nitrogen_charge > 0.0

    def test_the_default_start_is_the_coupling_year_not_the_first_year(self):
        """The run starts in 2001 but the farms only join it in 2023.

        Taking the first simulated year would put instruments across the whole
        historic period, which is the mistake a literal start_year invites and
        the reason for reading it off the run instead.
        """
        config = SimpleNamespace(
            firstyear=2001, lastyear=2030, start_coupling=2023
        )

        assert default_policy_span(config) == (2023, 2030)

    def test_a_stated_year_still_wins_over_the_default(self, farm):
        """Phasing in later than the coupling stays possible."""
        model, farmer = farm
        policy = set_policy(model, nitrogen_charge=2.0)
        del policy.end_year
        policy.start_year = model.lpjml.sim_year + 1

        assert farmer.policy is None

    def test_an_instrument_can_carry_its_own_window(self, farm):
        """A carbon price introduced later than everything else."""
        model, farmer = farm
        year = model.lpjml.sim_year
        set_policy(
            model,
            carbon_price=50.0,
            nitrogen_charge=2.0,
            start_year=year,
            windows={"carbon_price": [year + 10, year + 20]},
        )

        assert farmer.policy.carbon_price == 0.0
        assert farmer.policy.nitrogen_charge > 0.0

    def test_one_can_be_withdrawn_while_another_runs_on(self, farm):
        model, farmer = farm
        year = model.lpjml.sim_year
        set_policy(
            model,
            nitrogen_charge=2.0,
            cover_crop_subsidy=50.0,
            start_year=year - 5,
            windows={"cover_crop": [year - 5, year - 1]},
        )

        assert farmer.policy.cover_crop_subsidy == 0.0
        assert farmer.policy.residue_subsidy == 0.0
        assert farmer.policy.nitrogen_charge > 0.0

    def test_a_withdrawn_subsidy_stops_being_anticipated(self, farm):
        """Or a farm would keep switching for money that has stopped."""
        model, farmer = farm
        year = model.lpjml.sim_year
        set_policy(
            model,
            cover_crop_subsidy=50.0,
            start_year=year - 5,
            end_year=year - 1,
        )
        target = farmer.behaviour.practice_bundle.change_practices(cover_crop=1)

        assert farmer.behaviour.anticipated_policy_transfer(target, 10.0) == 0.0

    def test_an_unknown_instrument_is_refused(self, farm):
        model, farmer = farm
        set_policy(model, nitrogen_charge=2.0, windows={"nitrogen_tax": [1, 2]})

        with pytest.raises(ValueError, match="no instrument called"):
            farmer.policy

    def test_a_window_that_ends_before_it_starts_is_refused(self, farm):
        """Silently never acting is the worst way for a typo to behave."""
        model, farmer = farm
        set_policy(
            model, nitrogen_charge=2.0, windows={"nitrogen_charge": [2040, 2030]}
        )

        with pytest.raises(ValueError, match="before it starts"):
            farmer.policy


class TestPracticeSubsidies:
    """Payment per hectare for holding a practice.

    These used to be negative components inside ``practice_costs``, which hid
    a live instrument in the cost stack and made the policy block's claim to
    be the counterfactual untrue. Moving them here changed how they are paid,
    not just where they are written: as a cost they only ever reached capital
    through a difference against the baseline bundle, so a farm that already
    held the practice was paid nothing.
    """

    RESIDUE_BUNDLE = ManagementBundle.from_practices(0, 0, 1)
    BOTH_BUNDLE = ManagementBundle.from_practices(0, 1, 1)

    def test_the_default_pays_nothing(self, farm):
        model, farmer = farm

        assert farmer.policy is None
        assert subsidy(farmer) == 0.0

    def test_a_subsidy_alone_is_an_instrument(self, farm):
        """No carbon price, no charge, and the policy is still in force."""
        model, farmer = farm
        set_policy(model, residue_subsidy=25.0)

        assert farmer.policy is not None
        assert credit(farmer) == 0.0
        assert bill(farmer) == 0.0

    def test_it_is_paid_for_holding_not_for_switching(self, farm):
        """The substantive change from its old life as a cost component.

        A farm that has always retained residue draws the payment every year.
        Priced as a cost it would net out against its own baseline bundle and
        this farm would see nothing.
        """
        model, farmer = farm
        set_policy(model, residue_subsidy=25.0)
        farmer.baseline_bundle = self.RESIDUE_BUNDLE

        held = farmer.practice_subsidy_for(self.RESIDUE_BUNDLE)

        assert held == pytest.approx(25.0 * farmer.farm_size)
        assert farmer.get_bundle_costs_difference(
            self.RESIDUE_BUNDLE, self.RESIDUE_BUNDLE
        ) == pytest.approx(0.0), "Unchanged practice, so no cost difference."

    def test_the_practices_add_up(self, farm):
        model, farmer = farm
        set_policy(model, cover_crop_subsidy=50.0, residue_subsidy=25.0)

        assert farmer.practice_subsidy_for(
            self.BOTH_BUNDLE
        ) == pytest.approx(75.0 * farmer.farm_size)

    def test_a_bundle_holding_neither_is_paid_nothing(self, farm):
        model, farmer = farm
        set_policy(model, cover_crop_subsidy=50.0, residue_subsidy=25.0)
        bare = ManagementBundle.from_practices(1, 0, 0)

        assert farmer.practice_subsidy_for(bare) == 0.0

    def test_it_follows_commitment_not_certification(self, farm):
        """Stubble that happens to clear 30% cover never enrolled in anything."""
        model, farmer = farm
        set_policy(model, residue_subsidy=25.0)

        # Certified by litter cover, but the farm never decided to retain.
        farmer.behaviour.practice_bundle = self.RESIDUE_BUNDLE
        farmer.behaviour.residue_committed = False

        assert farmer.behaviour.committed_bundle.residue_on_field == 0
        assert subsidy(farmer) == 0.0

        farmer.behaviour.residue_committed = True
        assert subsidy(farmer) == pytest.approx(
            25.0 * farmer.farm_size
        )

    def test_it_scales_with_income_like_the_other_local_instruments(self, farm):
        model, farmer = farm
        policy = model.config.coupled_config.policy
        policy.practice_subsidies = {
            "cover_crop": [10.0, 60.0],
            "residue_on_field": [0.0, 0.0],
        }
        policy.start_year = model.lpjml.sim_year

        ratio = farmer.cell.country.gdp_cost_ratio
        expected = 10.0 + (60.0 - 10.0) * min(max(ratio, 0.0), 1.0)

        assert farmer.policy.cover_crop_subsidy == pytest.approx(expected)

    def test_nothing_is_paid_before_the_start_year(self, farm):
        model, farmer = farm
        set_policy(
            model,
            residue_subsidy=25.0,
            start_year=model.lpjml.sim_year + 5,
        )

        assert farmer.policy is None
        assert subsidy(farmer) == 0.0

    def test_the_anticipated_payment_grows_with_the_horizon(self, farm):
        """A flow, unlike the carbon payment, so the horizon multiplies it."""
        model, farmer = farm
        set_policy(model, cover_crop_subsidy=50.0)
        current = farmer.behaviour.practice_bundle
        target = current.change_practices(cover_crop=1)

        short = farmer.behaviour.anticipated_policy_transfer(target, 5.0)
        long = farmer.behaviour.anticipated_policy_transfer(target, 20.0)

        assert short > 0.0
        assert long == pytest.approx(short * 4.0)

    def test_giving_the_practice_up_forfeits_the_payment(self, farm):
        model, farmer = farm
        set_policy(model, cover_crop_subsidy=50.0)
        farmer.behaviour.practice_bundle = self.BOTH_BUNDLE
        target = self.BOTH_BUNDLE.change_practices(cover_crop=0)

        assert farmer.behaviour.anticipated_policy_transfer(target, 10.0) < 0.0

    def test_it_is_no_longer_a_cost_component(self, farm):
        """Regression guard: counted in both places it would pay twice."""
        model, farmer = farm
        costs = farmer.practice_costs

        for practice in ("cover_crop", "residue_on_field", "tillage"):
            components = getattr(costs, practice).cost_components
            assert "subsidy" not in components, (
                f"{practice} still carries a subsidy cost component; it is "
                "paid through policy.practice_subsidies now."
            )


class TestTheResidueSubsidyAsACostShare:
    """Retention paid as a fraction of the straw income actually given up.

    The form EQIP and the CAP agri-environment payments take, and the one that
    survives this model's residue cost being derived per farm rather than
    posted: ``revenue x 0.125 x`` the share of straw the region removes. That
    last factor spans roughly 0.05 to 0.85 across cells, so a flat rate is a
    wildly varying cost-share while a stated share is exactly itself.
    """

    RESIDUE_BUNDLE = ManagementBundle.from_practices(0, 0, 1)
    SELLING_BUNDLE = ManagementBundle.from_practices(0, 0, 0)

    def test_it_pays_the_stated_fraction_of_the_cost(self, farm):
        model, farmer = farm
        set_policy(model, residue_subsidy_share=0.75)

        paid = farmer.practice_subsidy_for(self.RESIDUE_BUNDLE)

        assert farmer.baseline_residue_cost > 0.0, "Otherwise nothing is shown."
        assert paid == pytest.approx(0.75 * farmer.baseline_residue_cost)

    def test_the_share_is_of_the_cost_the_farm_is_actually_charged(self, farm):
        """The point of the instrument: the two read the same figure.

        A retaining farm is charged ``baseline_residue_cost``, so a share of
        that leaves it bearing exactly the remainder.
        """
        model, farmer = farm
        set_policy(model, residue_subsidy_share=0.75)

        charged = farmer.get_residue_cost_for_bundle(1)
        paid = farmer.practice_subsidy_for(self.RESIDUE_BUNDLE)

        assert charged - paid == pytest.approx(0.25 * charged)

    def test_a_farm_that_sells_is_paid_nothing(self, farm):
        model, farmer = farm
        set_policy(model, residue_subsidy_share=0.75)

        assert farmer.practice_subsidy_for(self.SELLING_BUNDLE) == 0.0

    def test_it_is_an_instrument_on_its_own(self, farm):
        model, farmer = farm
        set_policy(model, residue_subsidy_share=0.5)

        assert farmer.policy is not None
        assert farmer.policy.any_in_force

    def test_it_does_not_act_outside_its_window(self, farm):
        """A window names the instrument, not one of its two shapes.

        A carbon price runs alongside so that the policy record survives at
        all: with every instrument out of its window there is nothing in
        force and ``farmer.policy`` is None.
        """
        model, farmer = farm
        year = model.lpjml.sim_year
        set_policy(
            model,
            carbon_price=50.0,
            residue_subsidy_share=0.75,
            windows={"residue_on_field": [year + 5, year + 10]},
        )

        assert farmer.policy.residue_subsidy_share == 0.0
        assert farmer.practice_subsidy_for(self.RESIDUE_BUNDLE) == 0.0

    def test_nothing_is_paid_before_the_start_year(self, farm):
        model, farmer = farm
        set_policy(
            model,
            residue_subsidy_share=0.75,
            start_year=model.lpjml.sim_year + 1,
        )

        assert farmer.practice_subsidy_for(self.RESIDUE_BUNDLE) == 0.0

    def test_the_farm_anticipates_it_when_weighing_retention(self, farm):
        model, farmer = farm
        set_policy(model, residue_subsidy_share=0.75)

        behaviour = farmer.behaviour
        gap = behaviour.anticipated_policy_transfer(
            self.RESIDUE_BUNDLE, benefit_years=1
        ) - behaviour.anticipated_policy_transfer(
            self.SELLING_BUNDLE, benefit_years=1
        )

        assert gap == pytest.approx(0.75 * farmer.baseline_residue_cost)

    @staticmethod
    def _resolve(gdp_cost_ratio=0.5, **subsidies):
        return PolicySettings.from_config(
            {"practice_subsidies": subsidies, "start_year": 0},
            gdp_cost_ratio=gdp_cost_ratio,
        )

    def test_a_ranged_share_is_refused(self):
        """The cost it applies to is already local; a range would double it."""
        with pytest.raises(ValueError, match="single fraction"):
            self._resolve(residue_on_field_share=[0.5, 0.75])

    def test_a_share_above_full_compensation_is_refused(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            self._resolve(residue_on_field_share=1.5)

    def test_the_same_share_resolves_whatever_the_country_earns(self):
        """Unlike the flat rate, which interpolates on the GDP ratio."""
        poor = self._resolve(gdp_cost_ratio=0.0, residue_on_field_share=0.75)
        rich = self._resolve(gdp_cost_ratio=1.0, residue_on_field_share=0.75)

        assert poor.residue_subsidy_share == rich.residue_subsidy_share == 0.75

        flat_poor = self._resolve(
            gdp_cost_ratio=0.0, residue_on_field=[10.0, 25.0]
        ).residue_subsidy
        flat_rich = self._resolve(
            gdp_cost_ratio=1.0, residue_on_field=[10.0, 25.0]
        ).residue_subsidy
        assert flat_poor != flat_rich, "The flat rate is the one that scales."

    def test_setting_both_shapes_is_refused(self):
        """Two ways of writing one instrument, so they must not be added."""
        with pytest.raises(ValueError, match="pay a farm twice"):
            self._resolve(
                residue_on_field=[20.0, 30.0], residue_on_field_share=0.75
            )


class TestTheTaxBaseIsWholeCellLeaching:
    """Leaching is read as whole-cell ``leaching``, not stand-specific.

    This matches the boundaries package measurement: what matters for water
    quality is the nitrogen that actually reaches the water body, regardless
    of which stand it came from. The aquifer does not know which hectare
    contributed the nitrogen.
    """

    def test_it_reads_whole_cell_leaching(self, farm):
        model, farmer = farm

        assert "leaching" in farmer.cell.from_earth
        assert farmer.leaching == pytest.approx(farmer.cell_leaching)

    def test_it_is_per_m2_of_cell(self, farm):
        """The flux is what reaches the water body, not per-hectare of cropland."""
        model, farmer = farm
        raw = float(
            farmer.get_from_earth("leaching", as_scalar=True, time_idx=-1)
        )

        # No normalization by cropped fraction - raw cell flux
        assert farmer.cell_leaching == pytest.approx(raw)

    def test_the_baseline_is_built_on_the_same_quantity(self, farm):
        """Scored rate and taxed flux must not be measured differently."""
        model, farmer = farm
        n_years = len(farmer.cell.from_earth.time)
        record = [agricultural_leaching(farmer, time_idx=i) for i in range(n_years)]

        assert farmer.baseline_leaching == pytest.approx(float(np.mean(record)))


class TestOnlyTheLocalInstrumentVariesByIncome:
    """Nitrogen damage is local and carbon damage is not, so only one scales.

    The nitrogen rate a country can set depends on the aquifer, on who drinks
    from it and on what its farms can bear, so it is interpolated on the same
    GDP ratio as every practice cost. A tonne of CO2 is worth the same wherever
    it is put away, so a price that varied by income would misprice it and
    would draw sequestration towards whoever pays most rather than whoever
    abates cheapest.
    """

    @staticmethod
    def resolve(nitrogen_charge, gdp_cost_ratio):
        return PolicySettings.from_config(
            {"nitrogen_charge": nitrogen_charge, "start_year": 0},
            gdp_cost_ratio=gdp_cost_ratio,
        ).nitrogen_charge

    def test_a_poor_country_is_charged_the_low_end(self):
        assert self.resolve([0.5, 3.0], 0.0) == pytest.approx(0.5)

    def test_a_rich_country_is_charged_the_high_end(self):
        assert self.resolve([0.5, 3.0], 1.0) == pytest.approx(3.0)

    def test_in_between_interpolates(self):
        assert self.resolve([0.5, 3.0], 0.25) == pytest.approx(1.125)

    def test_a_single_figure_means_the_same_rate_everywhere(self):
        assert self.resolve(2.0, 0.0) == pytest.approx(2.0)
        assert self.resolve(2.0, 1.0) == pytest.approx(2.0)

    def test_a_ranged_carbon_price_is_refused(self):
        """Averaging it would bury the reason it must not vary."""
        with pytest.raises(ValueError, match="single global figure"):
            PolicySettings.from_config(
                {"carbon_price": [30.0, 185.0], "start_year": 0}
            )

    def test_the_farm_is_charged_at_its_own_country_rate(self, farm):
        model, farmer = farm
        policy = model.config.coupled_config.policy
        policy.nitrogen_charge = [0.5, 3.0]
        policy.start_year = model.lpjml.sim_year

        ratio = farmer.cell.country.gdp_cost_ratio
        expected_rate = 0.5 + (3.0 - 0.5) * min(max(ratio, 0.0), 1.0)

        assert farmer.policy.nitrogen_charge == pytest.approx(expected_rate)
        # On the taxable part, since the shipped config carries an allowance.
        assert bill(farmer) == pytest.approx(
            farmer.taxable_leaching
            * GN_M2_TO_KGN_HA
            * expected_rate
            * farmer.farm_size
        )
        assert 0.0 < farmer.taxable_leaching < farmer.leaching


class TestAnticipationTreatsStocksAndFlowsApart:
    """Both payments are on a flow, so the horizon multiplies both.

    The carbon payment is settled each year on the slope of the farm's soil
    carbon, so its expectation is read off peers' slopes and repeats every
    year like the nitrogen saving. Peers' carbon *levels* do not enter it: a
    level difference can be negative where the slope improved.
    """

    def test_the_carbon_term_grows_with_the_horizon(self, farm):
        model, farmer = farm
        set_policy(model, carbon_price=50.0)
        year = model.lpjml.sim_year
        current = farmer.behaviour.practice_bundle
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, current, None, {}, year, trends={"soilc": 0.0})
            remember_either_type(
                f, COVER_CROP_BUNDLE, {}, year, trends={"soilc": 0.01}
            )

        short = farmer.behaviour.anticipated_policy_transfer(COVER_CROP_BUNDLE, 5.0)
        long = farmer.behaviour.anticipated_policy_transfer(COVER_CROP_BUNDLE, 20.0)

        assert short > 0.0, "The peers need to show a carbon gain worth paying."
        assert long == pytest.approx(short * 4.0)

    def test_a_level_gain_without_a_slope_gain_is_not_paid(self, farm):
        """A one-off step up in the stock is not a yearly flow."""
        model, farmer = farm
        set_policy(model, carbon_price=50.0)
        year = model.lpjml.sim_year
        current = farmer.behaviour.practice_bundle
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, current, None, {"soilc": 0.0}, year)
            remember_either_type(f, COVER_CROP_BUNDLE, {"soilc": 0.3}, year)

        assert farmer.behaviour.anticipated_policy_transfer(
            COVER_CROP_BUNDLE, 10.0
        ) == pytest.approx(0.0)

    def test_the_nitrogen_term_grows_with_the_horizon(self, farm):
        model, farmer = farm
        set_policy(model, nitrogen_charge=2.0)
        year = model.lpjml.sim_year
        current = farmer.behaviour.practice_bundle
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, current, None, {"leaching": 0.0}, year)
            remember_either_type(f, COVER_CROP_BUNDLE, {"leaching": -0.3}, year)

        short = farmer.behaviour.anticipated_policy_transfer(COVER_CROP_BUNDLE, 5.0)
        long = farmer.behaviour.anticipated_policy_transfer(COVER_CROP_BUNDLE, 20.0)

        assert short > 0.0, "Less nitrogen lost is a smaller bill, so cash in."
        assert long == pytest.approx(short * 4.0)

    def test_an_expected_carbon_loss_is_not_an_expected_invoice(self, farm):
        """Floored in the anticipation as it is in the payment, or the farm
        would decline a switch over a bill it would never receive. Pinned to a
        farm drawing no payment now, which has none to forfeit."""
        model, farmer = farm
        set_policy(model, carbon_price=50.0)
        year = model.lpjml.sim_year
        current = farmer.behaviour.practice_bundle
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, current, None, {}, year, trends={"soilc": 0.0})
            remember_either_type(
                f, COVER_CROP_BUNDLE, {}, year, trends={"soilc": -0.01}
            )
        own = zero_metrics()
        farmer.behaviour.performance_tracker = SimpleNamespace(
            n=5, trend=own, clean_trend=own
        )

        transfer = farmer.behaviour.anticipated_policy_transfer(
            COVER_CROP_BUNDLE, 10.0
        )

        assert transfer == 0.0


class TestTheHistoricPeriodIsInstrumentFree:
    """Whatever the scenario sets, the years the farm learned from had none."""

    def test_the_remembered_profit_does_not_move_with_the_policy(
        self, ca_model_builder
    ):
        from inseeds.components.farming.ca_management import (
            ManagementPerformanceTracker,
        )

        model = ca_model_builder(seed=707)
        farmer = list(model.world.farmers)[0]
        year = model.lpjml.sim_year

        _, without = ManagementPerformanceTracker._load_history(farmer)
        set_policy(model, carbon_price=200.0, nitrogen_charge=13.0)
        _, with_policy = ManagementPerformanceTracker._load_history(farmer)

        assert [row["profit"] for row in with_policy] == [
            row["profit"] for row in without
        ]


class TestTheEstimatorSeparatesTheTwoCoverCrops:
    """One bundle, two crops: the comparison the type decision rests on."""

    def test_two_types_on_one_bundle_are_a_real_comparison(self, farm):
        model, farmer = farm
        year = model.lpjml.sim_year
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, COVER_CROP_BUNDLE, COVER_CROP_NONLEGUME, {"yield": 0.0}, year)
            remember(f, COVER_CROP_BUNDLE, COVER_CROP_LEGUME, {"yield": 0.2}, year)

        gain = farmer.behaviour.anticipated_change_between(
            "yield",
            (COVER_CROP_BUNDLE, COVER_CROP_NONLEGUME),
            (COVER_CROP_BUNDLE, COVER_CROP_LEGUME),
        )

        assert gain == pytest.approx(np.expm1(0.2))

    def test_a_key_against_itself_is_no_change(self, farm):
        model, farmer = farm
        key = (COVER_CROP_BUNDLE, COVER_CROP_LEGUME)

        assert farmer.behaviour.anticipated_change_between("yield", key, key) == 0.0


class TestTheTypeChoiceIsEconomic:
    """Which crop gets preferred, and why.

    The preference has no random draw in it, so these read it directly.
    """

    def test_with_no_evidence_the_cheaper_seed_wins(self, farm):
        """Nothing says the legume pays, so nobody buys the dearer bag."""
        model, farmer = farm

        assert farmer.indicate_cover_crop_type() == COVER_CROP_NONLEGUME

    def test_enough_yield_evidence_buys_the_legume(self, farm):
        model, farmer = farm
        year = model.lpjml.sim_year
        bundle = farmer.behaviour.practice_bundle
        if not bundle.cover_crop:
            bundle = bundle.change_practices(cover_crop=1)
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, bundle, COVER_CROP_NONLEGUME, {"yield": 0.0}, year)
            remember(f, bundle, COVER_CROP_LEGUME, {"yield": 0.5}, year)

        assert farmer.indicate_cover_crop_type() == COVER_CROP_LEGUME

    def test_a_nitrogen_charge_argues_for_the_catch_crop(self, farm):
        """The instrument's whole purpose, in one comparison.

        Peers show the legume yielding more and leaching more. Without a charge
        the extra yield carries it; with one, the nitrogen it lets go is priced
        and the catch crop wins on the same evidence.
        """
        model, farmer = farm
        year = model.lpjml.sim_year
        bundle = farmer.behaviour.practice_bundle
        if not bundle.cover_crop:
            bundle = bundle.change_practices(cover_crop=1)
        for f in [farmer, *farmer.neighbourhood]:
            remember(
                f,
                bundle,
                COVER_CROP_NONLEGUME,
                {"yield": 0.0, "leaching": 0.0},
                year,
            )
            remember(
                f,
                bundle,
                COVER_CROP_LEGUME,
                {"yield": 0.5, "leaching": 1.0},
                year,
            )

        assert farmer.indicate_cover_crop_type() == COVER_CROP_LEGUME

        set_policy(model, nitrogen_charge=50.0)
        assert farmer.indicate_cover_crop_type() == COVER_CROP_NONLEGUME

    def test_the_preference_is_a_question_not_a_draw(self, farm):
        """Costing a bundle asks this repeatedly; it must answer the same."""
        model, farmer = farm

        first = farmer.indicate_cover_crop_type()
        answers = {farmer.indicate_cover_crop_type() for _ in range(50)}

        assert answers == {first}


class TestTypeExplorationBreaksTheLockIn:
    """Without it the estimator would be reading a record its own answer wrote."""

    @staticmethod
    def _cover_crop_bundle(farmer):
        bundle = farmer.behaviour.practice_bundle
        return bundle if bundle.cover_crop else bundle.change_practices(cover_crop=1)

    def test_exploration_sows_the_other_crop(self, farm, monkeypatch):
        model, farmer = farm
        monkeypatch.setattr(
            type(farmer.behaviour), "explores_cover_crop_type", lambda self: True
        )

        # With no evidence the preference is the catch crop, so an exploring
        # farm must sow the legume - the one nobody would otherwise try.
        assert farmer.indicate_cover_crop_type() == COVER_CROP_NONLEGUME
        farmer.behaviour.apply_bundle(self._cover_crop_bundle(farmer))

        assert farmer.cover_crop == COVER_CROP_LEGUME

    def test_without_the_draw_the_preference_is_what_gets_sown(
        self, farm, monkeypatch
    ):
        model, farmer = farm
        monkeypatch.setattr(
            type(farmer.behaviour), "explores_cover_crop_type", lambda self: False
        )
        farmer.behaviour.apply_bundle(self._cover_crop_bundle(farmer))

        assert farmer.cover_crop == COVER_CROP_NONLEGUME

    def test_it_fires_at_its_own_rate(self, farm):
        """Asked once per commitment, on the crop's rate and not the bundle's."""
        model, farmer = farm
        behaviour = farmer.behaviour
        expected = behaviour.get_aft_param("cover_crop_type_exploration_prob")

        np.random.seed(11)
        fired = sum(behaviour.explores_cover_crop_type() for _ in range(20000))

        assert expected > 0.0
        assert fired / 20000 == pytest.approx(expected, abs=0.01)

    def test_the_rate_is_below_the_bundle_rate(self, farm):
        """The draw is over two crops, not eight bundles.

        At the bundle rate a standing fifteen to thirty per cent of farms would
        grow the crop they judged worse - a permanent randomisation rather than
        a probe, and a large one beside a legume advantage worth about a per
        cent of revenue.
        """
        model, farmer = farm
        behaviour = farmer.behaviour

        assert behaviour.get_aft_param(
            "cover_crop_type_exploration_prob"
        ) < behaviour.get_aft_param("exploration_base_prob")


class TestTheCropIsReopenedAtDecisionPoints:
    """The crop is not a bundle field, so the peer scan cannot re-open it.

    A farm whose neighbours run the same bundle with the other crop reads that
    as TARGET_SAME and stops. Left there, the crop would be settled once at the
    last bundle change and a settled farm would sow against a charge introduced
    afterwards. So a decision that leaves the bundle alone still re-asks.
    """

    @staticmethod
    def _sowing(farmer, crop, served=True):
        """Put a cover crop in the ground with an episode's worth of record.

        ``served`` ages the crop past its commitment, which is the state these
        tests are about: a farm that has held one long enough to be free to
        change. Pass False to leave it inside the term.
        """
        bundle = farmer.behaviour.practice_bundle
        if not bundle.cover_crop:
            bundle = bundle.change_practices(cover_crop=1)
            farmer.behaviour.practice_bundle = bundle
        farmer.cover_crop = crop
        farmer.behaviour.performance_tracker.n = 9
        farmer.behaviour._cover_crop_type_years = (
            farmer.behaviour.get_aft_param("cover_crop_type_commitment_years")
            if served
            else 0
        )
        return bundle

    def test_a_charge_turns_a_settled_farm_off_the_legume(self, farm):
        """The case the whole mechanism exists for."""
        model, farmer = farm
        year = model.lpjml.sim_year
        bundle = self._sowing(farmer, COVER_CROP_LEGUME)
        for f in [farmer, *farmer.neighbourhood]:
            remember(
                f, bundle, COVER_CROP_NONLEGUME, {"yield": 0.0, "leaching": 0.0}, year
            )
            remember(
                f, bundle, COVER_CROP_LEGUME, {"yield": 0.5, "leaching": 1.0}, year
            )

        assert farmer.indicate_cover_crop_type() == COVER_CROP_LEGUME
        farmer.behaviour.reconsider_cover_crop_type()
        assert farmer.cover_crop == COVER_CROP_LEGUME, (
            "Nothing changed, so the farm should have kept its crop."
        )

        set_policy(model, nitrogen_charge=50.0)
        farmer.behaviour.reconsider_cover_crop_type()

        assert farmer.cover_crop == COVER_CROP_NONLEGUME

    def test_the_switch_closes_the_episode(self, farm):
        """A new crop is a new practice, counted from scratch."""
        model, farmer = farm
        year = model.lpjml.sim_year
        bundle = self._sowing(farmer, COVER_CROP_LEGUME)
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, bundle, COVER_CROP_LEGUME, {"yield": -0.5}, year)

        farmer.behaviour.reconsider_cover_crop_type()

        assert farmer.cover_crop == COVER_CROP_NONLEGUME
        assert farmer.behaviour.performance_tracker.n == 1

    def test_the_outgoing_record_is_filed_under_the_crop_that_earned_it(self, farm):
        """Not under the crop that replaced it, which would poison the estimator."""
        model, farmer = farm
        year = model.lpjml.sim_year
        bundle = self._sowing(farmer, COVER_CROP_LEGUME)
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, bundle, COVER_CROP_LEGUME, {"yield": -0.5}, year)

        farmer.behaviour.reconsider_cover_crop_type()

        assert farmer.behaviour.performance_memory.get(
            (bundle, COVER_CROP_LEGUME)
        ).duration == 9

    def test_a_bundle_transition_files_the_old_crop_under_its_own_name(
        self, farm, monkeypatch
    ):
        """The record has to be filed before the crop it describes is replaced.

        A farm changing tillage while keeping a cover crop may come out of it
        sowing the other one. Applying the bundle first would put the legume's
        nine years under the catch crop's name, and the estimator reads those
        slots to decide which crop pays.
        """
        model, farmer = farm
        monkeypatch.setattr(
            type(farmer.behaviour), "explores_cover_crop_type", lambda self: False
        )
        old_bundle = self._sowing(farmer, COVER_CROP_LEGUME)
        new_bundle = old_bundle.change_practices(
            tillage=1 - old_bundle.tillage
        )

        farmer.behaviour.commit_bundle(new_bundle)

        assert farmer.cover_crop == COVER_CROP_NONLEGUME, (
            "With no evidence the farm should have moved to the cheaper seed, "
            "which is what makes the keying observable at all."
        )
        assert farmer.behaviour.performance_memory.get(
            (old_bundle, COVER_CROP_LEGUME)
        ).duration == 9
        assert farmer.behaviour.performance_memory.get(
            (old_bundle, COVER_CROP_NONLEGUME)
        ).duration == 0

    @staticmethod
    def _watch(monkeypatch, farmer):
        """Record the years the farm re-asks, with no target so none transitions."""
        calls = []
        behaviour = type(farmer.behaviour)
        monkeypatch.setattr(
            behaviour, "reconsider_cover_crop_type", lambda self: calls.append(1)
        )
        for finder in (
            "most_promising_bundle_local",
            "most_promising_bundle_country",
            "most_promising_bundle_cluster",
            "maybe_explore_bundle",
        ):
            monkeypatch.setattr(behaviour, finder, lambda self: None)
        return calls

    def test_the_farmer_re_asks_on_a_decision_year(self, farm, monkeypatch):
        """The wiring: a due decision that changed no bundle still re-opens it."""
        model, farmer = farm
        calls = self._watch(monkeypatch, farmer)
        farmer.behaviour._observation_years = 0

        farmer.update(t=model.lpjml.sim_year)

        assert calls

    def test_it_stays_shut_between_decisions(self, farm, monkeypatch):
        """Off the cadence it is not a decision year, so the crop is not re-opened."""
        model, farmer = farm
        calls = self._watch(monkeypatch, farmer)
        farmer.behaviour._observation_years = 5

        farmer.update(t=model.lpjml.sim_year)

        assert not calls

    def test_a_farm_with_no_cover_crop_is_left_alone(self, farm):
        model, farmer = farm
        bundle = farmer.behaviour.practice_bundle.change_practices(cover_crop=0)
        farmer.behaviour.practice_bundle = bundle
        farmer.cover_crop = 0

        farmer.behaviour.reconsider_cover_crop_type()

        assert farmer.cover_crop == 0

    def test_a_transition_year_does_not_re_ask(self, farm, monkeypatch):
        """Or it would undo the exploration draw apply_bundle just made."""
        model, farmer = farm
        monkeypatch.setattr(
            type(farmer.behaviour), "explores_cover_crop_type", lambda self: True
        )
        bundle = farmer.behaviour.practice_bundle
        if not bundle.cover_crop:
            bundle = bundle.change_practices(cover_crop=1)

        farmer.behaviour.commit_bundle(bundle)

        assert farmer.cover_crop == COVER_CROP_LEGUME, (
            "Exploration should have overridden the catch-crop preference."
        )
        # The guard lives in the farmer's update. What it is guarding against
        # is this: re-asking reverses the draw outright. The commitment holds
        # the draw for its term first, so the reversal only becomes possible
        # once the crop has been in the ground long enough to be judged - which
        # is the whole point of giving it a term.
        farmer.behaviour.reconsider_cover_crop_type()
        assert farmer.cover_crop == COVER_CROP_LEGUME, (
            "Inside its term the draw should still be standing."
        )

        farmer.behaviour._cover_crop_type_years = farmer.behaviour.get_aft_param(
            "cover_crop_type_commitment_years"
        )
        farmer.behaviour.reconsider_cover_crop_type()
        assert farmer.cover_crop == COVER_CROP_NONLEGUME


class TestTheCropIsHeldForItsTerm:
    """The type answers to a longer clock than the bundle.

    The two crops differ by one to two per cent of yield against a seed
    premium of much the same size, so the comparison turns on peer records and
    those need episodes long enough to average a few seasons of weather.
    Re-opening it every three to five years splits the records across the two
    crops and leaves the estimator reading its own noise.
    """

    @staticmethod
    def _wants_the_other_crop(farmer, model):
        """A settled legume farm whose evidence now favours the catch crop."""
        bundle = farmer.behaviour.practice_bundle
        if not bundle.cover_crop:
            bundle = bundle.change_practices(cover_crop=1)
            farmer.behaviour.practice_bundle = bundle
        farmer.cover_crop = COVER_CROP_LEGUME
        for f in [farmer, *farmer.neighbourhood]:
            remember(f, bundle, COVER_CROP_LEGUME, {"yield": -0.5},
                     model.lpjml.sim_year)
        assert farmer.indicate_cover_crop_type() == COVER_CROP_NONLEGUME
        return bundle

    def test_inside_the_term_the_crop_stays(self, farm):
        model, farmer = farm
        self._wants_the_other_crop(farmer, model)
        term = farmer.behaviour.get_aft_param("cover_crop_type_commitment_years")
        farmer.behaviour._cover_crop_type_years = term - 1

        farmer.behaviour.reconsider_cover_crop_type()

        assert farmer.cover_crop == COVER_CROP_LEGUME

    def test_once_the_term_is_served_it_may_change(self, farm):
        model, farmer = farm
        self._wants_the_other_crop(farmer, model)
        farmer.behaviour._cover_crop_type_years = farmer.behaviour.get_aft_param(
            "cover_crop_type_commitment_years"
        )

        farmer.behaviour.reconsider_cover_crop_type()

        assert farmer.cover_crop == COVER_CROP_NONLEGUME

    def test_a_change_of_crop_starts_the_term_again(self, farm):
        model, farmer = farm
        self._wants_the_other_crop(farmer, model)
        farmer.behaviour._cover_crop_type_years = farmer.behaviour.get_aft_param(
            "cover_crop_type_commitment_years"
        )

        farmer.behaviour.reconsider_cover_crop_type()

        assert farmer.behaviour._cover_crop_type_years == 0

    def test_changing_bundle_alone_does_not_shorten_it(self, farm, monkeypatch):
        """Or a farm could serve the term by moving between bundles."""
        model, farmer = farm
        monkeypatch.setattr(
            type(farmer.behaviour), "explores_cover_crop_type", lambda self: False
        )
        bundle = farmer.behaviour.practice_bundle
        if not bundle.cover_crop:
            bundle = bundle.change_practices(cover_crop=1)
        farmer.behaviour.apply_bundle(bundle)
        held = farmer.cover_crop
        farmer.behaviour._cover_crop_type_years = 7

        # Same crop, different tillage: the crop has not been replaced, so the
        # seasons it has already served still count.
        farmer.behaviour.apply_bundle(
            bundle.change_practices(tillage=1 - bundle.tillage)
        )

        assert farmer.cover_crop == held
        assert farmer.behaviour._cover_crop_type_years == 7

    def test_the_clock_only_runs_while_a_crop_is_in_the_ground(self, farm):
        """A farm with no cover crop is not serving anything."""
        model, farmer = farm
        farmer.behaviour._cover_crop_type_years = 9
        farmer.cover_crop = 0

        farmer.behaviour.observe()

        assert farmer.behaviour._cover_crop_type_years == 0


class TestTheLegumeEnhancement:
    """The one per-hectare rate that separates the two crops.

    The base cover crop payment is paid whichever is sown, so it cancels in the
    type comparison and cannot steer it. The enhancement does not cancel, which
    is what makes it the instrument a scenario reaches for when the question is
    which crop rather than whether one.
    """

    @staticmethod
    def _cover_crop_bundle(farmer):
        bundle = farmer.behaviour.practice_bundle
        return bundle if bundle.cover_crop else bundle.change_practices(cover_crop=1)

    def test_it_is_paid_only_on_a_legume(self, farm):
        model, farmer = farm
        set_policy(model, cover_crop_subsidy=40.0,
                   cover_crop_legume_subsidy=25.0)
        bundle = self._cover_crop_bundle(farmer)

        legume = farmer.practice_subsidy_for(bundle, COVER_CROP_LEGUME)
        catch = farmer.practice_subsidy_for(bundle, COVER_CROP_NONLEGUME)

        assert legume - catch == pytest.approx(25.0 * farmer.farm_size)

    def test_a_farm_with_no_cover_crop_gets_neither(self, farm):
        model, farmer = farm
        set_policy(model, cover_crop_subsidy=40.0,
                   cover_crop_legume_subsidy=25.0)
        bare = farmer.behaviour.practice_bundle.change_practices(cover_crop=0)

        assert farmer.practice_subsidy_for(bare, COVER_CROP_LEGUME) == 0.0

    def test_it_tips_a_choice_the_seed_premium_would_lose(self, farm):
        """The case the instrument exists for."""
        model, farmer = farm
        assert farmer.indicate_cover_crop_type() == COVER_CROP_NONLEGUME, (
            "With no evidence the cheaper seed should win, which is the "
            "starting point the enhancement has to overturn."
        )

        set_policy(model, cover_crop_legume_subsidy=200.0)

        assert farmer.indicate_cover_crop_type() == COVER_CROP_LEGUME

    def test_the_base_rate_alone_cannot_tip_it(self, farm):
        """Paid on both crops, so it cancels out of the comparison."""
        model, farmer = farm
        set_policy(model, cover_crop_subsidy=200.0)

        assert farmer.indicate_cover_crop_type() == COVER_CROP_NONLEGUME

    def test_a_window_switches_it_off_with_the_rest(self, farm):
        model, farmer = farm
        # A carbon price left in force, so the record itself survives the
        # window and the enhancement can be read as zeroed rather than the
        # whole policy having gone to None.
        set_policy(
            model,
            carbon_price=10.0,
            cover_crop_subsidy=40.0,
            cover_crop_legume_subsidy=200.0,
            windows={"cover_crop": [3000, 3010]},
        )

        assert farmer.policy.cover_crop_legume_subsidy == 0.0
        assert farmer.policy.cover_crop_subsidy == 0.0
        assert farmer.indicate_cover_crop_type() == COVER_CROP_NONLEGUME


class TestTheSeedBillFollowsTheCropSown:
    """Switching to the legume has to cost what the legume costs.

    The type is not a bundle field, so a farm can change crop without the
    bundle moving. Capital moves by the cost of the current bundle against the
    baseline one, and if the baseline leg re-prices itself with whatever is in
    the ground today, both legs carry the same seed cost and the difference
    cancels - the farm sows the dearer crop for free.
    """

    COVER_BUNDLE = ManagementBundle.from_practices(0, 1, 0)

    @staticmethod
    def _seed_gap(farmer):
        seed = farmer.practice_costs.cover_crop
        return (
            seed.seed_cost(COVER_CROP_LEGUME) - seed.seed_cost(COVER_CROP_NONLEGUME)
        ) * farmer.farm_size

    def _settled_on_the_catch_crop(self, farmer):
        """A farm whose baseline already had a cover crop in it."""
        farmer.behaviour.practice_bundle = self.COVER_BUNDLE
        farmer.behaviour.residue_committed = False
        farmer.cover_crop = COVER_CROP_NONLEGUME
        farmer.baseline_bundle = farmer.behaviour.committed_bundle
        farmer.baseline_cover_crop_type = COVER_CROP_NONLEGUME

    def _delta_costs(self, farmer):
        """The quantity update_capital charges against capital."""
        return farmer.get_bundle_costs_difference(
            farmer.baseline_bundle,
            farmer.behaviour.committed_bundle,
            from_type=farmer.baseline_cover_crop_type,
        )

    def test_a_farm_that_started_with_a_cover_crop_pays_the_difference(self, farm):
        _, farmer = farm
        self._settled_on_the_catch_crop(farmer)

        before = self._delta_costs(farmer)
        farmer.cover_crop = COVER_CROP_LEGUME
        after = self._delta_costs(farmer)

        assert after - before == pytest.approx(self._seed_gap(farmer))

    def test_a_farm_that_adopted_one_pays_it_too(self, farm):
        """The case that already worked, kept so the fix cannot regress it."""
        _, farmer = farm
        self._settled_on_the_catch_crop(farmer)
        farmer.baseline_bundle = ManagementBundle.from_practices(0, 0, 0)
        farmer.baseline_cover_crop_type = None

        before = self._delta_costs(farmer)
        farmer.cover_crop = COVER_CROP_LEGUME
        after = self._delta_costs(farmer)

        assert after - before == pytest.approx(self._seed_gap(farmer))

    def test_it_reaches_capital(self, farm):
        """End to end, so update_capital cannot quietly drop the baseline crop."""
        _, farmer = farm
        self._settled_on_the_catch_crop(farmer)

        opening = farmer.capital
        farmer.update_capital()
        on_the_catch_crop = farmer.capital - opening

        farmer.capital = opening
        farmer.cover_crop = COVER_CROP_LEGUME
        farmer.update_capital()
        on_the_legume = farmer.capital - opening

        # The dearer seed leaves the farm that much worse off over the year.
        assert on_the_catch_crop - on_the_legume == pytest.approx(
            self._seed_gap(farmer)
        )

    def test_the_baseline_crop_is_frozen_at_the_start(self, farm):
        """Not re-read later, which is what made the two legs cancel."""
        _, farmer = farm
        recorded = farmer.baseline_cover_crop_type

        farmer.cover_crop = COVER_CROP_LEGUME
        farmer.behaviour.practice_bundle = self.COVER_BUNDLE

        assert farmer.baseline_cover_crop_type == recorded

    def test_a_farm_that_never_had_one_records_no_baseline_crop(self, farm):
        _, farmer = farm
        if farmer.baseline_bundle.cover_crop:
            pytest.skip("fixture farm starts with a cover crop")

        assert farmer.baseline_cover_crop_type is None


class TestPricingReadsTheCropInTheGround:
    """A divergent residue label must not change which crop the farm is billed for.

    ``committed_bundle`` differs from ``practice_bundle`` only in the residue
    flag, which says nothing about what was sown.
    """

    def test_the_committed_bundle_is_priced_on_what_was_sown(self, farm):
        _, farmer = farm
        behaviour = farmer.behaviour
        behaviour.practice_bundle = ManagementBundle.from_practices(0, 1, 1)
        behaviour.residue_committed = False
        farmer.cover_crop = COVER_CROP_LEGUME

        assert behaviour.committed_bundle != behaviour.practice_bundle
        assert behaviour.cover_crop_type_for(behaviour.committed_bundle) == (
            COVER_CROP_LEGUME
        )

    def test_a_candidate_bundle_is_still_priced_on_the_preference(self, farm):
        """The other half of the rule, so the fix does not swallow candidates."""
        _, farmer = farm
        behaviour = farmer.behaviour
        behaviour.practice_bundle = ManagementBundle.from_practices(0, 0, 0)
        behaviour.residue_committed = False
        farmer.cover_crop = COVER_CROP_LEGUME

        candidate = ManagementBundle.from_practices(1, 1, 0)

        assert behaviour.cover_crop_type_for(candidate) == (
            farmer.indicate_cover_crop_type()
        )


class TestTheReportedTransfersAreTheSettledOnes:
    """What the output says was paid is what actually moved capital.

    ``policy`` resolves against ``lpjml.sim_year``, and the coupler advances
    that to the next year as soon as the year's exchange is done - which is
    before the outputs are collected. A transfer recomputed on read therefore
    answers for the following year: zero in the run's final year, because the
    span has ended, and nonzero in the year before a late ``start_year``. It
    went unnoticed for a long time because inside the window the rate is the
    same either way, so only the boundaries disagree. The properties report
    what ``update_capital`` settled on instead, which is also the only way
    ``raw_profit`` can be reconciled against its parts.
    """

    def _charged_farm(self, farm):
        model, farmer = farm
        set_policy(model, nitrogen_charge=5.0)
        farmer.update_capital()
        assert farmer.nitrogen_charge > 0.0, "fixture leaches nothing to charge"
        return model, farmer

    def test_raw_profit_is_the_sum_of_its_reported_parts(self, farm):
        """The reconciliation an output table is read for."""
        _, farmer = self._charged_farm(farm)

        assert farmer.raw_profit == pytest.approx(
            farmer.delta_revenue
            - farmer.delta_costs
            + farmer.carbon_payment
            + farmer.practice_subsidy
            - farmer.nitrogen_charge
        )

    def test_the_charge_survives_the_window_closing_afterwards(self, farm):
        """Standing in for the collector reading a year after the span ends."""
        model, farmer = self._charged_farm(farm)
        settled = farmer.nitrogen_charge

        model.config.coupled_config.policy.end_year = model.lpjml.sim_year - 1

        # Out of force now, so a property that re-resolved would report zero
        # for a bill the farm has already paid out of capital.
        assert bill(farmer) == 0.0
        assert farmer.nitrogen_charge == settled

    def test_a_payment_is_not_reported_before_it_is_made(self, farm):
        """The other boundary: in force on read, but nothing paid yet."""
        model, farmer = farm
        set_policy(model, nitrogen_charge=5.0)

        # No step has run, so nothing has moved capital.
        assert farmer.nitrogen_charge == 0.0
        assert bill(farmer) > 0.0
