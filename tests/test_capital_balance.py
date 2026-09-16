"""The capital stock reconciles against the terms the model reports.

An output table is only worth reading if the numbers in it add up, and for a
while these did not: ``raw_profit`` was documented as the change in capital,
but a transition is charged after ``update_capital`` has run and so lands
outside it, and the payments were resolved again when the output was collected
rather than reported as settled - which answered for the following year,
because the coupler advances ``sim_year`` before the collector runs.

So the balance is pinned here, term by term, in a form that fails if any future
movement of capital is added without being reported alongside it.
"""

import pytest

from inseeds.components.farming.ca_management import ManagementBundle


def force_the_move(monkeypatch, behaviour, target, pathway="local"):
    """Make the next step transition to ``target``, whatever the score says.

    ``behaviour.update`` picks its own candidate, so setting ``proposed_bundle``
    beforehand would be overwritten by the step; it is replaced instead.
    """

    def propose(self):
        self.proposed_bundle = target
        self.target_pathway = pathway

    monkeypatch.setattr(type(behaviour), "update", propose)
    monkeypatch.setattr(type(behaviour), "should_transition", lambda self: True)


@pytest.fixture
def farm(ca_model_builder):
    """One farm out of a private model, so stepping it disturbs nobody."""
    model = ca_model_builder(seed=4242)
    return model, list(model.world.farmers)[0]


def balance(farmer):
    """The year's change in capital as the reported terms account for it."""
    return (
        farmer.raw_profit
        - farmer.transition_cost
        + farmer.residue_sale_income
    )


class TestTheYearsCapitalBalances:
    """ΔK = raw_profit - transition_cost + residue_sale_income."""

    def test_it_closes_over_a_run_of_years(self, farm):
        """Every year, including any in which the farm moved."""
        model, farmer = farm
        opening = model.lpjml.sim_year

        for t in range(opening, opening + 6):
            before = farmer.capital
            farmer.update(t=t)

            if farmer.capital == 0.0:
                # The floor absorbed the remainder, so there is nothing to
                # reconcile; that it clamped at all is the finding.
                continue

            assert farmer.capital - before == pytest.approx(
                balance(farmer), abs=1e-6
            ), f"capital does not reconcile in {t}"

    def test_it_closes_in_a_year_the_farm_moves(self, farm, monkeypatch):
        """The case the plain identity got wrong: a charge outside raw_profit.

        Left to itself the fixture farm holds its bundle for the whole window,
        so the transition leg would never be exercised and the balance would
        look closed only because the term was always zero.
        """
        model, farmer = farm
        behaviour = farmer.behaviour
        held = behaviour.practice_bundle

        # Flipping tillage buys equipment, so the move carries a real cost.
        force_the_move(
            monkeypatch,
            behaviour,
            ManagementBundle.from_practices(
                1 - held.tillage, held.cover_crop, held.residue_on_field
            ),
        )

        before = farmer.capital
        farmer.update(t=model.lpjml.sim_year)

        assert farmer.transition_cost > 0.0, "the move was free, so it proves nothing"
        assert farmer.capital - before == pytest.approx(
            balance(farmer), abs=1e-6
        )

    def test_the_move_is_missing_from_raw_profit_alone(self, farm, monkeypatch):
        """States the trap directly, so nobody folds the one-offs back in."""
        model, farmer = farm
        behaviour = farmer.behaviour
        held = behaviour.practice_bundle

        force_the_move(
            monkeypatch,
            behaviour,
            ManagementBundle.from_practices(
                1 - held.tillage, held.cover_crop, held.residue_on_field
            ),
        )

        before = farmer.capital
        farmer.update(t=model.lpjml.sim_year)

        assert farmer.capital - before != pytest.approx(farmer.raw_profit)

    def test_raw_profit_is_the_sum_of_its_own_parts(self, farm):
        """The operating terms, separately from the one-offs."""
        _, farmer = farm
        farmer.update_capital()

        assert farmer.raw_profit == pytest.approx(
            farmer.delta_revenue - farmer.delta_costs + farmer.transfers
        )

    def test_transfers_are_the_sum_of_the_three_instruments(self, farm):
        _, farmer = farm
        farmer.update_capital()

        assert farmer.transfers == pytest.approx(
            farmer.carbon_payment
            + farmer.practice_subsidy
            - farmer.nitrogen_charge
        )

    def test_the_one_offs_are_zero_in_a_year_without_a_move(self, farm):
        """So a nonzero one is evidence of a transition, not of carry-over."""
        model, farmer = farm
        bundle_before = farmer.behaviour.practice_bundle

        farmer.update(t=model.lpjml.sim_year)

        if farmer.behaviour.practice_bundle == bundle_before:
            assert farmer.transition_cost == 0.0
            assert farmer.residue_sale_income == 0.0

    def test_a_one_off_does_not_carry_into_the_next_year(self, farm):
        """Cleared at the top of update_capital, not left to be double counted."""
        _, farmer = farm
        farmer._transition_cost = 123.0
        farmer._residue_sale_income = 456.0

        farmer.update_capital()

        assert farmer.transition_cost == 0.0
        assert farmer.residue_sale_income == 0.0


class TestTheTwoStrawValuations:
    """``delta_costs`` prices residue as a difference between them."""

    def test_the_dynamic_one_follows_this_years_revenue(self, farm):
        _, farmer = farm
        farmer.update_capital()

        assert farmer.residue_opportunity_cost == pytest.approx(
            farmer.compute_residue_opportunity_cost()
        )

    def test_the_baseline_one_does_not_move(self, farm):
        """Frozen at the start, which is what makes the retention charge a
        practice cost rather than a weather signal."""
        model, farmer = farm
        frozen = farmer.baseline_residue_cost
        opening = model.lpjml.sim_year

        for t in range(opening, opening + 4):
            farmer.update(t=t)
            assert farmer.baseline_residue_cost == frozen

    def test_retention_is_charged_the_frozen_one(self, farm):
        """And selling the shortfall against it, so the pair is recoverable."""
        _, farmer = farm

        assert farmer.get_residue_cost_for_bundle(1) == pytest.approx(
            farmer.baseline_residue_cost
        )
        assert farmer.get_residue_cost_for_bundle(0) == pytest.approx(
            farmer.baseline_residue_cost
            - farmer.compute_residue_opportunity_cost()
        )
