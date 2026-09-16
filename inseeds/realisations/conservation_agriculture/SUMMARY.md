# Conservation Agriculture Realisation: Model Summary

This document describes the **Conservation Agriculture (CA)** realisation of InSEEDS. The model couples farmer decision-making based on Theory of Planned Behaviour (TPB; Ajzen 1991) with LPJmL crop model outputs.

---

## 1. Architecture Overview

The realisation has a hierarchical structure:

```
Model
 └── World
      ├── Countries (CACountry)
      │    ├── FAO economic data (capital, prices)
      │    ├── Country-level statistics for social learning
      │    └── Agroecological cluster membership
      │
      ├── Agroecological Clusters
      │    └── Cross-border social learning statistics
      │
      └── Cells
           └── Farmers (CAFarmer)
                ├── Capital dynamics
                ├── Practice bundle state
                └── TPB decision model (ca_behaviour.TPB)
```

| Component | File | Purpose |
|-----------|------|---------|
| `CACountry` | `ca_country.py` | Country-level FAO data, capital parameters, prices |
| `CAFarmer` | `ca_farmer.py` | Farmer agent with capital, costs, revenue |
| `TPB` | `ca_behaviour.py` | Decision model: bundles, social learning, attitude, norms, PBC |
| `ca_agroecology` | `ca_agroecology.py` | Agroecological clustering for cross-border learning |
| `Country` | `region.py` | Base class with cropland area calculation |
| `FaoDataset` | `data/fao/base.py` | Abstract base for FAO data handlers |

---

## 2. Practice Bundles

Conservation Agriculture is represented as a combinatorial decision over three binary practices:

| Practice | 0 | 1 |
|----------|---|---|
| **Tillage** | No-till (CA) | Conventional tillage |
| **Cover crop** | None | Planted |
| **Residue** | Baseline removal | Retained on field |

This yields 8 possible bundles:

| ID | Tillage | Cover | Residue | Name |
|----|---------|-------|---------|------|
| 0 | No-till | No | No | `notill_only` |
| 1 | No-till | No | Yes | `notill_residue` |
| 2 | No-till | Yes | No | `notill_cover_crop` |
| 3 | No-till | Yes | Yes | `conservation` (full CA) |
| 4 | Conv | No | No | `conventional` |
| 5 | Conv | No | Yes | `residue_only` |
| 6 | Conv | Yes | No | `cover_crop_only` |
| 7 | Conv | Yes | Yes | `cover_crop_residue` |

Full Conservation Agriculture (bundle 3) requires all three practices: no-till + cover crops + residue retention.

---

## 3. Theory of Planned Behaviour (TPB)

The TPB model (`ca_behaviour.TPB`) implements the intention equation:

**TPB = (w_att × Attitude + w_norm × SocialNorm) × PBC**

PBC enters multiplicatively: a farmer with positive attitude and supportive norms but low PBC (cannot afford transition) will not adopt.

### 3.1 Attitude

Attitude combines two sources, weighted by the AFT parameters `weight_own_land` and `weight_social_learning`:

1. **Own-land experience** (`compute_attitude_own_land`): how a reference state scores against where the farm stands now. With remembered experience of the candidate bundle, that memory is the reference, which is a genuine attitude towards the behaviour. Without it, the reference is the farm's own state at the last transition, and the channel expresses dissatisfaction with the status quo instead; in that case it is one-sided, floored at the neutral 0.5, because thriving is no argument against a practice the farmer has never tried.

2. **Social learning** (`compute_attitude_social_learning_*`): performance comparison at three spatial scales.
   - **Local**: pairwise against each neighbour, weighted by bundle and crop similarity and attenuated by the neighbour's years of experience
   - **Country**: against the country aggregate for the candidate bundle
   - **Cluster**: against the aggregate across agroecologically similar countries

Both sources are shrunk towards 0.5 in proportion to the evidence behind them (`confidence_years`), so a single year cannot swing an attitude. The scale weights `weight_attitude_local`, `weight_attitude_country` and `weight_attitude_cluster` are AFT-specific and redistributed when a level is disabled.

#### 3.1.1 Performance Scoring

Every comparison above reduces to one score, computed by `compute_performance_score` over five metrics: yield, soil carbon, root-zone moisture, nitrogen leached and profit.

Leaching is read from LPJmL's whole-cell `leaching` output, which is the total nitrogen reaching the water body regardless of which stand it came from. This matches the boundaries package measurement and keeps the tax base consistent with environmental outcomes: the aquifer does not know which hectare contributed the nitrogen. The flux is reported per m² of cell, in gN/m²/yr.

Leaching is taken as the flux in gN/m²/yr, the mass that left the root zone, not as its concentration in the drainage water. A mass is what adds up over a catchment and what a charge on losses can be written on, and it is the same quantity the peer estimator differences, so a reduction remembered by a neighbour converts into kilograms without assuming their field drained like yours — which is exactly what a cover crop changes. The concentration in mgN/l is still reported as `leaching_concentration`, because that is the form a nitrate standard takes and so what makes the output comparable to one, but nothing in the model decides on it.

**Self-referenced rates.** Each metric arrives as a relative deviation from that farm's own 2015-2025 baseline: `ln((value + n) / (baseline + n))` for the four environmental metrics, and `asinh(delta_profit / baseline_revenue)` for profit, which is the signed counterpart of the log and stays defined when costs exceed baseline revenue. Both equal `value / baseline - 1` to first order, so a small deviation reads as a percentage change. Because the rates are self-referenced, climate, soil, farm size and income level divide out by construction and no country means are involved.

The log form rather than the plain ratio `value / baseline - 1`, which is bounded at -1 below but unbounded above and so treats an improvement as larger than a deterioration of the same multiplicative size; under the log a doubling (+0.69) and a halving (-0.69) are equidistant from zero.

`n` is the level at which that metric stops being worth noticing — 0.1 gN/m² of leached nitrogen, 100 gC/m² of soil carbon, 10 mm of root-zone water, 10 gC/m² of yield — set under `materiality` in the config. Offsetting both sides of the ratio by it is what makes the log usable across the full range of baselines the global run contains. A bare `ln(v/b)` needs a positive baseline and diverges as either side approaches zero, so a cell whose historic mean is a rounding error reports a spectacular change from a physically trivial one. With the offset such a cell reports zero when nothing changed, whatever its baseline, and a collapse to zero is bounded at `-ln(1 + b/n)` instead of an arbitrary sentinel.

For leaching the offset does a second job. A year that drained nothing leached nothing, so the flux really does go to zero, and `n` is what stops `ln(0/b)` from running away — a dry year against a 2 gN/m² baseline reads about -3, a large improvement but a bounded one. That leaves leaching weather-driven in the same way yield is, which the fitted trend averages out over an episode. Where the baseline is well above `n`, which is the normal case for all four metrics, the offset shifts the rate by under 0.02 and the form is indistinguishable from `ln(v/b)`.

Profit keeps a lower bound on `baseline_revenue` rather than an offset, because it divides by revenue without taking its log.

**Population rulers.** A rate becomes a score by division by the spread of that same rate across farmers:

```
score_metric = 1 + rate / level_std + trend_weight × (trend × horizon) / trend_std
```

`level_std` is the robust cross-farmer spread of the rate, bounded below by the median per-farmer interannual std; `trend_std` is the robust MAD of projected trends. Both come from `CAWorld.compute_reference_scales` and are recomputed at the start of every simulated year, so there is no tuning constant anywhere in the score.

The cross-farmer spread is what makes a stock and a flow comparable. A flow such as yield wanders around its baseline, so its deviation stays of the order of one year's fluctuation. Soil carbon integrates its increments, so its deviation grows with time while its year-to-year wobble stays tiny; measured against that wobble it would run away to z of several tens and override the configured weight of every other metric. The cross-farmer spread grows on the same timescale as the signal it scales, so both land at z of order one.

The z-scores are deliberately unbounded, because the magnitude carries information: a sustained 25% gain in soil carbon should not score the same as a 6% one.

For **leaching**, lower is better, so its z-scores are inverted.

**Two weightings per metric.** `trend_weight_*` sets how much a metric's trend matters against its level, which follows from how fast the metric moves; `weight_*` sets how much the farmer cares about it, which follows from the AFT.

| Metric | Trend weight | Rationale |
|--------|-------------|-----------|
| Yield | 0.2 | Level-dominant: current productivity matters most |
| Soil C | 0.7-0.8 | Trend-dominant: decades to rebuild, legacy effects |
| Moisture | 0.3 | Mostly level, with some trend sensitivity |

Pioneers carry a higher `trend_weight_soil` (0.8 against 0.7), reflecting their longer-term orientation (Rogers 2003).

**Scientific basis**: relative stock change methods for soil carbon (Paustian et al. 2016); multi-attribute utility (Andrews et al. 2004, Soil Quality Index); yield gap framing (van Ittersum et al. 2013).

### 3.2 Social Norm

Social norm reflects "what others are doing" (descriptive norm) at three scales:

- **Local** (`compute_social_norm_local`): Similarity-weighted adoption fraction among neighbours
- **Country** (`compute_social_norm_country`): Fraction of farmers using the bundle nationally
- **Cluster** (`compute_social_norm_cluster`): Adoption fraction across climatically similar countries

Each uses a sigmoid transformation with AFT-specific thresholds (e.g., `threshold_social_norm_local`).

The sigmoid threshold model (Granovetter 1978): below threshold → social drag; above threshold → social boost.

Weights (`weight_social_norm_local`, `weight_social_norm_country`, `weight_social_norm_cluster`) are AFT-specific and redistributed when levels are disabled.

### 3.3 Perceived Behavioral Control (PBC)

PBC asks whether the switch can be afforded out of income:

> PBC = pbc_base × 2 × sigmoid(−ratio), capped at 1
> ratio = net_financial_impact / (baseline_revenue × benefit_years)

The net impact is the one-time transition cost, plus the change in annual direct costs, minus the anticipated revenue gain, the last two counted over the discounted benefit-years of `annuity_factor(projection_horizon, discount_rate)`. It is compared against income over the same view, so the ratio is dimensionless.

`2 × sigmoid(−ratio)` is `1 − tanh(ratio)`, which anchors a cost-neutral switch at `pbc_base` and lets it move both ways: a costly switch drives PBC to 0, a profitable one raises it above the base until the cap. `pbc_base` is AFT-specific (pioneers 0.85, traditionalists 0.65) and is the only place risk tolerance enters PBC.

The anticipated yield change comes from `anticipated_relative_yield_change`, which uses only peers that have run **both** bundles, and takes the experience-weighted median of their within-farm differences. Differencing inside each peer removes that farm's site conditions, so what is averaged is a set of effects rather than a set of levels.

### 3.4 Weight Redistribution

When a spreading level is disabled (e.g., `enable_cluster: false` for single-country runs), its weight is redistributed proportionally to enabled levels:

```python
# Example: if cluster disabled with weights (0.7, 0.2, 0.1)
# Redistributed: (0.778, 0.222, 0.0) - total unchanged
```

---

## 4. Decision Flow

Observing and deciding are separate. `observe()` runs for every farmer every year, whether or not a decision follows, so a farmer too poor to change practice is still farming and is still learned from:

1. **File this year's litter cover** — for the residue certification, which reads the window rather than the year (section 8.2)
2. **Add observation** — update the regression accumulators for the five metrics
3. **Update bundle memory** — store current level and trend, which is what neighbours read
4. **Decay old memories** — bounded rationality (`memory_decay_years`)

`update()` then decides:

5. **Check the re-evaluation cadence** — `_observation_years` must have run down
6. **Re-evaluate residue status** — certify from the median litter cover, if `residue_status_by_cover` is set (section 8.2); this changes the label, not the management, so costs and the LPJmL input are untouched. Before the target search, which reads `practice_bundle`
7. **Check affordability deselection** — drop practices whose running costs have outgrown the stock (forced, bypasses TPB)
8. **Check fallback** — revert after `fallback_years` consecutive years below the state left behind at the last switch (forced, bypasses TPB)
9. **Find target bundle** — best-performing local neighbour, else country, else cluster, else random exploration
10. **Adjust for affordability** — trim the target to what the capital can buy
11. **Compute TPB components** — attitude, social norm, PBC
12. **Transition decision** — compare the intention to `transition_threshold`

### 4.1 Thresholds (Global, not AFT-specific)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `transition_threshold` | 0.5 | TPB score for any practice change |

### 4.2 Evaluation Timing

Three different things are kept apart:

| Parameter | Meaning |
|-----------|---------|
| `min_observation_years` | Re-evaluation cadence: mean years between decisions |
| `projection_horizon` | How far ahead a farmer projects when judging an investment |
| `discount_rate` | How heavily those future years are discounted |

The cadence counter is drawn from `Normal(min_obs, min_obs/2)`, which both requires enough of one's own harvests to judge a change and desynchronises farmers so no artificial evaluation wave forms.

The counter is reset once a decision has actually been weighed, whether it was taken or declined. A farmer who looked and found nothing distinct to consider (`no_target`, `target_same`) has not used the commitment period up and looks again next year; see `BLOCKERS_WITHOUT_DECISION`.

### 4.3 Transition Blockers

The model tracks why transitions don't happen:

| Blocker | Meaning |
|---------|---------|
| `observation_years` | Not yet time to re-evaluate (commitment period) |
| `no_target` | No better neighbour + exploration didn't trigger |
| `target_same` | Target bundle equals current (already optimal) |
| `tpb_low_pbc` | TPB failed due to cost/capital constraints |
| `tpb_low_social_norm_local/country/cluster` | TPB failed due to low social norm |
| `tpb_low_attitude_*` | TPB failed due to low attitude |
| `capital_survival` | Capital below survival threshold |
| `affordability_forced` | Practices deselected due to unaffordable direct costs |

### 4.4 Transition Drivers

When transitions succeed, the model tracks the enabling factor:

| Driver | Meaning |
|--------|---------|
| `local_*` | Inspired by local neighbour |
| `country_*` | Inspired by country-level performer |
| `cluster_*` | Inspired by cluster-level performer |
| `exploration_*` | Discovered via random exploration |
| `fallback` | Reverted due to sustained decline |

---

## 5. Capital Dynamics (CAFarmer)

### 5.1 Initial Capital

Capital per hectare is derived from FAO Net Capital Stocks:

> K₀/ha = (NCS × ag_share × crop_share) / cropland_area

Where:
- **NCS**: Net Capital Stocks for "Agriculture, Forestry and Fishing" (million USD)
- **ag_share**: Agriculture fraction of AFF (country-specific static table)
- **crop_share**: Field crops fraction of agriculture (from FAO GPV data)

### 5.2 FAO Endowment + Own Deviations

FAO sets the initial stock; afterwards it moves by this farm's own margin:

> K_{t+1} = K + Δrevenue − Δcosts + payments − charges

Where:
- **Δrevenue**: current_revenue − baseline_revenue
- **Δcosts**: cost difference between current and baseline bundle
- **payments − charges**: policy transfers — carbon payment and practice subsidies in, nitrogen charge out — zero unless a scenario turns one on (section 5.7). Absolute rather than deviations from baseline, unlike the two terms before them, which is what lets an instrument move the capital stock persistently in either direction

Whether the stock may *refuse* a practice change is set separately, by
`farm_economics.capital_constraints`.

### 5.3 Baselines (Set at Initialization)

| Baseline | Source | Represents |
|----------|--------|------------|
| **baseline_revenue** | Average over 2015-2025 (historic LPJmL data) | This farm's typical income |
| **baseline_bundle** | Initial practice bundle at simulation start | This farm's starting costs |
| **baseline_cover_crop_type** | Cover crop sown at simulation start, `None` if there was none | Which seed the starting costs were paid for |

`baseline_cover_crop_type` is frozen for the same reason `baseline_residue_cost`
is. The cover crop *type* is not one of the bundle's three fields, so a farm can
switch from a catch crop to a legume without `baseline_bundle` moving at all. If
the baseline leg of `Δcosts` were priced with whatever is in the ground today,
both legs of the difference would carry the legume's seed cost and it would
cancel — a farm that started with a cover crop would sow the roughly twice
dearer seed for free. Farms that adopt a cover crop during the run were never
affected, since their baseline leg has no cover crop to price.

### 5.4 How Deviations Work

| Condition | Effect |
|-----------|--------|
| Δrevenue = 0, Δcosts = 0 | Farmer follows FAO baseline exactly |
| Better yields (Δrevenue > 0) | Capital grows faster than baseline |
| Worse yields (Δrevenue < 0) | Capital grows slower than baseline |
| CA with higher costs (Δcosts > 0) | Reduces capital vs baseline |
| CA with savings, e.g. no-till (Δcosts < 0) | Increases capital vs baseline |

### 5.5 Why Deviations

The FAO endowment is a sector average and already carries depreciation, typical reinvestment, subsidies and credit at that average. Modelling those components explicitly per farm would add mechanisms the model cannot constrain, so only the deviation is modelled: what this farm earns and spends relative to its own starting point. Individual performance then matters without the stock needing an exogenous growth law.

### 5.6 Practice Costs

These are the costs that differ from conventional farming, used in Δcosts calculation.

**GDP-Scaled Costs**

Each cost is a `[min, max]` range interpolated by where the country's GDP per capita falls between the global 10th and 90th percentiles:

> cost = min + (max − min) × gdp_cost_ratio,  gdp_cost_ratio ∈ [0, 1]

Countries below p10 pay the minimum, above p90 the maximum. This accounts for global variation in equipment and input costs. Variation is **conservative** (1.3-2x), reflecting that:
- Equipment: India custom hire $15-27/ha vs US $66/ha (~2-3x) [1,6]
- Fuel prices: Global variation ~1.5x (excluding oil-producer outliers)
- Seeds: Similar global prices; import costs can increase developing country prices

| Practice | Transition Range | Direct Range | Source |
|----------|-----------------|--------------|--------|
| No-till | $40-66/ha | -$40 to -$55/ha (savings) | [1,2,3,6,7] |
| Cover crop, catch crop | $12-22/ha | $45-75/ha | [4,5,15,16] |
| Cover crop, legume | $12-22/ha | $57-120/ha | [4,14,15,16] |
| Residue retention | $0 | *dynamic* | Singh & Schiere 1995 |

The two cover crops differ only in seed, and they differ in it by about a
factor of two: drilled cereal rye runs $44-89/ha against $82-109/ha for hairy
vetch, and farmdoc's Illinois budget separates seed alone at $15/acre for rye
and $45/acre for a rye/vetch blend [15,16]. The model carries them as
`seeds_nonlegume` and `seeds_legume`, and exactly one applies — the one the
farm actually sowed. Section 5.8 is how it decides which.

**Transition costs** (one-time equipment/setup):
- No-till: US literature $66/ha [1]; India custom hire $15-27/ha [6]; Bangladesh small seeder $7-10/ha [7]
- Cover crop: US literature $20/ha [4]; range $12-22/ha for broadcasting to precision
- Residue: $0/ha (modern combines have spreaders as standard equipment)

**Direct costs** (annual operating):
- No-till: US literature -$50/ha fuel savings [2,3]; range -$40 to -$55/ha (less/more mechanized)
- Cover crop: US literature $75/ha [4]; range $55-85/ha accounting for seed cost variation
- Residue: Computed **dynamically** as crop_revenue × use_cost_fraction

**Dynamic Residue Opportunity Cost**

The residue direct cost is computed dynamically in `compute_residue_opportunity_cost()`:

```
residue_cost = crop_revenue × Σ(use_fraction × cost_fraction)
```

Where:
- `use_fraction`: Spatially-explicit fractions from MADRaT data (Smerald et al. 2023)
- `cost_fraction`: From Singh & Schiere (1995) finding that straw value = 10-15% of crop value:
  - Burnt: 0% (no economic value)
  - Removed (feed/sale): 12.5% (midpoint of 10-15%)
  - Recycled: 0% (returns to field)

This makes residue costs vary spatially (different use patterns) and temporally (yield changes).

**Residue Economics by State**

Residue costs depend on whether the farmer is **retaining** or **selling**:

| State | Cost Formula | Rationale |
|-------|-------------|-----------|
| **Retention** (residue=1) | `baseline_residue_cost` (fixed) | Farmer committed to giving up baseline income |
| **Selling** (residue=0) | `baseline - current` | Income/loss relative to baseline |

The state here is the *committed* one, not the certified one; see section 8.2.

**Retention (residue=1):**
- Cost = baseline (frozen) — farmer committed to giving up this income
- Higher yields: extra residue is "free" to leave on field (no extra cost)
- Lower yields: still pays baseline cost, just leaves less physically

**Selling (residue=0):**
- Cost = baseline - current (can be negative = income)
- Higher yields: sells more than baseline → negative cost (extra income)
- Lower yields: sells less than baseline → positive cost (lost income)

This captures the economic reality:
- When you commit to retention, you budgeted to forgo baseline income
- When you sell, your income fluctuates with yields

**Capital Update vs Affordability:**
- **delta_costs** (capital update): Uses frozen baseline for practice CHANGES only (yield-driven income changes are in delta_revenue)
- **Affordability/PBC**: Uses dynamic `get_bundle_direct_costs()` which includes current residue economics

**Tracing it in the output.** Both straw valuations are reported, since the
residue term of `delta_costs` is a difference between them and cannot otherwise
be recovered: `residue_opportunity_cost` is this year's dynamic figure (current
revenue x the cell's weighted use fraction) and `baseline_residue_cost` is the
same calculation frozen on baseline revenue, which is what retention is charged
and what the cost-share subsidy is paid on.

**The year's capital balance.** `raw_profit` is operating profit,
`delta_revenue - delta_costs + transfers`, and closes against its parts
exactly. It is *not* the whole change in capital: a transition is charged after
`update_capital` has run, so

```
capital_t - capital_{t-1} = raw_profit - transition_cost + residue_sale_income
```

exact except where the floor at zero clamped it. Both one-off terms are
reported, are zero in a year the farm held its bundle, and are cleared at the
top of `update_capital` so no year inherits another's. `residue_sale_income` is
raised only by a farm driven off retention by affordability, selling the straw
it still had. `tests/test_capital_balance.py` pins the identity, including in a
year the farm moves.

**Why the transfers are reported as settled, not recomputed.** `Farmer.policy`
resolves against `lpjml.sim_year`, and the coupler advances that to the next
year as soon as the year's exchange is done - which is before the outputs are
collected. A payment recomputed on read therefore answers for the following
year: zero in the run's final year, because the span has ended, and nonzero in
the year before a late `start_year`. `carbon_payment`, `practice_subsidy` and
`nitrogen_charge` report what `update_capital` settled on instead. The live
computations remain available as `carbon_payment_for`, `practice_subsidy_for`
and `nitrogen_charge_for`, which is what the decision logic uses, evaluated at
decision time where that is the correct year.

**References:**
1. University of Illinois farmdoc (2023). *Machinery Cost Estimates: Field Operations*.
   https://farmdoc.illinois.edu/assets/management/machinery-costs/field_operations_2023.pdf
2. USDA CEAP (2022). *Save Money on Fuel with No-Till Farming*.
   https://www.farmers.gov/blog/save-money-on-fuel-with-no-till-farming
3. SARE (2019). *Cover Crop Economics*.
   https://www.sare.org/wp-content/uploads/Cover-Crop-Economics.pdf
4. Indigo Ag / Iowa State (2020). *Cover Crop Equipment & Custom Rates*.
   https://app.indigoag.com/programs/learn/paper/what-equipment-is-needed-to-plant-cover-crops
5. Singh, K. & Schiere, J.B. (eds.) (1995). *Handbook for Straw Feeding Systems*.
   ICAR, New Delhi. Ch. 1.1, Box 1. https://edepot.wur.nl/333326

### 5.7 Policy Instruments

Every instrument the model has lives in the `policy` block, all zero by default, so the model as configured is the counterfactual and any run with one has something to be compared against.

| Instrument | Priced on | Unit | Varies by income? | Direction |
|------------|-----------|------|-------------------|-----------|
| `carbon_price` | soil carbon accumulated above the farm's own baseline, floored at zero | $/tCO2e | no, one global figure | payment in |
| `nitrogen_charge` | nitrogen leached below the root zone, above the allowance | $/kgN | yes, `[low, high]` on the GDP ratio | charge out |
| `levy_free_concentration` | sets that allowance, as a nitrate standard the drainage converts to a mass | mgN/l | no, a health limit | exempts |
| `practice_subsidies.cover_crop` | holding a cover crop, per hectare per year | $/ha/yr | yes, `[low, high]` on the GDP ratio | payment in |
| `practice_subsidies.cover_crop_legume` | *on top of the above*, where the crop sown is a legume | $/ha/yr | yes, `[low, high]` on the GDP ratio | payment in |
| `practice_subsidies.residue_on_field` | retaining residue, per hectare per year | $/ha/yr | yes, `[low, high]` on the GDP ratio | payment in |

**The practice subsidies are paid per hectare per year of commitment.** That is how EQIP and the CAP eco-schemes work: the payment is for holding the practice, not only for switching into it, and it lands on the committed bundle rather than the certified one. A farm whose stubble happens to clear the 30% threshold draws nothing — it enrolled in nothing (section 8.2).

They scale with income for a different reason than the nitrogen charge: a subsidy comes out of a national budget, so what a country pays tracks what it can afford. Calibration, both roughly a full cost-share: US EQIP pays $84-185/ha at 50-75% cost-share and EU CAP eco-schemes €40-165/ha for soil conservation [6,7]. Annual costs are $40/ha low-capital and $85/ha high-capital for a non-legume cover crop, $52 and $105 for a legume, so no single flat rate is a cost-share of both: one that covers the legume over-pays the catch crop by 20-30 points of its cost, and vice versa. `cover_crop` therefore covers the catch crop at 40-85 $/ha/yr and `cover_crop_legume` covers the difference. US CSP pays $5-25/ha and CAP €20-50/ha for residue management [6,7], so 20-30 $/ha/yr softens retention without covering the straw income given up.

`cover_crop_legume` is an enhancement rather than a separate rate, which is the shape the real schemes take: EQIP 340 and the CAP eco-schemes both post a base cover crop payment and pay more for a multi-species or legume mix. It is the only per-hectare rate here that distinguishes the two crops, and therefore the only one that can steer the type: the base payment is made whichever crop is sown, so it cancels out of the legume-versus-catch-crop comparison entirely and can only change *whether* a cover crop is grown, never *which*. For the enhancement to decide rather than nudge it has to be worth about the seed premium it is overturning, so 15-30 $/ha/yr sits just above that at both ends of the income range.

There is no no-till subsidy, though EQIP pays for one. It is a deliberate gap rather than an oversight.

**Only the local instrument varies by income.** Nitrogen damage depends on the aquifer and on who drinks from it, and the rate a country can levy depends on what its farms can bear, so the charge is a `[low-income, high-income]` pair interpolated on the same GDP ratio as every practice cost. Carbon is a global externality: a tonne is worth the same wherever it is put away, so one price is what correctly values it. A price that varied by income would misprice the tonne and would pull sequestration towards whoever pays most rather than whoever abates cheapest — the spillover a differentiated price creates and a uniform one does not. Passing a range for `carbon_price` therefore raises rather than being averaged.

Rough calibration for the charge, all on a per-kgN basis. The Dutch MINAS levy is the one precedent with the same base — nitrogen *lost*, not nitrogen bought — at €0.68/kgN in 1998 and €2.53-5.07/kgN from 2002, the latter set high enough that eliminating the loss always beat paying [18]. Taxes on fertiliser purchased were milder and did correspondingly little: Sweden SEK 1.80/kgN, Denmark DKK 5/kgN [17]. Keeler et al. put the damage itself at $0.60-13/kgN [19]. So $0.5-1 is a mild tax, $3 is MINAS at its prohibitive setting, and 13 is a damage ceiling no government has levied.

**The charge can fall on the excess over a standard.** MINAS taxed only the surplus above a levy-free allowance, and `levy_free_concentration` does the same here. It is set as a nitrate concentration in mgN/l, not as a mass, for the reason that matters physically: it is the strength of what reaches the aquifer that harms anyone, so the same kilogram is tolerable where there is water to dilute it and not where there is none. Each farm's allowance in mass terms is therefore its own,

> allowance [gN/m²] = levy_free_concentration × runoff [mm] / 1000

a millimetre over a square metre being a litre. Multiplying a critical concentration by the drainage is how the EU critical-nitrogen work derives a critical leaching rate [21], and there are three candidate concentrations that differ by a factor of four:

| mgN/l | What it protects | Source | Allowance at 300 mm |
|-------|------------------|--------|---------------------|
| 2.5 | surface water against eutrophication — the **planetary boundary** | Earth system boundary work [22], range 1-4 | 7.5 kgN/ha |
| 5.6 | groundwater, precautionary | EU Drinking Water Directive guide value of 25 mg NO3/l [23] | 17 kgN/ha |
| 11.3 | groundwater drinkability | WHO/EU limit of 50 mg NO3/l [21] | 34 kgN/ha |

The choice is about the receptor, and the two ends are not equally binding: the EU-27 assessment needs a 43% cut in N inputs to meet the surface-water threshold but finds the groundwater threshold close to actual inputs and exceeded on only 18% of agricultural land [21]. An instrument written against 11.3 therefore collects little and changes little, which is the case for 2.5 if it is meant to bite. The config ships **5.0**, a lenient reading of the boundary — cite it as a choice and not as the planetary boundary, and treat 2.5 / 5.0 / 11.3 as a sensitivity axis, since the allowance scales almost linearly with it.

Note that the code never computes a concentration to do this. It converts the standard *into* a flux and compares in gN/m² throughout, which is algebraically the same test but avoids dividing by drainage — the division is what produces 0/0 at zero runoff and the 47,000 mgN/l that `nitrate_concentration` has to cap. As drainage falls the allowance falls smoothly to zero, and a cell that barely drains barely leaches, so the bill goes with it. The denominator assumption is that cropland drains at about the rate its cell does, LPJmL not reporting runoff by stand.

The Nitrates Directive's own 170 kgN/ha is a cap on manure *applied* and says nothing about losses, and MINAS's 100-140 kgN/ha allowance was a farm-gate *surplus* of which only a fraction leaches, so neither number transfers.

Note where this leaves the concentration. It was demoted out of the scored metrics in section 4 because a concentration cannot be summed over a catchment or differenced between farms whose fields drain differently. Here it is read exactly once, as a threshold, and never carried through a baseline, a ruler or a peer's memory — which is the use it was always suited to, since a standard is written as a concentration in the first place. `taxable_leaching` reports what is left above the line.

The allowance makes the charge kinked, so all three places that price it — the bill, the anticipated saving from a switch, and the legume-versus-catch-crop comparison — go through one `nitrogen_charge_for`. A farm already draining cleaner than the standard expects nothing from leaching less, correctly, and one just above it saves only down as far as the line.

**The carbon payment is floored and the nitrogen charge is not.** A scheme pays for carbon put in and does not invoice a farm for carbon coming back out, which is how soil carbon schemes are written and what stops enrolment being a wager on the weather. It is not free: a farm can build the stock, take the payments, shed the practice and keep both. The nitrogen charge is a tax on a loss, so it falls on the nitrogen that actually left, not on the change against baseline — a farm leaching exactly what it always did still owes it.

**The payment is on the annual increment; the anticipation is on the total.** These are the same quantity seen from two ends, and it is worth being explicit because they look like different instruments. What a farm is actually paid each year is `creditable_soilc_accumulation`, the fitted slope of its soil carbon in gC/m²/yr — the increment, taken from the slope rather than from one year's difference so a payment does not swing on rainfall. What a farm *expects* when weighing a switch comes from peers, and peers remember carbon *levels*, so the estimate is the total the switch eventually banks. If peers on CA hold 5% more carbon and this farm's baseline is 2000 gC/m², that is +100 gC/m²; built over ten years it pays out as 10 gC/m²/yr, and ten years of that comes to the same 100. So `anticipated_policy_transfer` must not multiply the carbon term by the planning horizon — that would pay for each tonne once a year — while it must multiply the nitrogen term, which is a genuine annual flow that repeats.

**What the ledger cannot represent.** Every other term in the annual cash flow is a deviation from baseline, and those average near zero over a run. An absolute charge does not, so a high enough `nitrogen_charge` walks every farm down to the capital floor whether or not it changes practice. That is harsher than reality, where a tax is paid out of gross income rather than out of the machinery shed, so read a run with a large charge as an upper bound on the decapitalisation rather than as a forecast of it. A levy-free allowance is the main thing standing between a run and that outcome, since it exempts the farms that were never the problem.

**Nothing is charged or paid over the historic period**, whatever the scenario sets. The 2015-2025 record is what farms and their peers learned from, and it was made without an instrument; back-dating one would put payments into a memory that never received them.

**Each instrument runs over a span, and can be withdrawn as well as introduced.** `policy.start_year` and `policy.end_year` bound them all, both inclusive, and default to the coupled run itself — `start_coupling` to `lastyear`, read off the LPJmL config by `default_policy_span`, so that neither has to be written down twice and go out of step when the run period moves. Taking the coupling year rather than the first simulated year matters: the run begins long before the farms join it, and the difference between the two is exactly the historic period an instrument must not reach back into. `policy.windows` overrides the span for any one instrument by name — a carbon price arriving late, a subsidy pulled early. `PolicySettings.in_force(year)` resolves the whole set at once by zeroing the rates outside their window, rather than each point of use testing the year for itself, which is what stops the bill and the expectation of it disagreeing about whether a scheme is running. A window that ends before it starts raises rather than silently never acting.

Withdrawal is the more interesting direction to use. The carbon payment and the practice subsidies are absolute transfers, so left running they accumulate in the capital stock indefinitely; ending one asks whether the practice survives without it, or whether the adoption was only ever rented. Note that a withdrawn subsidy also stops being anticipated, so farms stop switching for money that has stopped.

### 5.8 Choosing the Cover Crop

The bundle records only *whether* a cover crop is grown. *Which one* lives on the farmer, because LPJmL needs the type to pick the PFT, and it is a separate decision with its own economics:

> value(legume vs catch crop) = Δrevenue + legume enhancement − Δseed − Δcharge

all against the other type and over the same discounted horizon the rest of the decision uses. A legume fixes nitrogen, which shows up as yield, and costs about twice as much to sow. A catch crop only holds on to nitrogen already there, which shows up as less leaching — worth nothing at all until `nitrogen_charge` prices it, which is the instrument's point.

Both terms come from peers running each crop under this farm's own bundle, which is what keying the performance memory by `(bundle, cover_crop_type)` was for: a legume's record says nothing about a catch crop, and a farm that grew one for ten years and the other for one has not eleven years of experience with either.

**The crop changes on the same clock as every other practice, and never between decisions.** `indicate_cover_crop_type()` is a preference, not a state — a question about prices and peer records with no draw in it and nothing cached, which is what lets candidate costing ask it as often as it likes. What the farm actually sows changes in exactly two places, both of them decisions:

| | Which crop | When |
|---|---|---|
| `commit_bundle()` | chosen, with an exploration draw | a bundle transition |
| `reconsider_cover_crop_type()` | chosen, deterministically | a decision whose bundle came through unchanged |

The second exists because the crop is not a bundle field, so the peer scan cannot re-open it: a farm whose neighbours run the same bundle with the other crop reads that as `TARGET_SAME` and stops. Without it the crop would be settled once at the last bundle change, and a farm that had reached a bundle it was happy with would go on sowing against a `nitrogen_charge` introduced afterwards. It is guarded to fire only where the bundle *did not* change, because where it did, `commit_bundle` has just chosen the crop and re-asking would immediately reverse its exploration draw.

A change of crop is a change of practice either way, so both paths close the episode through the ordinary `record_transition`, and the commitment period starts over. This is why `commit_bundle` exists as one method rather than two calls: the outgoing record is keyed on the crop in the ground, so it has to be filed *before* the new crop replaces it, or the legume's years land in the catch crop's slot — the very slots the choice reads back. Recording first also means the new crop is priced against the bundle the farm is moving to rather than the one it is leaving.

**The comparison is knife-edge, which is what the rest of this section is about.** Measured against forced-management runs, the legume's yield advantage over a catch crop is 1-2% at cell level, and the seed premium (`seeds_legume` less `seeds_nonlegume`) is $12-20/ha. On typical revenue those two very nearly cancel, so nothing about the choice is decisive and everything about how it is asked matters.

**Exploration is what keeps it honest**, but on its own rate. With no evidence the comparison is settled by seed cost alone and the catch crop wins, which would lock the model in: no farm sows a legume, so no record of one accumulates, so no farm ever learns whether it pays, and the estimator ends up reading a record its own answer wrote. So the same impulse that has farms try unfamiliar bundles has them sow the other crop, drawn once per commitment — at `cover_crop_type_exploration_prob`, not `exploration_base_prob`. Bundle exploration picks one of eight, so a 15% draw perturbs a trajectory slightly; here the draw is over two, so the same rate would leave a standing 15-30% of farms growing the crop they judged worse. Against an advantage worth about a per cent of revenue that is not a probe but a large permanent randomisation, so the crop's rate is set well below the bundle's.

The draw sits on the commitment path only, and deliberately not on the re-decision path. Evidence about the crop nobody picks is generated by farms adopting cover crops; responding to that evidence is what the re-decision is for. Putting a draw on both would have a settled farm flipping crop every cadence, and since each flip closes an episode, cover-crop farms would never accumulate enough observations to be confident in either crop.

**And the crop is held for its own term.** `reconsider_cover_crop_type()` fires only once the crop has been in the ground for `cover_crop_type_commitment_years`, which is longer than the bundle cadence and matches `confidence_years` — the span over which the farmer comes to trust an observation at all. A crop it would not keep that long is one it can never form a view on. Without the term the type turns over every three to five years on a comparison whose two sides differ by about a per cent of revenue, so it tracks noise in the peer records rather than the difference between the crops, and the records themselves end up split across both keys. The term survives a bundle change that keeps the same crop, so a farm cannot serve it out by moving between bundles.

### 5.9 Survival Threshold

When capital falls below minimum threshold (configurable, default $100/ha), costly practices are dropped in order: cover crop first (highest direct cost), then residue, then no-till.

### 5.10 Capital Dynamics References

- Solow, R.M. (1956). "A Contribution to the Theory of Economic Growth." *Quarterly Journal of Economics*, 70(1), 65-94.
- OECD (2009). *Measuring Capital - OECD Manual*, 2nd ed. (Methodology for computing rates)
- FAO (2023). FAOSTAT Capital Stock database. (Source data: GFCF, CFC, NCS)

---

## 6. FAO Data Integration (CACountry)

### 6.1 Data Sources

| Dataset | FAO Domain | Variables |
|---------|------------|-----------|
| Capital Stock | CS | NCS, GFCF, CFC |
| Producer Prices | PP | Prices by crop |
| Gross Production Value | QV | GPV crops, GPV agriculture |

### 6.2 Tiered Fallback

When country data is missing (`get_value_with_fallback`):

1. **Country**: Expanding time window (up to 20 years back)
2. **Neighbours**: Mean from neighbouring countries
3. **Global**: Mean across all available countries

### 6.3 Crop Capital Share

FAO Capital Stock covers "Agriculture, Forestry and Fishing". For field crops:

> crop_capital_share = ag_share_of_aff × crop_share_of_ag

Where:
- **ag_share_of_aff**: Country-specific static table (e.g., NLD: 0.95, USA: 0.60)
- **crop_share_of_ag**: GPV_crops / GPV_agriculture from FAO QV domain

### 6.4 Caching

FAO data is cached at class level. Pre-download via `CACountry.preload_fao_data()` before multi-country simulations.

---

## 7. Agroecological Clustering

Countries are grouped by climate similarity to enable cross-border social learning.

### 7.1 Clustering Method

K-means clustering on 6 climate features per country:

| Feature | Description |
|---------|-------------|
| temp_mean | Annual mean temperature (°C) |
| temp_amplitude | Seasonal temperature range |
| prec_mean | Annual mean precipitation (mm) |
| prec_amplitude | Seasonal precipitation range |
| pet_mean | Annual mean potential evapotranspiration |
| pet_amplitude | Seasonal PET range |

Number of clusters determined via elbow method (default) or manually specified.

### 7.2 Usage

- `init_agroecological_clusters()`: Runs at model initialization
- `cluster_management_performance()`: Aggregates country stats into cluster stats after each year
- Farmers use cluster statistics for cross-border social learning (attitude + social norm)

---

## 8. Residue Economics

### 8.1 Opportunity Cost

Residue retention has an opportunity cost based on spatial MADRaT data (Smerald et al. 2023).

**Scientific basis:**
Singh & Schiere (1995, Ch. 1.1, Box 1): "the value of the straw yield can represent between **10-15% or higher** of the total crop value"

**Dynamic implementation:**
Instead of fixed costs, we compute: `residue_cost = crop_revenue × use_fraction`
- This makes costs spatially variable (higher yields → higher opportunity cost)
- This makes costs temporally variable (responds to climate/yield changes)

| Use Type | Fraction of Crop Value | Justification |
|----------|------------------------|---------------|
| burnt | 0% | No market value (common in South Asia) |
| removed | 12.5% | Midpoint of 10-15% (Singh & Schiere 1995) |
| recycled | 0% | Returns to field via manure |
| other | 6.25% | Conservative (half of removed) |

**Cross-validation:**
- 12.5% of $600/ha crop value = $75/ha
- Germany market: ~$66-71/ha (Karras et al. 2024)
- India market: ~$30-35/ha (Duncan et al. 2020)
- Our dynamic approach gives reasonable values

**References:**
- Singh, K. & Schiere, J.B. (eds.) (1995). *Handbook for Straw Feeding Systems*. ICAR, New Delhi / Wageningen Agricultural University. https://edepot.wur.nl/333326
- Smerald, A. et al. (2023). A global dataset for the production and usage of cereal residues. *Scientific Data* 10: 639.
- Karras, T., Noack, V. & Thrän, D. (2024). The Costs of Straw in Germany. *Waste and Biomass Valorization* 15: 5369-5385.
- Duncan, A.J., Samaddar, A. & Blümmel, M. (2020). Rice and wheat straw fodder trading in India. *Field Crops Research* 246: 107680.

Fractions are weighted by actual crop composition from LPJmL.

### 8.2 Residue Status: Certification vs Commitment

Under `residue_status_by_cover`, the residue component of the bundle follows the
FAO definition and is measured, not declared. A farm is certified as retaining
residue when the **median** of its litter cover over the last
`min_observation_years` reaches 30%, and loses the certification when it falls
below, symmetrically in both directions.

The statistic and the timing both matter. Cover moves with the harvest, so a
label read off a single season would follow the weather rather than the
practice — and since every flip redates the label and re-keys the slot the
performance memory records under, rainfall alone could stop a farm ever
accumulating experience of anything. The median asks whether the field was
covered in *most* years of the period, which is what a persistent practice
means; a mean lets two exceptional harvests carry a bare field across the line.
It is read on the re-evaluation cadence rather than annually, so the label
changes when the farm is being looked at, on the same clock as the bundle and
the cover crop.

Reaching the threshold is therefore not the same as choosing to. A farm can be
certified because its crop simply left enough stubble, and the model keeps the
two apart:

| | Certified (`practice_bundle`) | Committed (`committed_bundle`) |
|---|---|---|
| What it means | The field meets the 30% cover threshold | The farmer decided to leave the straw |
| Set by | `reevaluate_residue_status()`, from litter cover | `apply_bundle()`, on a transition |
| Used for | Reporting, bundle memory, social learning | LPJmL residue input, all costs and subsidies |

A farm that crosses the threshold on its own keeps selling straw at the baseline
rate, so it pays no opportunity cost and draws no retention payment; a farm
that has to leave its residue to get there pays both. Because the certification
can change without a decision, it is reported with its own transition driver
(`residue_reclassified`), and the experience the farm claims with the new label
is counted from the year the field was actually covered rather than from the last
transition.

---

## 9. Agent Functional Types (AFT)

Farmers are differentiated into pioneers (25%) and traditionalists (75%).

### Key AFT Parameter Differences

| Parameter | Pioneer | Traditionalist |
|-----------|---------|----------------|
| `pbc_base` | 0.85 | 0.65 |
| `weight_attitude` | 0.8 | 0.6 |
| `weight_norm` | 0.2 | 0.4 |
| `weight_social_learning` | 0.6 | 0.4 |
| `weight_own_land` | 0.4 | 0.6 |
| `min_observation_years` | 5 | 8 |
| `exploration_base_prob` | 0.12 | 0.02 |
| `fallback_years` | 8 | 10 |
| `threshold_social_norm_local` | 0.10 | 0.20 |
| `weight_attitude_local` | 0.5 | 0.7 |
| `weight_attitude_cluster` | 0.2 | 0.1 |

**Pioneers**: Higher PBC, weight attitude more, explore more, re-evaluate more frequently, more open to cross-border learning.
**Traditionalists**: Lower PBC, weight norms more, explore less, more cautious, rely more on local community.

---

## 10. Adaptive Management

### 10.1 Fallback Mechanism

After minimum observation period:
1. Track consecutive years where performance < baseline at transition
2. If decline continues for `fallback_years` → revert to previous bundle
3. Mark failed bundle (increment failure count)
4. Bundles with `failure_count ≥ max_failures` (default 2) excluded from exploration

### 10.2 Bundle Memory

Per-bundle memory with:
- Observed trends (soil C, moisture, yield)
- Duration on bundle
- Year of last update
- Failure count

Memories decay after `memory_decay_years`.

---

## 11. Spreading Level Toggles

Social learning can be enabled/disabled at each spatial scale:

| Config | Default | Purpose |
|--------|---------|---------|
| `enable_local` | true | Neighbour-based spreading |
| `enable_country` | true | Country-level spreading |
| `enable_cluster` | true | Agroecological cluster spreading |

When a level is disabled, its weight is redistributed to enabled levels. Set `enable_cluster: false` for single-country runs.

---

## 12. File Structure

| File | Purpose |
|------|---------|
| `ca_farmer.py` | CAFarmer: capital, costs, revenue, practice application |
| `ca_behaviour.py` | TPB: bundles, memory, social learning, norms, PBC, fallback |
| `ca_country.py` | CACountry: FAO data loading, economic parameters |
| `ca_agroecology.py` | Agroecological clustering for cross-border learning |
| `region.py` | Country base class: cropland area |
| `model.py` | Model: initialization, update loop |
| `config.yaml` | All configuration parameters |
| `data/fao/*.py` | FAO data handlers |
| `data/residue.py` | MADRaT residue fraction data |

---

## References

- Ajzen, I. (1991). The theory of planned behavior. *Organizational Behavior and Human Decision Processes*, 50(2), 179–211.
- Bandura, A. (1977). *Social Learning Theory*. Prentice Hall.
- Granovetter, M. (1978). Threshold models of collective behavior. *American Journal of Sociology*, 83(6), 1420–1443.
- Holling, C.S. (1978). *Adaptive Environmental Assessment and Management*. Wiley.
- Solow, R.M. (1956). "A Contribution to the Theory of Economic Growth." Quarterly Journal of Economics, 70(1), 65-94.
- Kassam, A., Friedrich, T., Shaxson, F., & Pretty, J. (2009). The spread of Conservation Agriculture. *International Journal of Environmental Studies*, 66(6), 677–697.
- OECD (2009). *Measuring Capital — OECD Manual*, 2nd ed. OECD Publishing.
- Rogers, E.M. (2003). *Diffusion of Innovations*, 5th ed. Free Press.
- Smerald, A. et al. (2023). Global crop residue management data. MADRaT.