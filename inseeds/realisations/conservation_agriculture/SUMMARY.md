# Conservation Agriculture Realisation: Model Summary

This document describes the theoretical basis of the **Conservation Agriculture (CA)** realisation and contrasts it with the **Regenerative Tillage** realisation. Both use Theory of Planned Behaviour (TPB; Ajzen 1991) as their decision framework. The CA realisation extends TPB with multi-practice decisions, trend-based learning, adaptive management, capital dynamics, and similarity-weighted social learning.

---

## 1. Overview

| Dimension | Regenerative Tillage | Conservation Agriculture |
|-----------|----------------------|---------------------------|
| **Practices** | Tillage only (0/1) | Three: tillage, cover crop, residue retention |
| **Decision unit** | Binary flip | 8 practice bundles (all combinations) |
| **Attitude formation** | Year-to-year comparison (soil C, yield) | Trend-based (annual rates of change over observation period) |
| **Social learning** | Compare to average of neighbours with *other* strategy | Similarity-weighted: bundle similarity, crop similarity, confidence |
| **Social norm** | Proportion of neighbours using no-till | Similarity-weighted prevalence + conformity pressure |
| **PBC** | Heuristic decay after transition | Economics-based: cost-capital ratio × risk aversion |
| **Economics** | None | FAO-based capital dynamics (depreciation, investment, profit) |
| **Memory** | Previous year only | Per-bundle memory with temporal decay |
| **Fallback** | None | Revert after sustained decline (adaptive management) |
| **Agent heterogeneity** | Pioneer vs traditionalist (AFT)e | Pioneer vs traditionalist (AFT) |
| **Spatial structure** | Cells only | Countries → Cells → Farmers (hierarchical) |

---

## 2. Shared Foundation: Theory of Planned Behaviour

Both realisations implement the TPB intention equation (Ajzen 1991):

**I = (w_att × A + w_norm × SN) × PBC**

where:
- **A** (Attitude): the farmer's evaluation of adopting a practice, formed from own experience and social learning
- **SN** (Subjective Norm): perceived social pressure based on what neighbours do
- **PBC** (Perceived Behavioural Control): perceived ability to perform the behaviour

PBC enters multiplicatively. A farmer with positive attitude and supportive norms but low PBC (e.g. unaffordable transition) will not adopt. This captures the empirical finding that intention requires both motivation and perceived capability (Ajzen 1991).

---

## 3. Regenerative Tillage: Baseline Model

### 3.1 Decision structure

The farmer chooses between conventional tillage (0) and no-till (1). Decisions are re-evaluated at fixed intervals (`strategy_transition_duration`), staggered randomly across agents to avoid synchronisation artefacts.

### 3.2 Attitude

Attitude has two sources, weighted by `weight_own_land` and `weight_social_learning`:

1. **Own-land experience**: Compare current soil carbon and crop yield to the values stored at the time of the last transition. A sigmoid maps the relative change to (0, 1). This captures experiential learning but is sensitive to single-year noise.

2. **Social learning**: Compare own soil C and yield to the average of neighbours who use the *other* strategy. If neighbours using no-till outperform, attitude toward no-till increases. This follows a simple observational learning logic but does not weight neighbours by relevance.

### 3.3 Social norm

Proportion of neighbours using no-till. A sigmoid centred at 0.5 maps this to (0, 1): when more than half of neighbours use no-till, the norm favours adoption.

### 3.4 PBC

PBC is a heuristic variable:
- Starts at 1.0 (full perceived control)
- Drops by a fixed amount (−0.25) after each transition
- Minimum value of 0.5

There is no explicit economic or cost model. PBC loosely captures "adjustment difficulty" but is not grounded in economic constraints.

### 3.5 Limitations motivating the CA extension

- **Single practice**: CA is defined by three pillars (minimum soil disturbance, permanent soil cover, crop diversification; Kassam et al. 2009), not by tillage alone.
- **No memory beyond one year**: Farmers cannot detect trends or distinguish noise from signal.
- **No economic constraints**: Adoption is not limited by capital, cost, or risk.
- **No adaptive fallback**: Once transitioned, there is no mechanism to revert if outcomes deteriorate.
- **Undifferentiated social learning**: All neighbours contribute equally, regardless of crop similarity or experience duration.

---

## 4. Conservation Agriculture: Theoretical Extensions

### 4.1 Multi-practice bundle decisions

Conservation Agriculture requires the joint adoption of minimum soil disturbance, permanent soil cover, and crop rotation/diversification (Kassam et al. 2009). The CA model represents this as a combinatorial decision over three binary practices (tillage, cover crop, residue retention), yielding 8 possible bundles:

| ID | Tillage | Cover crop | Residue | Name |
|----|---------|------------|---------|------|
| 0 | Conv | No | Baseline | `conventional` |
| 1 | Conv | No | Retained | `residue_only` |
| 2 | Conv | Yes | Baseline | `cover_crop_only` |
| 3 | Conv | Yes | Retained | `cover_crop_residue` |
| 4 | No-till | No | Baseline | `notill_only` |
| 5 | No-till | No | Retained | `notill_residue` |
| 6 | No-till | Yes | Baseline | `notill_cover_crop` |
| 7 | No-till | Yes | Retained | `conservation` |

This captures that practices interact: no-till without residue cover exposes soil (see §4.11), and full CA requires all three simultaneously. Farmers do not merely toggle a single transition; they navigate a space of complementary and competing practices.

### 4.2 Trend-based learning

The Regenerative Tillage model compares current values to the previous year, making it susceptible to inter-annual climate variability. The CA model instead evaluates **annual rates of change** in soil carbon, root-zone moisture, and crop yield since the last practice transition:

> trend_x = (x_current − x_at_transition) / years_since_transition

This has several consequences:
- **Noise reduction**: Trends average over multiple years, filtering out single-year fluctuations from weather or market shocks.
- **Minimum observation period**: Farmers must observe a new practice for at least `min_observation_years` (typically 3) before reconsidering. This reflects the empirical finding that CA benefits often take several years to materialise (Pittelkow et al. 2015).
- **Comparability**: Trends are normalised to annual rates, allowing comparison across bundles with different durations.

Performance is summarised as a weighted score across the three indicators (soil C, moisture, yield), with weights reflecting farmer priorities. This score is normalised to the local neighbourhood range (min–max scaling) to enable cross-farm comparison.

### 4.3 Bundle memory and bounded rationality

Farmers maintain a **per-bundle memory** recording the observed trends, duration, year of last update, and failure count for each of the 8 bundles they have tried. This allows them to draw on past experience when evaluating alternatives.

Memories **decay after a configurable period** (default: 30 years). This implements bounded rationality (Simon 1955): agents do not have perfect recall. Old experiences lose relevance as environmental conditions (climate, markets, technology) change. Once a memory expires, the farmer treats the bundle as unexplored, allowing re-evaluation under current conditions.

When evaluating a candidate bundle, the farmer's **own-experience attitude** is the remembered trend weighted by **confidence** (duration / confidence_years). Short experience → low confidence → attitude falls back toward neutral (0.5). This avoids strong beliefs from brief trials.

### 4.4 Target bundle selection

Before computing TPB, the farmer must identify a **target bundle** to evaluate. This is a two-stage process: neighbour imitation, then (if no better neighbour exists) random exploration.

**Stage 1: Imitate best-performing neighbour**

The farmer scans all neighbours and identifies those with higher performance scores (weighted combination of soil C, moisture, and yield trends). Among better-performing neighbours, the one with the **largest performance gap** is selected. The target bundle is then set to that neighbour's current practice bundle.

This implements observational learning (Bandura 1977): farmers adopt practices they observe working well for others. By selecting the *best* neighbour rather than a random better one, the model captures aspiration toward high performers.

If no neighbour outperforms the focal farmer, Stage 2 is triggered.

**Stage 2: Random exploration (innovation diffusion)**

When no neighbour is better, the farmer may still explore a new bundle with some probability. This captures innovation without social influence — the "pioneer" behaviour that seeds diffusion processes (Rogers 2003).

Exploration probability depends on:

1. **Farmer type**: Pioneers have higher base probability (5%) than traditionalists (1%).
2. **Current performance**: Poor performers (normalised score < 0.3) double their exploration probability — they are "searching for better options".
3. **Experience**: Exploration probability ramps from 0.5× (early, cautious) to 1.5× (experienced, confident) over `confidence_years`. This reflects that farmers with longer experience on their current bundle are more willing to experiment.

The exploration probability is capped at 15% to prevent excessive randomness.

**Bundle filtering**

Not all bundles are valid exploration targets:
- The current bundle is excluded (no point exploring what you already do)
- Bundles that have **failed too many times** (failure_count ≥ max_failures, default 2) are excluded — the farmer has learned to avoid them
- **Agronomically unreasonable** bundles (e.g. no-till without residue when residue is cheap) are excluded (see §4.11)

From the remaining valid bundles, one is selected uniformly at random.

**No target → no transition**

If neither neighbour imitation nor exploration yields a target bundle, no transition is proposed and TPB is set to 0. The farmer continues with the current bundle.

### 4.6 Similarity-weighted social learning

Social learning theory (Bandura 1977) predicts that individuals learn preferentially from models they perceive as similar and successful. The CA model implements this through three weighting dimensions:

1. **Bundle similarity**: fraction of practices that match between the focal farmer and the neighbour. A neighbour using (1,1,1) is more informative for evaluating (1,1,0) than a neighbour using (0,0,0). This reflects that practice-specific experience is more transferable between similar management systems.

2. **Crop similarity**: whether the neighbour grows the same dominant crop at a similar area share. Neighbours facing similar agronomic conditions provide more relevant information. This is a fast heuristic (argmax comparison) rather than a full portfolio distance.

3. **Confidence weighting**: the neighbour's duration on their current bundle modulates the reliability of their signal. A neighbour who has used a bundle for 10 years provides a more stable signal than one who transitioned last year.

The combined weight (similarity × confidence) determines each neighbour's contribution to the focal farmer's attitude. The total similarity is a configurable weighted average of bundle and crop similarity (default: 60% bundle, 40% crop).

In contrast, the Regenerative Tillage model groups neighbours by strategy (conventional vs no-till) and compares to the average of the *other* group, without weighting by relevance or experience.

### 4.7 Social norm with conformity pressure

The **descriptive social norm** (Cialdini et al. 1990) — "what others do" — is computed as the average total similarity (bundle + crop) to neighbours. A bundle that many similar neighbours use has a higher norm score.

On top of this, the model adds **conformity pressure** based on neighbourhood homogeneity. When most neighbours use the same bundle (low diversity of practices), conformity pressure is high:
- Adopting the **majority bundle** receives a bonus (up to +0.2)
- Adopting a **minority bundle** receives a penalty (up to −0.1)

This asymmetry reflects empirical findings that deviating from established local practices carries higher social cost than conforming (Cialdini & Goldstein 2004). In the farming context, this captures phenomena such as peer scepticism toward innovators, shared equipment and knowledge networks favouring the majority practice, and reduced social support for non-conformists.

Neighbourhood homogeneity is measured as 1 − (unique_bundles − 1) / N_neighbours. This means conformity pressure is strongest when all neighbours use the same bundle and weakest in diverse neighbourhoods.

The Regenerative Tillage model's social norm is simply the proportion of neighbours using no-till, without conformity pressure or similarity weighting.

### 4.8 Adaptive management and fallback

Adaptive management (Holling 1978; Walters 1986) treats management interventions as experiments: if outcomes deteriorate, the intervention should be revised. The CA model implements this through a **fallback mechanism**:

1. After a grace period (`min_observation_years`), the model tracks consecutive years where performance falls below the **baseline score recorded at transition time**.
2. If performance declines for `fallback_years` consecutive years (default: 5), the farmer **reverts to the previous bundle**.
3. The failed bundle's **failure count** is incremented. Bundles that have failed more than `max_failures` times (default: 2) are excluded from future exploration.

Comparing against the baseline at transition time (rather than the previous year) avoids false positives from gradual trends and focuses on whether the transition itself led to improvement. The grace period allows time for transition effects (e.g. soil biology adjustment after no-till adoption) before evaluation begins.

Fallback bypasses TPB: it is an emergency response, not a planned behaviour change. TPB intention is set to 1.0 directly, ensuring the reversion occurs.

The Regenerative Tillage model has no fallback mechanism. Once transitioned, a farmer can only transition again after the next evaluation interval.

### 4.9 Capital dynamics and affordability

The Regenerative Tillage model has no economic dimension. The CA model introduces **capital dynamics** grounded in FAO data and standard capital accounting (OECD 2009; Jorgenson 1963):

**Initial capital**:

> K₀ = (NCS × crop_share) / agricultural_land_area

where:
- NCS = Net Capital Stocks from FAO (country-specific, in million USD)
- crop_share = fraction of agricultural capital attributable to field crops

**Capital scaling**: FAO's Capital Stock (CS) domain reports total capital for "Agriculture, Forestry and Fishing" combined, including livestock, greenhouses, and fishing infrastructure. Since LPJmL simulates only field crops, we scale capital by a country-specific **crop_capital_share** derived from FAO Gross Production Value (QV domain):

> crop_share ≈ GPV_crops / GPV_agriculture

This scaling is essential for realistic depreciation/revenue ratios. For example:
- Netherlands: crop_share = 0.18 (dairy and horticulture dominate)
- India: crop_share = 0.70 (crop-dominated agriculture)
- Default: crop_share = 0.85 (for countries without specific data)

See `inseeds/components/data/fao/crop_capital_share.py` for country-specific values and methodology.

**Annual capital update**:

> K_{t+1} = K_t − δ K_t + s × max(π_t, 0)

where:
- δ = CFC / NCS: **depreciation rate** from FAO Consumption of Fixed Capital. This captures the annual wear of machinery, equipment, and infrastructure (Jorgenson 1963).
- π_t = revenue − variable_costs − depreciation: **net profit**. Revenue is computed from LPJmL harvest (gC) converted to monetary value via FAO producer prices.
- s = **savings rate** (behavioral parameter, default 15%): fraction of net profit reinvested into farm capital. Literature suggests 10-30% depending on region and farm type (Lowder et al. 2016; FAO 2017).

Only positive profit contributes to reinvestment; losses lead to capital decline through depreciation without offsetting investment.

**Revenue calculation**:

Revenue is calculated by:
1. Getting per-PFT harvest from LPJmL (gC/m²)
2. Multiplying by crop fraction and cell area to get total production (gC)
3. Converting gC to tonnes dry matter (using 0.45 C fraction)
4. Aggregating rainfed/irrigated variants to match FAO price categories (e.g., "rainfed temperate cereals" + "irrigated temperate cereals" → "temperate cereals")
5. Multiplying by FAO producer prices (USD/tonne dry matter)

**Affordability constraints**:
- **Transition costs** (one-time costs for equipment, training) are deducted from capital at transition time.
- **Direct costs** (annual costs for seeds, labour, foregone income) reduce profit.
- If the full target bundle is unaffordable, the farmer adopts a **partial bundle** (cheapest changes first).
- When capital falls below a **minimum threshold** (n_survival_years × δ × K₀, cf. Bandiera et al. 2017 on poverty traps), costly practices are dropped in order: cover crop first (highest direct cost), then residue retention, then no-till.

**Residue economics**: Residue retention has an **opportunity cost** (value of residue as feed or for sale). Retention level is constrained by the ratio of current capital to opportunity cost.

### 4.10 PBC as cost–capital relationship with risk aversion

PBC in the CA model is derived from the farmer's economic situation rather than a heuristic decay:

> PBC = pbc_base × cost_factor × (1 − risk_factor)

**Cost factor**: A hyperbolic function of cost relative to disposable capital:

> cost_factor = 1 / (1 + cost_impact / disposable_capital)

where cost_impact = transition cost + annual direct cost increase, and disposable_capital = capital − min_capital. PBC approaches 0 as cost approaches disposable capital and approaches 1 when costs are negligible.

**Risk aversion** (Chavas & Holt 1996): Risk-averse farmers weight potential losses more heavily than gains. Currently uses AFT base risk aversion (traditionalists > pioneers).

High risk → low PBC → lower adoption intention, even if attitude and norm are positive. This captures the empirical observation that risk is a key barrier to CA adoption in developing countries (Pannell et al. 2014).

### 4.11 Agronomic reasonableness filter

Some practice combinations are agronomically problematic. No-till without residue cover (bundles 4, 6) leaves soil unprotected against erosion, crusting, and temperature extremes. The model flags these bundles as **unreasonable** when residue opportunity cost is below a threshold (i.e. when retaining residue would be cheap). Unreasonable bundles are excluded from exploration and adoption.

This is not a hard constraint (when residue is expensive, these bundles are permitted) but a soft agronomic heuristic that prevents clearly counterproductive combinations.

### 4.12 Cover crop type selection

When cover crops are adopted, the type (legume vs non-legume) is determined from environmental conditions provided by LPJmL:

- **Non-legume** (catch crop): when both nitrogen leaching and fertiliser application are high, indicating excess reactive nitrogen. Non-legumes scavenge surplus N without adding more.
- **Legume**: otherwise, to provide biological nitrogen fixation.

This links farmer decisions to biogeochemical feedbacks in the coupled model.

### 4.13 Agent functional types: pioneer vs traditionalist

Farmers are differentiated into two agent functional types (AFTs):

| Parameter | Pioneer | Traditionalist |
|-----------|---------|----------------|
| Exploration probability | Higher (0.05) | Lower (0.01) |
| Transition threshold | Lower (easier to adopt) | Higher |
| Revert threshold | Higher (harder to revert) | Lower |
| Risk aversion | Lower | Higher |
| Min observation years | Fewer | More |

**Pioneers** adopt earlier, explore more, and tolerate more risk — corresponding to Rogers' "innovators" and "early adopters". **Traditionalists** require more evidence, are more risk-averse, and revert more easily — corresponding to "late majority" and "laggards".

Exploration probability is further modulated by:
- **Performance**: poor performers explore more (searching for better options)
- **Experience**: longer duration on current bundle increases willingness to experiment (confidence ramp)

### 4.14 Hysteresis: asymmetric transition/revert thresholds

The transition threshold for adopting a new bundle differs from the revert threshold for returning to a previous one (default: 0.5 vs 0.6). This asymmetry creates **hysteresis**: once adopted, a practice is retained even under moderate dissatisfaction. This prevents rapid oscillation and reflects the sunk-cost effect and learning investments associated with practice changes.

---

## 5. Spatial Structure: Countries, Cells, Farmers

The CA realisation introduces a hierarchical spatial structure:

### 5.1 Country level (`CACountry`)

Countries aggregate cells and provide country-level economic parameters from FAO:
- **Depreciation rate** (δ = CFC / NCS)
- **Investment rate** (i = GFCF / NCS)
- **Initial capital per hectare** (NCS / cropland_area)
- **Producer prices** by crop type (USD/tonne dry matter)

FAO data is loaded **once per simulation** for all countries, minimizing API calls. Data is cached at the class level to avoid redundant downloads during parallelization.

### 5.2 Cropland area calculation

Country-level cropland area is calculated as:

> cropland_ha = Σ (cftfrac × cell_area)

where:
- `cftfrac` = crop functional type fractions from LPJmL (sum over all crop bands)
- `cell_area` = cell area in m² (from pycopanlpjml), converted to hectares

This provides the denominator for computing capital per hectare from FAO Net Capital Stocks.

### 5.3 Cell level

Cells belong to countries and contain spatial data from LPJmL:
- Grid coordinates (lat, lon)
- Crop fractions (`cftfrac`)
- Harvest data (`pft_harvestc`)
- Environmental conditions (soil moisture, leaching, fertilizer)

### 5.4 Farmer level

Farmers are initialized on cells with crops (`cftfrac.sum() > 0`). Each farmer:
- Inherits country-level economic parameters
- Maintains individual capital, practice bundle, and TPB state
- Interacts with neighbouring farmers for social learning

---

## 6. FAO Data Integration

### 6.1 Data sources

The model uses two FAO datasets:

1. **Capital Stock (CS domain)**: Net Capital Stocks (NCS), Gross Fixed Capital Formation (GFCF), Consumption of Fixed Capital (CFC) for the Agriculture, Forestry and Fishing sector. Used to derive depreciation rate, investment rate, and initial capital.

2. **Producer Prices (PP domain)**: Prices by crop type in USD/tonne. Translated to LPJmL crop categories using `copan_eval`'s `FaoCropTranslator`.

### 6.2 Data retrieval

FAO data is retrieved via the `copan_eval` library:
- **API access**: Downloads from FAOSTAT API with authentication
- **Caching**: Downloaded data is cached locally as NetCDF files
- **Fallback**: If API fails, generates dummy data with warning

### 6.3 Crop price matching

LPJmL uses detailed crop bands (e.g., "rainfed temperate cereals", "irrigated temperate cereals") while FAO prices use aggregated categories ("temperate cereals"). The revenue calculation:
1. Strips "rainfed " or "irrigated " prefixes from LPJmL bands
2. Aggregates production across irrigation variants
3. Matches with FAO price categories
4. Multiplies production × price for each category

---

## 7. Decision Flow Summary

Each year, the CA farmer executes the following decision sequence:

1. **Parent update** — base farmer logic (soil C, yield, moisture tracking)
2. **Skip if control run** — no CA dynamics in baseline scenarios
3. **Update capital** — depreciation, profit calculation, reinvestment
4. **Check affordability** — deselect practices if capital too low
5. **Skip TPB if capital-constrained** — survival mode, no voluntary changes
6. **Decay old memories** — bounded rationality (§4.3)
7. **Check minimum observation period** — require sufficient data before transitioning (§4.2)
8. **Check fallback condition** — revert if sustained decline (§4.8)
9. **Find target bundle** — imitate best-performing neighbour OR explore randomly (§4.4)
10. **Adjust for affordability** — partial bundle if full target is too expensive (§4.9)
11. **Compute TPB** — attitude (own memory + social learning, §4.6), social norm (similarity + conformity, §4.7), PBC (cost-capital × risk, §4.10)
12. **Transition decision** — compare TPB to threshold (with hysteresis; §4.14)
13. **Apply transition** — deduct transition cost, update practices, record in memory

---

## 8. File Structure

| File | Purpose |
|------|---------|
| `tillage_farmer.py` | Regenerative Tillage realisation (single file, ~200 lines) |
| `ca_farmer.py` | CA farmer agent — capital dynamics, FAO data, revenue, costs, practice application |
| `ca_behaviour.py` | TPB decision model — bundles, memory, social learning, social norm, PBC, fallback, exploration |
| `ca_country.py` | Country-level FAO data loading and economic parameter extraction |
| `region.py` | Base Country class with cropland area calculation |
| `model.py` | Model class — entity initialization, update loop |
| `config.yaml` | Configuration — AFT parameters, practice costs, thresholds |

---

## 9. Configuration Parameters

Key parameters in `config.yaml`:

### AFT parameters (`aftpar.pioneer` / `aftpar.traditionalist`)
- `exploration_base_prob`: Base probability of random exploration
- `transition_threshold` / `revert_threshold`: TPB thresholds for adoption/reversion
- `min_observation_years`: Minimum years before considering transition
- `fallback_years`: Consecutive decline years before fallback
- `memory_decay_years`: Years until old memories expire
- `confidence_years`: Years to reach full confidence
- `risk_aversion`: Base risk aversion (0-1)
- `weight_bundle_similarity` / `weight_crop_similarity`: Social learning weights
- `conformity_bonus` / `conformity_penalty`: Social norm adjustments

### Farm economics (`farm_economics`)
- `n_survival_years`: Years of depreciation buffer for min_capital
- `savings_rate`: Fraction of profit reinvested

### Practice costs (`practice_costs`)
- `tillage.direct` / `tillage.transition`: No-till costs
- `cover_crop.direct` / `cover_crop.transition`: Cover crop costs
- `residue_on_field.direct` / `residue_on_field.transition`: Residue retention costs

### Residue economics (`residue_economics`)
- `use_costs.feed` / `use_costs.sale`: Opportunity costs by use type
- `default_removal_use`: Default use for opportunity cost calculation

---

## References

- Ajzen, I. (1991). The theory of planned behavior. *Organizational Behavior and Human Decision Processes*, 50(2), 179–211.
- Bandiera, O. et al. (2017). Labor markets and poverty in village economies. *Quarterly Journal of Economics*, 132(2), 811–870.
- Bandura, A. (1977). *Social Learning Theory*. Prentice Hall.
- Chavas, J.P. & Holt, M.T. (1996). Economic behavior under uncertainty: A joint analysis of risk preferences and technology. *Review of Economics and Statistics*, 78(2), 329–335.
- Cialdini, R.B., Reno, R.R., & Kallgren, C.A. (1990). A focus theory of normative conduct. *Journal of Personality and Social Psychology*, 58(6), 1015–1026.
- Cialdini, R.B. & Goldstein, N.J. (2004). Social influence: Compliance and conformity. *Annual Review of Psychology*, 55, 591–621.
- FAO (2017). The State of Food and Agriculture: Leveraging Food Systems for Inclusive Rural Transformation.
- FAO (2023). FAOSTAT Capital Stock methodology. Food and Agriculture Organization.
- Holling, C.S. (1978). *Adaptive Environmental Assessment and Management*. John Wiley & Sons.
- Jorgenson, D.W. (1963). Capital theory and investment behavior. *American Economic Review*, 53(2), 247–259.
- Kassam, A., Friedrich, T., Shaxson, F., & Pretty, J. (2009). The spread of Conservation Agriculture. *International Journal of Environmental Studies*, 66(6), 677–697.
- Katchova, A.L. & Dinterman, R. (2018). Evaluating financial stress and performance of beginning farmers during the agricultural downturn. *Agricultural Finance Review*, 78(4), 457–469.
- Lowder, S.K., Skoet, J., & Raney, T. (2016). The number, size, and distribution of farms, smallholder farms, and family farms worldwide. *World Development*, 87, 16–29.
- OECD (2009). *Measuring Capital — OECD Manual*, 2nd ed. OECD Publishing.
- Pannell, D.J., Llewellyn, R.S., & Corbeels, M. (2014). The farm-level economics of conservation agriculture for resource-poor farmers. *Agriculture, Ecosystems & Environment*, 187, 52–64.
- Pittelkow, C.M. et al. (2015). Productivity limits and potentials of the principles of conservation agriculture. *Nature*, 517, 365–368.
- Rogers, E.M. (2003). *Diffusion of Innovations*, 5th ed. Free Press.
- Simon, H.A. (1955). A behavioral model of rational choice. *Quarterly Journal of Economics*, 69(1), 99–118.
- Walters, C.J. (1986). *Adaptive Management of Renewable Resources*. Macmillan.
