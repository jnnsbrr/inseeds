# Conservation Agriculture Realisation: Model Summary

This document describes the **Conservation Agriculture (CA)** realisation of InSEEDS. The model couples farmer decision-making based on Theory of Planned Behaviour (TPB; Ajzen 1991) with LPJmL crop model outputs.

---

## 1. Architecture Overview

The realisation has a hierarchical structure:

```
Model
 └── World
      ├── Countries (CACountry)
      │    ├── FAO economic data (capital, prices, depreciation)
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

Attitude combines two sources, weighted by AFT parameters (`weight_own_land`, `weight_social_learning`):

1. **Own-land experience** (`compute_attitude_own_land`): Trend-based evaluation of soil carbon, root moisture, and crop yield since last transition. Uses linear regression over observation period to filter inter-annual noise. Declining performance → higher attitude toward change.

2. **Social learning** (`compute_attitude_social_learning`): Performance comparison at three spatial scales:
   - **Local**: Weighted by bundle similarity + crop similarity
   - **Country**: Country-level average performance for the proposed bundle
   - **Cluster**: Agroecological cluster average (climatically similar countries)

Weights (`weight_attitude_local`, `weight_attitude_country`, `weight_attitude_cluster`) are AFT-specific and redistributed when levels are disabled.

#### 3.1.1 Unified Performance Scoring

Performance comparisons use a unified scoring system that handles the challenge of comparing metrics with different scales (yield ~70 gC/m², soil carbon ~5000 gC/m², moisture ~200 mm):

**Per-Metric Scoring:**

```
score_metric = (1 - trend_weight) × norm_level + trend_weight × trend_score
```

Where:
- **norm_level** = level / reference — ratio to country mean, centered at 1.0
- **trend_score** = trend / ref_trend_country — ratio to country std (z-score style)
- **ref_trend_country** — std of trends across all farmers in the country
- **trend_weight** — from config, metric-specific (e.g., `trend_weight_soil: 0.7`)

**Symmetric Normalization Design:**

Both levels AND trends use the same normalization principle — ratio to country reference:

| Component | Normalization | Reference | Interpretation |
|-----------|---------------|-----------|----------------|
| Level | level / ref_level | country mean | 1.0 = average level |
| Trend | trend / ref_trend | country std | 0.0 = average, ±1 = ±1 std |

This provides:
1. **Automatic scale handling** — different metrics are normalized automatically
2. **Country-specific calibration** — adapts to local conditions
3. **Single calibration parameter** — `attitude_sensitivity` controls the overall sensitivity
4. **Clean architecture** — same normalization logic for both components

**Country Reference Values:**

| Reference | Computed From | Purpose |
|-----------|---------------|---------|
| `reference_yield_level` | mean(farmer yields) | Normalize yield levels |
| `reference_soilc_level` | mean(farmer soil C) | Normalize soil C levels |
| `reference_moisture_level` | mean(farmer moisture) | Normalize moisture levels |
| `reference_yield_trend` | std(farmer yield trends) | Normalize yield trends |
| `reference_soilc_trend` | std(farmer soil C trends) | Normalize soil C trends |
| `reference_moisture_trend` | std(farmer moisture trends) | Normalize moisture trends |

Using **std** (not mean) for trend references ensures:
- Mean trend is often ~0, which would break normalization
- Std represents "typical variation" — a natural scale for trends
- trend/ref_trend = 1 means "one std above average" (clearly good!)

**Attitude Calculation:**

```
attitude = sigmoid(score_diff × attitude_sensitivity)
```

The single `attitude_sensitivity` parameter (config: `tpb.attitude_sensitivity`) controls how strongly normalized score differences translate to attitudes:
- Typical CA benefit → score_diff ≈ 0.16 → with sensitivity 5 → attitude ≈ 0.70
- Exceptional CA benefit → score_diff ≈ 0.45 → with sensitivity 5 → attitude ≈ 0.91
- Declining farm → score_diff ≈ -0.25 → with sensitivity 5 → attitude ≈ 0.21

**Overall Performance Score:**

```
score = weight_yield × score_yield + weight_soil × score_soil + weight_moisture × score_moisture
```

**Why Metric-Specific Trend Weighting?**

| Metric | Trend Weight | Rationale |
|--------|-------------|-----------|
| Yield | 0.2 | Level-dominant: current productivity matters most |
| Soil C | 0.7-0.8 | Trend-dominant: legacy effects, decades to rebuild |
| Moisture | 0.3 | Mostly level-based with some trend sensitivity |

Pioneers have higher `trend_weight_soil` (0.8 vs 0.7) reflecting their longer-term orientation (Rogers 2003).

**Country Reference Values:**

Country means (`reference_yield`, `reference_soilc`, `reference_moisture`) are computed as the mean across all farmers in the country, updated each simulation year. This enables meaningful comparison within and across countries.

**Scientific Basis:**
- Normalization approach: Yield gap analysis (van Ittersum et al. 2013)
- Soil carbon trends: IPCC relative stock change methods (Paustian et al. 2016)
- Multi-attribute utility: Andrews et al. (2004) Soil Quality Index

### 3.2 Social Norm

Social norm reflects "what others are doing" (descriptive norm) at three scales:

- **Local** (`compute_social_norm_local`): Similarity-weighted adoption fraction among neighbours
- **Country** (`compute_social_norm_country`): Fraction of farmers using the bundle nationally
- **Cluster** (`compute_social_norm_cluster`): Adoption fraction across climatically similar countries

Each uses a sigmoid transformation with AFT-specific thresholds (e.g., `threshold_social_norm_local`).

The sigmoid threshold model (Granovetter 1978): below threshold → social drag; above threshold → social boost.

Weights (`weight_social_norm_local`, `weight_social_norm_country`, `weight_social_norm_cluster`) are AFT-specific and redistributed when levels are disabled.

### 3.3 Perceived Behavioral Control (PBC)

PBC is economics-based:

> PBC = pbc_base × cost_factor

Where:
- **pbc_base**: AFT-specific baseline (pioneers: 0.85, traditionalists: 0.65)
- **cost_factor**: `1 / (1 + cost_impact / disposable_capital)`

PBC approaches 0 as costs approach available capital. AFT differences are captured via `pbc_base`.

### 3.4 Weight Redistribution

When a spreading level is disabled (e.g., `enable_cluster: false` for single-country runs), its weight is redistributed proportionally to enabled levels:

```python
# Example: if cluster disabled with weights (0.7, 0.2, 0.1)
# Redistributed: (0.778, 0.222, 0.0) - total unchanged
```

---

## 4. Decision Flow

Each year, TPB executes:

1. **Re-evaluate residue status** — Update bundle based on actual litter cover vs CA threshold (30%)
2. **Add observation** — Update regression accumulators (soil C, moisture, yield)Up
3. **Update bundle memory** — Store current trends for neighbour visibility
4. **Decay old memories** — Bounded rationality (`memory_decay_years`)
5. **Check minimum observation** — Require sufficient data before reconsidering
6. **Check fallback** — Revert if sustained decline (`fallback_years` consecutive years below baseline)
7. **Find target bundle**:
   - Try local neighbour imitation (best-performing neighbour's bundle)
   - If none better, try country-level best performer
   - If none better, try cluster-level best performer
   - If still none, maybe explore randomly (innovation diffusion)
8. **Adjust for affordability** — Reduce to partial bundle if full target unaffordable
9. **Compute TPB components** — Attitude, social norm, PBC
10. **Transition decision** — Compare TPB to threshold

### 4.1 Thresholds (Global, not AFT-specific)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `transition_threshold` | 0.5 | TPB score for any practice change |

### 4.2 Evaluation Timing

`min_observation_years` serves dual purpose:
1. **Data quality**: n_obs must be >= min_observation_years before transitioning
2. **Evaluation timing**: Counter (`_observation_years`) is randomized around min_observation_years to desynchronize farmers and prevent artificial evaluation waves

After each evaluation (whether transition happens or not), the counter is reset to a random value drawn from `Normal(min_obs, min_obs/2)`.

### 4.3 Transition Blockers

The model tracks why transitions don't happen:

| Blocker | Meaning |
|---------|---------|
| `observation_years` | Not yet time to re-evaluate (commitment period) |
| `min_obs_years` | Not enough observation data yet |
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

### 5.2 FAO Baseline + Deviations Model

FAO provides the baseline capital dynamics. We track DEVIATIONS from this baseline:

> K_{t+1} = K × (1 + i − δ) + Δrevenue − Δcosts

Where:
- **i − δ**: FAO net rate (investment − depreciation, typically +2-7%/year)
- **Δrevenue**: current_revenue − baseline_revenue
- **Δcosts**: cost difference between current and baseline bundle

### 5.3 Baselines (Set at Initialization)

| Baseline | Source | Represents |
|----------|--------|------------|
| **baseline_revenue** | Average over 2015-2025 (historic LPJmL data) | Typical yields already in FAO |
| **baseline_bundle** | Initial practice bundle at simulation start | Typical costs already in FAO |

### 5.4 How Deviations Work

| Condition | Effect |
|-----------|--------|
| Δrevenue = 0, Δcosts = 0 | Farmer follows FAO baseline exactly |
| Better yields (Δrevenue > 0) | Capital grows faster than baseline |
| Worse yields (Δrevenue < 0) | Capital grows slower than baseline |
| CA with higher costs (Δcosts > 0) | Reduces capital vs baseline |
| CA with savings, e.g. no-till (Δcosts < 0) | Increases capital vs baseline |

### 5.5 Why This Works

FAO (i − δ) already captures everything at sector average:
- Depreciation (physical capital loss)
- Reinvestment from typical profits
- Government subsidies (EU CAP, US farm bill)
- Bank loans and credit
- Typical yields and costs

We don't need to model all these components explicitly - just the DEVIATIONS.
This keeps the model simple while allowing individual farmer performance to matter.

### 5.6 Practice Costs

These are the costs that differ from conventional farming, used in Δcosts calculation.

**Capital-Scaled Costs**

Costs are scaled by farmer's capital intensity relative to a reference country (default: USA, where literature values originate). The reference capital is dynamically retrieved from FAO data for the configured reference country:

> cost = min + (max - min) × clamp(capital_per_ha / reference_capital_per_ha, 0, 1)

Configuration: `practice_costs.reference_capital_country: "USA"` in config.yaml

This accounts for global variation in equipment and input costs. Variation is **conservative** (1.3-2x), reflecting that:
- Equipment: India custom hire $15-27/ha vs US $66/ha (~2-3x) [1,6]
- Fuel prices: Global variation ~1.5x (excluding oil-producer outliers)
- Seeds: Similar global prices; import costs can increase developing country prices

| Practice | Transition Range | Direct Range | Source |
|----------|-----------------|--------------|--------|
| No-till | $40-66/ha | -$40 to -$55/ha (savings) | [1,2,3,6,7] |
| Cover crop | $12-22/ha | $55-85/ha | [4,5] |
| Residue retention | $0 | *dynamic* | Singh & Schiere 1995 |

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

### 5.7 Survival Threshold

When capital falls below minimum threshold (configurable, default $100/ha), costly practices are dropped in order: cover crop first (highest direct cost), then residue, then no-till.

### 5.8 Capital Dynamics References

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

### 8.2 Residue Status

The bundle's residue component (0/1) is dynamically updated based on actual litter cover:
- If `litter_cover ≥ 30%` and residue=0 → upgrade to 1 (certified CA)
- If `litter_cover < 30%` and residue=1 → downgrade to 0 (de-certified)

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