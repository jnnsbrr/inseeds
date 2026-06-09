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
      │    └── Country-level statistics for social learning
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

2. **Social learning** (`compute_attitude_social_learning_local/country`): Performance comparison with neighbours using the proposed bundle.
   - **Local**: Weighted by bundle similarity + crop similarity (AFT params: `weight_bundle_similarity`, `weight_crop_similarity`)
   - **Country**: Uses cached country statistics comparing own performance to country average for the proposed bundle

Local vs country contributions are weighted by `weight_attitude_local` and `weight_attitude_country`.

### 3.2 Social Norm

Social norm reflects "what others are doing" (descriptive norm).

- **Local** (`compute_social_norm_local`): Similarity-weighted average across neighbours → sigmoid transformation with AFT-specific threshold (`threshold_social_norm_local`)
- **Country** (`compute_social_norm_country`): Fraction of farmers using the bundle → sigmoid with threshold (`threshold_social_norm_country`)

Local vs country contributions are weighted by `weight_social_norm_local` and `weight_social_norm_country`.

The sigmoid threshold model (Granovetter 1978): below threshold → social drag; above threshold → social boost.

### 3.3 Perceived Behavioral Control (PBC)

PBC is economics-based:

> PBC = pbc_base × cost_factor

Where:
- **pbc_base**: AFT-specific baseline (pioneers: 0.85, traditionalists: 0.65)
- **cost_factor**: `1 / (1 + cost_impact / disposable_capital)`

PBC approaches 0 as costs approach available capital. AFT differences are captured via `pbc_base`.

---

## 4. Decision Flow

Each year, TPB executes:

1. **Re-evaluate residue status** — Update bundle based on actual litter cover vs CA threshold (30%)
2. **Add observation** — Update regression accumulators (soil C, moisture, yield)
3. **Update bundle memory** — Store current trends for neighbour visibility
4. **Decay old memories** — Bounded rationality (`memory_decay_years`)
5. **Check minimum observation** — Require sufficient data before reconsidering (`min_observation_years`)
6. **Check fallback** — Revert if sustained decline (`fallback_years` consecutive years below baseline)
7. **Find target bundle**:
   - Try local neighbour imitation (best-performing neighbour's bundle)
   - If none better, try country-level best performer
   - If still none, maybe explore randomly (innovation diffusion)
8. **Adjust for affordability** — Reduce to partial bundle if full target unaffordable
9. **Compute TPB components** — Attitude, social norm, PBC
10. **Transition decision** — Compare TPB to threshold

### 4.1 Thresholds (Global, not AFT-specific)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `transition_threshold` | 0.5 | TPB score to adopt new bundle |
| `revert_threshold` | 0.6 | TPB score to revert (higher = hysteresis) |
| `evaluation_interval` | 0 | Years between re-evaluations |

### 4.2 Transition Blockers

The model tracks why transitions don't happen:

| Blocker | Meaning |
|---------|---------|
| `min_obs_years` | Not enough observation time yet |
| `no_target` | No better neighbour + exploration didn't trigger |
| `target_same` | Target bundle equals current (already optimal) |
| `tpb_low_pbc` | TPB failed due to cost/capital constraints |
| `tpb_low_social_norm_local/country` | TPB failed due to low social norm |
| `tpb_low_attitude_*` | TPB failed due to low attitude |
| `capital_survival` | Capital below survival threshold |

### 4.3 Transition Drivers

When transitions succeed, the model tracks the enabling factor:

| Driver | Meaning |
|--------|---------|
| `local_*` | Inspired by local neighbour |
| `country_*` | Inspired by country-level performer |
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

### 5.2 Annual Update

> K_{t+1} = K_t − δK_t + s × max(profit, 0)

Where:
- **δ**: Depreciation rate (CFC / NCS from FAO)
- **s**: Savings rate (default 0.15)
- **profit**: Revenue − variable costs − depreciation

### 5.3 Revenue Calculation

1. Get per-PFT harvest from LPJmL (gC/m²)
2. Multiply by crop fraction and cell area
3. Convert gC to tonnes dry matter (0.45 C fraction)
4. Aggregate irrigation variants to match FAO price categories
5. Multiply by FAO producer prices (USD/tonne)

### 5.4 Practice Costs (from config)

| Practice | Transition ($/ha) | Direct ($/ha/yr) |
|----------|-------------------|------------------|
| No-till | 70 | -50 (savings) |
| Cover crop | 25 | 75 |
| Residue retention | 10 | 50 |

When capital falls below minimum threshold (`n_survival_years × δ × K₀`), costly practices are dropped: cover crop first (highest direct cost), then residue, then no-till.

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

## 7. Residue Economics

### 7.1 Opportunity Cost

Residue retention has an opportunity cost based on spatial MADRaT data (Smerald et al. 2023):

> opportunity_cost = burnt × 0 + removed × 80 + recycled × 0 ($/ha/yr)

Fractions are weighted by actual crop composition from LPJmL.

### 7.2 Residue Status

The bundle's residue component (0/1) is dynamically updated based on actual litter cover:
- If `litter_cover ≥ 30%` and residue=0 → upgrade to 1 (certified CA)
- If `litter_cover < 30%` and residue=1 → downgrade to 0 (de-certified)

---

## 8. Agent Functional Types (AFT)

Farmers are differentiated into pioneers (25%) and traditionalists (75%).

### Key AFT Parameter Differences

| Parameter | Pioneer | Traditionalist |
|-----------|---------|----------------|
| `pbc_base` | 0.85 | 0.65 |
| `weight_attitude` | 0.8 | 0.6 |
| `weight_norm` | 0.2 | 0.4 |
| `exploration_base_prob` | 0.05 | 0.01 |
| `min_observation_years` | 2 | 3 |
| `fallback_years` | 8 | 10 |
| `threshold_social_norm_local` | 0.10 | 0.20 |

**Pioneers**: Higher PBC, weight attitude more, explore more, require less observation time.
**Traditionalists**: Lower PBC, weight norms more, explore less, more cautious.

---

## 9. Adaptive Management

### 9.1 Fallback Mechanism

After grace period (`min_observation_years`):
1. Track consecutive years where performance < baseline at transition
2. If decline continues for `fallback_years` → revert to previous bundle
3. Mark failed bundle (increment failure count)
4. Bundles with `failure_count ≥ max_failures` (default 2) excluded from exploration

### 9.2 Bundle Memory

Per-bundle memory with:
- Observed trends (soil C, moisture, yield)
- Duration on bundle
- Year of last update
- Failure count

Memories decay after `memory_decay_years`.

---

## 10. Cropland Area Calculation

Country-level cropland from LPJmL data:

> cropland_ha = Σ (cftfrac × cell_area)

Implementation uses `np.squeeze()` to prevent numpy broadcasting bugs when multiplying arrays with singleton dimensions.

Farm size at initialization uses mean `cftfrac` over spinup years for robustness.

---

## 11. File Structure

| File | Purpose |
|------|---------|
| `ca_farmer.py` | CAFarmer: capital, costs, revenue, practice application |
| `ca_behaviour.py` | TPB: bundles, memory, social learning, norms, PBC, fallback |
| `ca_country.py` | CACountry: FAO data loading, economic parameters |
| `region.py` | Country base class: cropland area |
| `model.py` | Model: initialization, update loop |
| `config.yaml` | All configuration parameters |
| `data/fao/*.py` | FAO data handlers |
| `data/residue.py` | MADRaT residue fraction data |

---

## References

- Ajzen, I. (1991). The theory of planned behavior. *Organizational Behavior and Human Decision Processes*, 50(2), 179–211.
- Granovetter, M. (1978). Threshold models of collective behavior. *American Journal of Sociology*, 83(6), 1420–1443.
- Jorgenson, D.W. (1963). Capital theory and investment behavior. *American Economic Review*, 53(2), 247–259.
- Kassam, A., Friedrich, T., Shaxson, F., & Pretty, J. (2009). The spread of Conservation Agriculture. *International Journal of Environmental Studies*, 66(6), 677–697.
- OECD (2009). *Measuring Capital — OECD Manual*, 2nd ed. OECD Publishing.
- Rogers, E.M. (2003). *Diffusion of Innovations*, 5th ed. Free Press.
- Smerald, A. et al. (2023). Global crop residue management data. MADRaT.
