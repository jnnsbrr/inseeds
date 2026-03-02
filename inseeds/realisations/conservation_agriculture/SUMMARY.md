# Conservation Agriculture Farmer Model - Summary

This document summarizes the CA Farmer model (`ca_farmer.py` + `ca_behaviour.py`) and compares it to the simpler Tillage Farmer model (`tillage_farmer.py`).

---

## Overview

| Aspect | Tillage Farmer | CA Farmer + CA Behaviour |
|--------|---------------|--------------------------|
| **Practices** | Single (tillage only) | Three: tillage, cover crops, residue retention |
| **Decision unit** | Binary flip (0↔1) | 8 practice bundles (combinations) |
| **Learning** | Year-to-year comparison | Trend-based over observation periods |
| **Economics** | Simple PBC decay | Full capital stock model |

---

## Key Features of the CA Model

### 1. Multi-Practice Bundle Decisions

- Farmers evaluate **8 practice combinations** ranging from conventional to full CA
- Can adopt **multiple practices simultaneously** (multi-flip adoption)
- Named bundles for interpretability:
  - `conventional` (0,0,0)
  - `residue_only` (0,0,1)
  - `cover_crop_only` (0,1,0)
  - `cover_residue` (0,1,1)
  - `notill_only` (1,0,0)
  - `notill_residue` (1,0,1)
  - `notill_cover` (1,1,0)
  - `full_ca` (1,1,1)

### 2. Trend-Based Adaptive Learning

- Tracks **performance trends** (soil carbon, root moisture, crop yield) over time
- **Bundle memory** for all 8 combinations with configurable decay (default: 30 years)
- **Failure tracking**: learns from past unsuccessful switches to avoid repeating mistakes
- Auto-scaling normalizes performance scores to neighborhood min/max

### 3. Neighbor-Guided Adoption

- Identifies **best-performing neighbor's full bundle**
- Can adopt their entire practice combination if affordable
- **Similarity-weighted social learning**: neighbors with similar bundles contribute more
- **Confidence weighting**: longer observation duration = higher confidence in neighbor's results

### 4. Economic Realism

- **Capital as annual stock**:
  - Income: profit from crop yield
  - Costs: maintenance + direct costs (annual) + transition costs (on switch)
- **Affordability check**: proposes affordable subset if full bundle too expensive
- **Practice deselection**: when capital drops below threshold, practices are dropped in order:
  1. Cover crops (highest direct cost)
  2. Residue retention (moderate opportunity cost)
  3. No-till (last, as it provides fuel savings)
- **Residue economics**: capital-constrained retention based on opportunity cost

### 5. Intelligent Exploration

- **Pioneer vs. traditionalist** AFTs with different exploration probabilities
- **Adaptive exploration**: increases when performance is poor or stagnant
- **Reasonableness filter**: excludes agronomically problematic combinations
  - `notill_only` and `notill_cover` flagged as unreasonable when residue is cheap
- **Learning from failures**: skips bundles that have failed multiple times

### 6. Fallback Mechanism

- **Automatic reversion** after sustained performance decline (default: 5 years)
- **Grace period** (default: 3 years) before fallback evaluation starts
- Compares against **baseline score at switch time**, not just previous year
- Failed bundles have their failure count incremented for future reference

### 7. Hysteresis / Learning Inertia

- **Minimum observation period** before considering a switch
- **Asymmetric thresholds**: higher threshold for reverting vs. adopting new practices
- Prevents rapid oscillation between practices

### 8. Conformity Pressure in Social Norm

- Social norm **boosted in homogeneous neighborhoods** (everyone doing the same thing)
- **Penalty for deviating** from dominant practice
- Captures real-world social dynamics in farming communities

### 9. Cover Crop Type Selection

- **Legume vs. non-legume** selection based on N dynamics
- Non-legumes only when BOTH high leaching AND high fertilizer (excess N)
- Otherwise legumes for additional N fixation
- Requires LPJmL inputs: `cell_leaching`, `cell_runoff`, `cell_fertilizer`

---

## What Tillage Farmer Does Simpler

| Aspect | Advantage |
|--------|-----------|
| Single practice | Simpler logic, easier to understand |
| No capital tracking | Fewer parameters to calibrate |
| Binary decisions | Faster computation |
| Year-to-year comparison | Less memory overhead |

---

## Theory of Planned Behaviour (TPB) Implementation

Both models use TPB, but with different sophistication:

### Tillage Farmer TPB
```
TPB = (weight_attitude × attitude + weight_norm × social_norm) × PBC
```
- Attitude: year-to-year soil/yield comparison
- Social norm: proportion of neighbors using alternative practice
- PBC: decays after switch, recovers slowly

### CA Farmer TPB
```
TPB = (weight_attitude × attitude + weight_norm × social_norm) × PBC
```
- **Attitude**: blend of own experience (from memory) + social learning (similarity-weighted)
- **Social norm**: similarity-weighted prevalence + conformity pressure
- **PBC**: hyperbolic decay based on disposable capital and total transition cost

---

## File Structure

```
inseeds/components/farming/
├── ca_farmer.py        # Main farmer class (economics, update loop)
├── ca_behaviour.py     # TPB decision model (bundle evaluation, learning)
└── tillage_farmer.py   # Simpler single-practice model
```

---

## Configuration

Key parameters in `config.yaml`:

```yaml
farm_economics:
  initial_capital: 100.0      # USD thousands
  min_capital: 10.0           # Liquidity threshold
  max_capital: 500.0          # Capital ceiling
  maintenance_cost: 20.0      # Annual base cost
  yield_profit_factor: 0.5    # Yield to profit conversion

practice_costs:
  tillage:
    transition: 70.0          # Equipment investment
    direct: -50.0             # Fuel SAVINGS (negative cost)
  cover_crop:
    transition: 25.0          # Seed drill, learning
    direct: 65.0              # Seeds, planting, termination
  residue_on_field:
    transition: 10.0          # Management change
    direct: 55.0              # Opportunity cost

exploration:
  residue_cost_threshold: 30.0
  max_failures: 2
```

---

## Future Work

See `CONSIDERATIONS.md` for detailed discussion of:

1. Regional economic data (FAO integration)
2. Crop-specific behavior
3. Weather/climate risk
4. Credit/loan system
5. Extension services
6. Land tenure / farm size
7. Labor constraints
8. Soil type heterogeneity
9. Market access
10. Unit system (Pint)
11. Empirical validation
