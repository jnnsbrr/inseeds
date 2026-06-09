# Conservation Agriculture Model - Considerations and Future Work

This document outlines considerations for the Conservation Agriculture (CA) farmer decision model and identifies areas for future development and potential enhancements.

## Overview

The CA model implements a Theory of Planned Behaviour (TPB) framework with combination-based adaptive learning for multi-practice adoption decisions (tillage, cover crops, residue retention). While scientifically grounded, several aspects require further development for improved realism and regional applicability.

---

## 1. Regional Economic Data

### Current State
- Practice costs are based on literature values (primarily US/EU sources)
- Values are placeholders that need regional calibration
- **✅ FAO Producer Prices IMPLEMENTED**: Use `ConservationAgricultureFarmerFAO` for crop-specific, region-specific profit calculation

### FAO Producer Prices (IMPLEMENTED)

The `ConservationAgricultureFarmer` class **requires** FAO producer prices (PP domain) for profit calculation. The data is automatically downloaded from FAOSTAT if not available in `{sim_path}/input/fao_pft_prices.nc`.

**How it works:**
- On farmer initialization, the system checks for `{sim_path}/input/fao_pft_prices.nc`
- If not found, it automatically downloads and processes data from FAOSTAT
- No manual setup required - just ensure internet access on first run

**Manual Data Preparation (if needed):**
```bash
python -m inseeds.components.farming.prepare_fao_producer_prices \
    --output {sim_path}/input/fao_pft_prices.nc --years 1990-2020
```

**Unit Conversion Chain:**
1. FAO: USD/tonne (fresh weight)
2. → USD/tonne (dry matter) via Wirsenius (2000) dry matter factors
3. In profit calculation: gC → tonnes dry matter via carbon fraction (0.45)

**Profit Calculation (in `_calculate_profit()`):**
```python
production_gC = pft_harvestc * cftfrac * area_m2
production_tonnes_dm = production_gC / (CARBON_FRACTION * 1e6)
profit = sum(production_tonnes_dm * fao_pft_prices)
```

### What's Still Missing
| Parameter | Current Source | Needed |
|-----------|---------------|--------|
| `practice_costs.transition` | US literature (farmdoc, SARE) | Regional labor/equipment costs |
| `practice_costs.direct` | US literature (USDA, SARE) | Regional input costs |
| `residue_opportunity_cost` | Generic estimates | Local livestock density, biofuel demand |

### Impact
- ~~Economic decisions may not reflect local realities~~ (addressed by FAO prices)
- Practice costs still need regional calibration
- Adoption patterns could differ in regions with different cost structures

### Potential Data Sources
- ~~FAOSTAT (crop prices)~~ **✅ IMPLEMENTED**
- FAOSTAT (production costs) - for practice costs
- World Bank Living Standards Measurement Studies
- Regional agricultural statistics (Eurostat, USDA NASS)
- Smerald et al. (2023) for residue usage patterns

---

## 2. Crop-Specific Behavior

### Current State
- Model treats all crops uniformly
- Single `cropyield` value aggregated across crops

### What's Missing
- **Differential CA benefits**: No-till benefits vary significantly:
  - Maize: Often shows yield drag in early years
  - Wheat: Generally responds well to CA
  - Soybeans: Benefit from cover crop N fixation
- **Crop rotation effects**: CA benefits compound with diverse rotations
- **Cover crop species selection**: Currently simplified to legume/non-legume

### Scientific Basis
- Pittelkow et al. (2015): Meta-analysis showing crop-specific CA responses
- Kassam et al. (2019): Crop-specific adoption patterns globally

### Potential Implementation
```yaml
crop_specific_effects:
  maize:
    notill_yield_penalty_years: 3  # Initial yield drag
    cover_crop_n_benefit: 0.8      # Moderate N benefit
  wheat:
    notill_yield_penalty_years: 1
    cover_crop_n_benefit: 0.6
  soybean:
    notill_yield_penalty_years: 0
    cover_crop_n_benefit: 1.2      # High N fixation benefit
```

---

## 3. Weather/Climate Risk

### Current State
- No explicit climate risk modeling
- Trend-based learning provides some implicit adaptation (poor years affect trends)
- No proactive behavior based on forecasts

### What's Missing
- **Drought risk perception**: Farmers in drought-prone areas may value CA's moisture retention more
- **Flood risk**: Affects residue management decisions
- **Seasonal forecasts**: Real farmers adjust practices based on El Niño/La Niña predictions
- **Insurance interactions**: Crop insurance affects risk-taking behavior

### Scientific Basis
- CA provides drought resilience through improved water infiltration
- Cover crops can reduce flood damage but may compete for water in dry years

### Potential Implementation
```python
def adjust_weights_for_climate_risk(self):
    """Adjust performance weights based on climate variability."""
    drought_risk = self.cell.drought_probability
    # Increase moisture weight in drought-prone areas
    self.weight_moisture *= (1 + drought_risk)
```

---

## 4. Credit/Loan System

### Current State
- Capital is a stock that cannot go negative
- No debt mechanism
- Farmers simply cannot adopt if capital insufficient

### What's Missing
- **Credit access**: Varies by region, farm size, tenure status
- **Interest rates**: Affect profitability of investments
- **Debt burden**: Accumulated debt affects future decisions
- **Microfinance**: Important in developing countries

### Real-World Relevance
- Many CA transitions are financed through loans
- Equipment purchases (no-till planters) often require credit
- Debt can trap farmers in conventional systems

### Potential Implementation
```python
class FarmFinance:
    def __init__(self):
        self.capital = 1000.0
        self.debt = 0.0
        self.credit_limit = 500.0
        self.interest_rate = 0.05
    
    def can_afford(self, cost):
        return (self.capital + self.credit_limit - self.debt) >= cost
    
    def take_loan(self, amount):
        self.debt += amount
        self.capital += amount
    
    def annual_debt_service(self):
        interest = self.debt * self.interest_rate
        self.capital -= interest
```

---

## 5. Extension Services / Information Diffusion

### Current State
- Learning is purely neighbor-based (local social learning)
- No external information sources

### What's Missing
- **Agricultural extension services**: Government/NGO programs promoting CA
- **Media/internet**: Information diffusion beyond local networks
- **Demonstration farms**: High-visibility adoption examples
- **Subsidies/incentives**: Government programs for CA adoption
- **Input supplier influence**: Agrochemical companies may discourage CA

### Scientific Basis
- Rogers (2003): Diffusion of Innovations theory
- Extension services are primary CA promotion mechanism in many countries

### Potential Implementation
```python
def information_exposure(self):
    """Calculate information exposure from multiple sources."""
    neighbor_info = self._attitude_social_learning(bundle)
    extension_info = self.region.extension_intensity * 0.3
    media_info = self.region.media_penetration * 0.1
    
    return (
        self.weight_neighbor * neighbor_info +
        self.weight_extension * extension_info +
        self.weight_media * media_info
    )
```

---

## 6. Land Tenure / Farm Size

### Current State
- **Farm size calculation implemented**: `farm_size = sum(cftfrac * area)` in hectares
- **All economic values now scale by farm size**:
  - Capital (initial, min, max thresholds)
  - Maintenance costs
  - Practice costs (direct and transition)
  - Profit from yield
  - Residue opportunity cost
- Config values are specified per hectare ($/ha), automatically scaled at runtime
- Farm size is available as output variable for analysis

### What's Still Missing
- **Tenant vs. owner**: Tenants have less incentive for long-term soil investment
- **Economies of scale**: Large farms may have lower per-ha costs
- **Land fragmentation**: Common in developing countries, affects mechanization

### Farm Size Calculation (Implemented)
```python
# In farmer.py
@property
def farm_size(self):
    """Farm size in hectares."""
    cftfrac = self.cell.from_earth.cftfrac
    area = self.cell.from_earth.terr_area  # km²
    total_cftfrac = float(np.sum(cftfrac.values))
    area_ha = float(np.asarray(area.values).mean()) * 100.0  # km² → ha
    return total_cftfrac * area_ha
```

### Used For (All Implemented)
- ✅ Scaling maintenance costs
- ✅ Scaling practice costs (direct and transition)
- ✅ Calculating total profit (yield × farm_size × yield_profit_factor)
- ✅ Capital thresholds (min_capital, max_capital)
- ✅ Residue opportunity cost for retention calculation

### Scientific Basis
- Knowler & Bradshaw (2007): Farm size affects CA adoption
- Tenure security strongly predicts long-term investment in soil health

### Future Enhancement: Tenure Effects
```yaml
tenure_effects:
  owner:
    discount_rate: 0.05      # Values long-term benefits
    investment_horizon: 20   # Years
  tenant:
    discount_rate: 0.15      # Higher discount rate
    investment_horizon: 5    # Shorter planning horizon

economies_of_scale:
  # Per-ha costs decrease with farm size
  scale_factor: 0.9  # Cost multiplier per doubling of size
```

---

## 7. Labor Constraints

### Current State
- Only capital constrains adoption
- Labor implicitly included in costs

### What's Missing
- **Labor availability**: Seasonal labor shortages
- **Labor costs**: Vary dramatically by region
- **Family vs. hired labor**: Different cost structures
- **Cover crop labor**: Planting and termination require labor at specific times
- **Opportunity cost of labor**: Off-farm employment options

### Real-World Relevance
- Cover crops add labor demand at potentially busy times
- No-till reduces labor but requires management skill
- Labor constraints are primary barrier in some regions

### Potential Implementation
```python
class LaborConstraints:
    def __init__(self):
        self.family_labor_hours = 2000  # Annual hours available
        self.hired_labor_cost = 15.0    # $/hour
        
    def labor_required(self, bundle):
        base = 500  # Conventional farming hours
        if bundle[0] == 1:  # No-till
            base -= 100  # Labor savings
        if bundle[1] == 1:  # Cover crops
            base += 50   # Additional labor
        return base
```

---

## 8. Soil Type Heterogeneity

### Current State
- CA benefits assumed uniform across soil types
- No soil-specific decision adjustments

### What's Missing
- **Clay soils**: May compact without tillage, but benefit from cover crop roots
- **Sandy soils**: Drain quickly, cover crops critical for organic matter
- **Soil drainage**: Affects residue decomposition and cover crop success
- **Slope**: Affects erosion risk and CA benefits

### Scientific Basis
- Pittelkow et al. (2015): CA yield effects vary by soil texture
- Heavy clay soils may show initial yield penalties with no-till

### Potential Implementation
```python
def soil_adjusted_attitude(self, bundle):
    """Adjust attitude based on soil suitability for CA."""
    base_attitude = self._compute_attitude(bundle)
    
    # Soil-specific adjustments
    if self.cell.soil_type == "clay":
        if bundle[0] == 1:  # No-till
            base_attitude *= 0.9  # Slight penalty for compaction risk
        if bundle[1] == 1:  # Cover crops
            base_attitude *= 1.1  # Bonus for root channels
    
    return base_attitude
```

---

## 9. Market Access

### Current State
- No spatial market modeling
- Prices assumed uniform

### What's Missing
- **Distance to markets**: Affects input costs and output prices
- **Infrastructure quality**: Roads, storage facilities
- **Market information**: Price transparency
- **Contract farming**: May restrict practice choices
- **Certification premiums**: Organic/sustainable premiums for CA products

### Real-World Relevance
- Remote farmers face higher input costs (seeds, equipment)
- Market access affects profitability of all farming systems

### Potential Implementation
```python
def market_adjusted_profit(self, yield_value):
    """Adjust profit based on market access."""
    base_price = self.region.crop_price
    transport_cost = self.distance_to_market * self.transport_rate
    
    effective_price = base_price - transport_cost
    return yield_value * effective_price
```

---

## 10. Subsidies and Incentives

### Current State
- No subsidy or incentive mechanism implemented
- Farmers bear full cost of CA transition

### What's Missing
- **Agri-environmental payments**: CAP (EU), EQIP (US), similar programs globally
- **Carbon credits**: Payments for soil carbon sequestration
- **Certification premiums**: Organic/sustainable product price premiums
- **Equipment subsidies**: Government programs for no-till equipment

### Real-World Relevance
- Many CA adopters receive significant financial support
- Subsidies can offset the initial capital drain during transition years
- Without subsidies, model may underestimate adoption rates in regions with strong policy support

### Potential Implementation
```python
def calculate_subsidies(self):
    """Calculate annual subsidies for CA practices."""
    subsidy = 0.0
    bundle = self.behaviour.practice_bundle
    
    if bundle[0] == 1:  # No-till
        subsidy += self.region.notill_subsidy_per_ha * self.farm_size
    if bundle[1] == 1:  # Cover crops
        subsidy += self.region.cover_crop_subsidy_per_ha * self.farm_size
    if bundle[2] == 1:  # Residue retention
        subsidy += self.region.residue_subsidy_per_ha * self.farm_size
    
    return subsidy
```

### Data Sources
- EU CAP payment databases
- USDA EQIP payment data
- National agricultural ministry reports

---

## 11. Residue Opportunity Cost Refinement

### Current State
- Residue opportunity cost is a fixed value from config
- Assumes residue has alternative use (fodder, biofuel, etc.)

### What's Missing
- **Regional livestock density**: Determines actual demand for residue as fodder
- **Biofuel demand**: Varies by region and policy
- **Actual farmer practice**: Many farmers burn residue (no opportunity cost) or have no alternative market
- **Potential double-counting**: If farmer wasn't selling residue before, there's no real opportunity cost

### Scientific Basis
- Smerald et al. (2023): Global dataset on cereal residue production and usage
- Residue usage varies dramatically: 
  - High livestock regions: High opportunity cost (fodder value)
  - Low livestock regions: Often burned or left (no opportunity cost)

### Potential Implementation
```python
def calculate_residue_opportunity_cost(self):
    """Calculate residue opportunity cost based on local conditions."""
    livestock_density = self.region.livestock_density  # heads/ha
    
    if livestock_density > 0.5:  # High livestock density
        # Residue has fodder value
        return self.residue_fodder_price * self.residue_production
    elif self.region.biofuel_demand > 0:
        # Residue has biofuel value
        return self.residue_biofuel_price * self.residue_production
    else:
        # Residue typically burned or left - no opportunity cost
        return 0.0
```

### Data Sources
- FAO livestock statistics
- Smerald et al. (2023) residue usage dataset
- Regional biofuel policy data

---

## 12. Unit System

### Current State
- Raw numbers without explicit unit handling
- Units documented in comments/config but not enforced

### What's Missing
- **Type safety**: No protection against unit mismatches
- **Automatic conversion**: Manual conversion error-prone
- **Documentation**: Units scattered across codebase

### Planned Solution
- Pint-based unit system (deferred to separate implementation)
- Would replace pycopancore's custom unit handling

### Example Implementation
```python
from pint import UnitRegistry
ureg = UnitRegistry()

class EconomicParameters:
    capital: ureg.Quantity = 1000 * ureg.USD
    yield_price: ureg.Quantity = 200 * ureg.USD / ureg.tonne
    transition_cost: ureg.Quantity = 70 * ureg.USD / ureg.hectare
```

---

## 13. Validation Against Empirical Data

### Current State
- Model logic is scientifically grounded (TPB, CA literature)
- Unit tests verify internal consistency
- No validation against real-world adoption data

### What's Missing
- **Historical validation**: Compare model predictions to observed CA adoption trends
- **Regional calibration**: Adjust parameters to match regional patterns
- **Sensitivity analysis**: Identify most influential parameters
- **Pattern-oriented modeling**: Validate emergent patterns (adoption curves, spatial clustering)

### Potential Validation Datasets
- Kassam et al. (2019): Global CA adoption statistics
- FAO AQUASTAT: Irrigation and land use data
- National agricultural censuses
- LSMS household surveys

### Validation Approach
```python
def validate_adoption_curve(model_results, empirical_data):
    """Compare model adoption curve to empirical data."""
    # S-curve fitting
    model_adoption = model_results.groupby('year')['practice_bundle.id'].apply(
        lambda x: (x > 0).mean()  # Fraction adopting any CA practice
    )
    
    # Compare to empirical adoption rates
    rmse = np.sqrt(((model_adoption - empirical_data) ** 2).mean())
    return rmse
```

---

## Priority Ranking

Based on impact and feasibility:

| Priority | Item | Impact | Status |
|----------|------|--------|--------|
| 1 | Regional Economic Data (FAO prices, capital init) | High | **✅ FAO Prices IMPLEMENTED** (capital init still TODO) |
| 2 | Validation Against Empirical Data | High | **TODO** |
| 3 | Farm Size-Dependent Costs | High | **✅ IMPLEMENTED** |
| 4 | Crop-Specific Behavior | High | **TODO** |
| 5 | Weather/Climate Risk | Medium | **TODO** |
| 6 | Extension Services | Medium | **TODO** |
| 7 | Land Tenure (owner vs tenant) | Medium | **TODO** |
| 8 | Credit System | Medium | **TODO** |
| 9 | Soil Heterogeneity | Medium | **TODO** |
| 10 | Subsidies/Incentives | Medium | **TODO** |
| 11 | Residue Opportunity Cost Refinement | Medium | **TODO** |
| 12 | Labor Constraints | Low | **TODO** |
| 13 | Market Access | Low | **TODO** |
| 14 | Unit System (pint integration) | Low | **TODO** |

---

## References

- Kassam, A., Friedrich, T., & Derpsch, R. (2019). Global spread of Conservation Agriculture. International Journal of Environmental Studies, 76(1), 29-51.
- Knowler, D., & Bradshaw, B. (2007). Farmers' adoption of conservation agriculture: A review and synthesis of recent research. Food Policy, 32(1), 25-48.
- Pittelkow, C. M., et al. (2015). Productivity limits and potentials of the principles of conservation agriculture. Nature, 517(7534), 365-368.
- Rogers, E. M. (2003). Diffusion of Innovations (5th ed.). Free Press.
- Smerald, A., et al. (2023). A global dataset for the production and usage of cereal residues. Scientific Data.
