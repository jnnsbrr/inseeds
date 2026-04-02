"""TPB-based decision model for multi-practice Conservation Agriculture adoption.

Implements Theory of Planned Behaviour (Ajzen 1991) for Conservation Agriculture
decisions over three practices: tillage (conventional vs no-till), cover crop,
and residue retention.

Key features:
- Social learning from neighbours (Bandura 1977)
- Own-experience memory with decay
- Fallback to previous practices under sustained decline (adaptive management)
- Affordability constraints on practice adoption

References:
- Ajzen, I. (1991). The theory of planned behavior. Organizational Behavior
  and Human Decision Processes, 50(2), 179-211.
- Bandura, A. (1977). Social Learning Theory. Prentice Hall.
- Holling, C.S. (1978). Adaptive Environmental Assessment and Management.
"""

from abc import ABC, abstractmethod

import numpy as np


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Map real values to (0, 1) for TPB attitude/norm scores.

    Uses tanh-based sigmoid: output of 0.5 when x=0, approaches 0/1 at extremes.
    Useful for converting unbounded scores to probability-like values.

    Parameters
    ----------
    x : float or array-like
        Input value(s).

    Returns
    -------
    float or ndarray
        Sigmoid output in (0, 1). Zero maps to 0.5.
    """
    return 0.5 * (np.tanh(x) + 1)


# =============================================================================
# PRACTICE BUNDLE DEFINITIONS
# =============================================================================

# Practice bundle: (tillage, cover_crop, residue_on_field), each 0 or 1
#
# tillage:         0 = no-till (CA practice), 1 = conventional tillage
# cover_crop:      0 = no cover crop,         1 = cover crop planted
# residue:         0 = baseline removal,      1 = residue retained
#
# Full CA = (0, 1, 1): no-till + cover crops + residue retention

BUNDLE_NAMES = {
    (0, 0, 0): "notill_only",         # No-till without cover/residue (risky)
    (0, 0, 1): "notill_residue",      # No-till + residue (common CA entry)
    (0, 1, 0): "notill_cover_crop",   # No-till + cover (needs residue ideally)
    (0, 1, 1): "conservation",        # Full Conservation Agriculture
    (1, 0, 0): "conventional",        # All conventional practices
    (1, 0, 1): "residue_only",        # Conventional tillage + residue retention
    (1, 1, 0): "cover_crop_only",     # Conventional tillage + cover crops
    (1, 1, 1): "cover_crop_residue",  # Conventional tillage + cover + residue
}

# Reverse lookup: name → tuple
BUNDLE_TUPLES = {v: k for k, v in BUNDLE_NAMES.items()}

# Numeric IDs for output/analysis (0-7)
BUNDLE_IDS = {
    (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 2, (0, 1, 1): 3,
    (1, 0, 0): 4, (1, 0, 1): 5, (1, 1, 0): 6, (1, 1, 1): 7,
}


# =============================================================================
# DEFAULT TIME PARAMETERS (can be overridden in config per AFT)
# =============================================================================
# These are fallback defaults. Actual values come from config.yaml aftpar.
# Different AFT types (pioneer vs traditionalist) can have different values.

# How long farmers remember outcomes from past practice bundles
# After this period, old experiences are "forgotten" (reset to neutral)
DEFAULT_MEMORY_DECAY_YEARS = 30

# How many consecutive years of declining performance before reverting
# to the previous practice bundle (adaptive management / fallback)
DEFAULT_FALLBACK_YEARS = 10

# Years to reach full confidence in own/neighbour experience (linear ramp)
DEFAULT_CONFIDENCE_YEARS = 10


# =============================================================================
# ABSTRACT DECISION MODEL BASE CLASS
# =============================================================================

class DecisionModel(ABC):
    """Abstract base for farmer decision models.

    Tracks:
    - Current practice bundle (tillage, cover_crop, residue)
    - Memory of past outcomes for each bundle tried
    - Current state snapshot for computing trends since last switch

    Subclasses implement specific decision logic (e.g., TPB).
    """

    def _get_aft_param(self, param_name):
        """Get AFT parameter from agent, raising error if missing.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve from the agent.

        Returns
        -------
        value
            The parameter value.

        Raises
        ------
        AttributeError
            If the parameter is not defined on the agent (missing from config).
        """
        if not hasattr(self.agent, param_name):
            aft_name = getattr(self.agent.aft, "name", "unknown")
            raise AttributeError(
                f"Missing AFT parameter '{param_name}' for AFT '{aft_name}'. "
                f"Add it to config.yaml under aftpar.{aft_name}.{param_name}"
            )
        return getattr(self.agent, param_name)

    def __init__(self, agent):
        """Initialize decision model from agent's current practices.

        Parameters
        ----------
        agent : ConservationAgricultureFarmer
            The farmer agent owning this behaviour.
        """
        self.agent = agent

        # -----------------------------------------------------------------
        # Initialize current practice bundle from agent state
        # -----------------------------------------------------------------
        # Bundle is a tuple: (tillage, cover_crop, residue_on_field)
        # Each element is 0 (practice OFF) or 1 (practice ON)
        #
        # Convention (matches LPJmL):
        #   tillage=0 means no tillage (no-till, CA practice)
        #   tillage=1 means tillage is ON (conventional)
        #   cover_crop=0 means no cover crop, >0 means cover crop planted
        #   residue=0 means baseline removal, >baseline means retention

        self._practice_bundle = (
            int(agent.tillage),
            1 if agent.cover_crop > 0 else 0,
            1 if agent.residue_on_field > agent.residue_baseline else 0,
        )

        # Proposed bundle for potential switch (set by update())
        self._proposed_bundle = None

        # Previous bundle (for fallback mechanism)
        self._previous_bundle = None

        # Counter for consecutive years of declining performance
        self._decline_years = 0

        # -----------------------------------------------------------------
        # Online regression state for trend computation
        # -----------------------------------------------------------------
        # Instead of storing full history, we track running sums for online
        # linear regression. This allows computing trends from all data points
        # without storing them.
        #
        # For y = a + b*t, slope b = (n*sum_ty - sum_t*sum_y) / (n*sum_tt - sum_t²)
        # We track separate sums for soil, moisture, and yield.

        t_start, history = self._get_initial_state(agent)

        self.current_state = {
            "t_start": t_start,           # Year of last practice switch
            "baseline_score": 0.0,        # Weighted score at switch (for fallback)
            # Online regression accumulators (initialized from history if available)
            "n": 0,                        # Number of observations
            "sum_t": 0.0,                  # Sum of time indices
            "sum_tt": 0.0,                 # Sum of t²
            "sum_soil": 0.0,               # Sum of soil values
            "sum_t_soil": 0.0,             # Sum of t * soil
            "sum_moisture": 0.0,           # Sum of moisture values
            "sum_t_moisture": 0.0,         # Sum of t * moisture
            "sum_yield": 0.0,              # Sum of yield values
            "sum_t_yield": 0.0,            # Sum of t * yield
            "last_obs_year": -1,          # Last year observation was added (prevent duplicates)
        }

        # Initialize accumulators with historic data if available
        self._initialize_regression_state(agent, history)

        # -----------------------------------------------------------------
        # Initialize per-bundle memory
        # -----------------------------------------------------------------
        # Farmers remember outcomes from each bundle they've tried
        # Memory includes: trends, duration used, when last updated, failure count

        self.bundle_memory = {
            (t, c, r): {
                "trend_soil": 0.0,      # Annual soil C change (gC/m²/yr)
                "trend_moisture": 0.0,  # Annual moisture change
                "trend_yield": 0.0,     # Annual yield change
                "duration": 0,          # Years this bundle was used
                "last_updated": 0,      # Year memory was last updated
                "failure_count": 0,     # Times this bundle led to fallback
            }
            for t in (0, 1) for c in (0, 1) for r in (0, 1)
        }

        # -----------------------------------------------------------------
        # Initialize memory for current bundle with historic data
        # -----------------------------------------------------------------
        # If we have historic data, compute the trend for the current bundle
        # so farmers start with meaningful memory (not all zeros)
        self._initialize_current_bundle_memory()

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _get_initial_state(self, agent):
        """Get initial t_start and historic time series for trend computation.

        If historic output data is available (from_earth has multiple time steps),
        use config.outputyear as t_start and return the full time series for
        soil, moisture, and yield. This allows computing proper regression-based
        trends from all data points.

        Otherwise, use current sim_year and return current values as single-point
        history.

        Parameters
        ----------
        agent : Farmer
            The farmer agent being initialized.

        Returns
        -------
        tuple[int, list[dict]]
            (t_start, [{"soilc": float, "moisture": float, "yield": float}, ...])
            List contains one dict per year of history.
        """
        from_earth = agent.cell.from_earth
        has_history = hasattr(from_earth, 'time') and len(from_earth.time) > 1

        if has_history:
            try:
                t_start = int(agent.model.config.outputyear)
            except AttributeError:
                t_start = agent.model.lpjml.sim_year
                has_history = False

        if has_history:
            # Extract full time series for regression
            n_years = len(from_earth.time)
            history = []
            for i in range(n_years):
                history.append({
                    "soilc": agent._get_from_earth("soilc_agr_layer", as_scalar=True, band=0, time_idx=i),
                    "moisture": agent._get_from_earth("rootmoist_agr", as_scalar=True, time_idx=i),
                    "yield": agent._get_from_earth("harvestc", as_scalar=True, time_idx=i),
                })
        else:
            # No history: single observation at current year
            t_start = agent.model.lpjml.sim_year
            history = [{
                "soilc": agent.soilc,
                "moisture": agent.root_moisture,
                "yield": agent.cropyield,
            }]

        return t_start, history

    def _initialize_regression_state(self, agent, history):
        """Initialize online regression accumulators from historic data.

        Parameters
        ----------
        agent : Farmer
            The farmer agent.
        history : list[dict]
            List of {"soilc", "moisture", "yield"} dicts, one per year.
        """
        state = self.current_state
        for i, obs in enumerate(history):
            t = i  # Time index (0, 1, 2, ...)
            state["n"] += 1
            state["sum_t"] += t
            state["sum_tt"] += t * t
            state["sum_soil"] += obs["soilc"]
            state["sum_t_soil"] += t * obs["soilc"]
            state["sum_moisture"] += obs["moisture"]
            state["sum_t_moisture"] += t * obs["moisture"]
            state["sum_yield"] += obs["yield"]
            state["sum_t_yield"] += t * obs["yield"]

        # Mark current year as already observed (history includes up to sim_year)
        state["last_obs_year"] = agent.model.lpjml.sim_year

    def _add_observation(self, soilc, moisture, cropyield):
        """Add a new observation to the online regression accumulators.

        Called each year to update the running sums for trend computation.
        Skips if observation for current year was already added (e.g., from
        historic initialization).

        Parameters
        ----------
        soilc : float
            Current soil carbon value.
        moisture : float
            Current root moisture value.
        cropyield : float
            Current crop yield value.
        """
        current_year = self.agent.model.lpjml.sim_year
        state = self.current_state

        # Skip if we already have an observation for this year
        if state["last_obs_year"] >= current_year:
            return

        t = state["n"]  # Next time index
        state["n"] += 1
        state["sum_t"] += t
        state["sum_tt"] += t * t
        state["sum_soil"] += soilc
        state["sum_t_soil"] += t * soilc
        state["sum_moisture"] += moisture
        state["sum_t_moisture"] += t * moisture
        state["sum_yield"] += cropyield
        state["sum_t_yield"] += t * cropyield
        state["last_obs_year"] = current_year

    def _compute_slope(self, sum_y, sum_ty):
        """Compute regression slope from running sums.

        Parameters
        ----------
        sum_y : float
            Sum of y values.
        sum_ty : float
            Sum of t * y values.

        Returns
        -------
        float
            Slope (trend) of the regression line, or 0.0 if insufficient data.
        """
        state = self.current_state
        n = state["n"]
        if n < 2:
            return 0.0

        sum_t = state["sum_t"]
        sum_tt = state["sum_tt"]
        denominator = n * sum_tt - sum_t * sum_t

        if abs(denominator) < 1e-10:
            return 0.0

        return (n * sum_ty - sum_t * sum_y) / denominator

    def _initialize_current_bundle_memory(self):
        """Initialize bundle_memory for current bundle from regression state.

        Uses the computed trends from the online regression to populate
        the bundle_memory entry for the current practice bundle.
        Also sets baseline_score for fallback comparison.
        """
        n = self.current_state["n"]
        if n < 2:
            return

        current_year = self.agent.model.lpjml.sim_year
        trend = self.current_trend

        # Set baseline_score from historic trend (for fallback comparison)
        self.current_state["baseline_score"] = self._weighted_score(trend)

        self.bundle_memory[self._practice_bundle] = {
            "trend_soil": trend["soil"],
            "trend_moisture": trend["moisture"],
            "trend_yield": trend["yield"],
            "duration": n,
            "last_updated": current_year,
            "failure_count": 0,
        }

    def _update_current_bundle_memory(self):
        """Update bundle_memory for current bundle with latest trends.

        Called every year to keep bundle_memory up-to-date. This ensures
        neighbours see current performance when evaluating which bundle
        to imitate, not stale data from the last switch.
        """
        n = self.current_state["n"]
        if n < 2:
            return

        trend = self.current_trend
        current_year = self.agent.model.lpjml.sim_year

        # Preserve failure_count from existing memory
        old_fc = self.bundle_memory[self._practice_bundle].get("failure_count", 0)

        self.bundle_memory[self._practice_bundle].update({
            "trend_soil": trend["soil"],
            "trend_moisture": trend["moisture"],
            "trend_yield": trend["yield"],
            "duration": n,
            "last_updated": current_year,
            "failure_count": old_fc,
        })

    # -------------------------------------------------------------------------
    # Properties for external access
    # -------------------------------------------------------------------------

    @property
    def practice_bundle(self):
        """Current bundle as numeric ID (0-7)."""
        return BUNDLE_IDS.get(self._practice_bundle, -1)

    @property
    def practice_bundle_name(self):
        """Current bundle as human-readable name."""
        return BUNDLE_NAMES.get(self._practice_bundle, "unknown")

    @property
    def proposed_bundle(self):
        """Proposed bundle as numeric ID (-1 if none)."""
        return BUNDLE_IDS.get(self._proposed_bundle, -1) if self._proposed_bundle else -1

    @property
    def proposed_bundle_name(self):
        """Proposed bundle as human-readable name."""
        return BUNDLE_NAMES.get(self._proposed_bundle, "unknown") if self._proposed_bundle else ""

    # -------------------------------------------------------------------------
    # Trend computation
    # -------------------------------------------------------------------------

    @property
    def current_trend(self):
        """Annual change in soil C, moisture, and yield since last switch.

        Computes trends using linear regression over all observations since
        the last practice switch, not just start and end points. This provides
        more robust trend estimates that are less sensitive to noise.

        Returns
        -------
        dict
            Keys: 'soil', 'moisture', 'yield'. Values: annual rate of change
            (regression slope).
        """
        state = self.current_state

        # Need at least 2 observations for regression
        if state["n"] < 2:
            return {"soil": 0.0, "moisture": 0.0, "yield": 0.0}

        return {
            "soil": self._compute_slope(state["sum_soil"], state["sum_t_soil"]),
            "moisture": self._compute_slope(state["sum_moisture"], state["sum_t_moisture"]),
            "yield": self._compute_slope(state["sum_yield"], state["sum_t_yield"]),
        }

    # -------------------------------------------------------------------------
    # Memory management
    # -------------------------------------------------------------------------

    def record_switch(self, new_bundle):
        """Store outcome of current bundle and prepare for switch to new_bundle.

        Called when farmer commits to switching practices. Records the
        performance of the outgoing bundle for future reference.

        Parameters
        ----------
        new_bundle : tuple of (int, int, int)
            The new (tillage, cover_crop, residue_on_field) bundle.
        """
        current_year = self.agent.model.lpjml.sim_year
        n_obs = self.current_state["n"]

        # -----------------------------------------------------------------
        # Store outcome of outgoing bundle (if used for at least 2 years)
        # -----------------------------------------------------------------
        if n_obs >= 2:
            trend = self.current_trend

            # Preserve failure count from previous memory
            old_fc = self.bundle_memory[self._practice_bundle].get("failure_count", 0)

            # Update memory for the bundle we're leaving
            self.bundle_memory[self._practice_bundle] = {
                "trend_soil": trend["soil"],
                "trend_moisture": trend["moisture"],
                "trend_yield": trend["yield"],
                "duration": n_obs,
                "last_updated": current_year,
                "failure_count": old_fc,
            }

        # -----------------------------------------------------------------
        # Prepare for new bundle: reset regression state
        # -----------------------------------------------------------------

        # Remember previous bundle (for potential fallback)
        self._previous_bundle = self._practice_bundle

        # Capture baseline score before resetting (for fallback comparison)
        # This is the performance level we expect to maintain or exceed
        baseline = self._weighted_score(self.current_trend) if n_obs >= 2 else 0.0

        # Reset regression accumulators for new bundle
        # Start with current values as first observation
        self.current_state = {
            "t_start": current_year,
            "baseline_score": baseline,  # Score to beat with new bundle
            "n": 1,
            "sum_t": 0.0,
            "sum_tt": 0.0,
            "sum_soil": self.agent.soilc,
            "sum_t_soil": 0.0,
            "sum_moisture": self.agent.root_moisture,
            "sum_t_moisture": 0.0,
            "sum_yield": self.agent.cropyield,
            "sum_t_yield": 0.0,
            "last_obs_year": current_year,  # Mark this year as observed
        }

        # Switch to new bundle
        self._practice_bundle = new_bundle

        # Reset decline counter
        self._decline_years = 0

    def _decay_old_memories(self):
        """Reset memories older than memory_decay_years.

        Old experiences become irrelevant as conditions change (climate,
        markets, technology). This implements bounded rationality.
        """
        current_year = self.agent.model.lpjml.sim_year
        memory_decay = self._get_aft_param("memory_decay_years")

        for bundle, mem in self.bundle_memory.items():
            if mem["duration"] > 0:
                years_since_update = current_year - mem["last_updated"]

                if years_since_update > memory_decay:
                    # Reset to neutral (no memory)
                    self.bundle_memory[bundle] = {
                        "trend_soil": 0.0,
                        "trend_moisture": 0.0,
                        "trend_yield": 0.0,
                        "duration": 0,
                        "last_updated": 0,
                        "failure_count": 0,
                    }

    # -------------------------------------------------------------------------
    # Abstract methods (implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    def update(self):
        """Update decision state (compute proposed bundle, TPB scores)."""
        pass

    @abstractmethod
    def should_switch(self):
        """Return True if farmer should switch to proposed bundle."""
        pass


# =============================================================================
# TPB (THEORY OF PLANNED BEHAVIOUR) IMPLEMENTATION
# =============================================================================

class TPB(DecisionModel):
    """TPB decision model with social learning and fallback.

    Implements Ajzen (1991) Theory of Planned Behaviour:
    - Attitude: evaluation of the behavior (own experience + social learning)
    - Social Norm: perceived social pressure (what neighbours do)
    - PBC: Perceived Behavioral Control (can I afford it?)

    Extended with:
    - Social learning from neighbours (Bandura 1977)
    - Fallback mechanism for adaptive management (Holling 1978)
    - Random exploration for innovation diffusion

    The TPB intention formula used is multiplicative:
        TPB = (w_att × Attitude + w_norm × SocialNorm) × PBC

    This means PBC acts as a gate: if farmer cannot afford the practice
    (PBC → 0), intention is blocked regardless of attitude/norms.
    """

    def __init__(self, agent):
        """Initialize TPB model with zero scores."""
        super().__init__(agent)

        # TPB components (updated each timestep)
        self._tpb = 0.0          # Overall intention score
        self._attitude = 0.0     # Attitude toward behavior
        self._social_norm = 0.0  # Subjective norm
        self._pbc = 0.0          # Perceived behavioral control

    # -------------------------------------------------------------------------
    # Properties for external access to TPB components
    # -------------------------------------------------------------------------

    @property
    def tpb(self):
        """Overall TPB intention score."""
        return self._tpb

    @property
    def attitude(self):
        """Attitude component of TPB."""
        return self._attitude

    @property
    def social_norm(self):
        """Social norm component of TPB."""
        return self._social_norm

    @property
    def pbc(self):
        """Perceived behavioral control component of TPB."""
        return self._pbc

    # =========================================================================
    # MAIN UPDATE LOGIC
    # =========================================================================

    def update(self):
        """Compute proposed bundle and TPB scores for this timestep.

        Decision flow:
        1. Add current observation to regression accumulators
        2. Update bundle_memory with current trends (for neighbour visibility)
        3. Decay old memories (bounded rationality)
        4. Check if minimum observation period has passed
        5. Check fallback condition (sustained decline → revert)
        6. Find best-performing neighbour's bundle OR explore randomly
        7. Adjust target bundle for affordability
        8. Compute TPB scores for the proposed bundle
        """
        # -----------------------------------------------------------------
        # Step 1: Add current year's observation to regression
        # -----------------------------------------------------------------
        self._add_observation(
            self.agent.soilc,
            self.agent.root_moisture,
            self.agent.cropyield
        )

        # -----------------------------------------------------------------
        # Step 2: Update bundle_memory with current trends
        # -----------------------------------------------------------------
        # Keep bundle_memory up-to-date so neighbours see current performance
        self._update_current_bundle_memory()

        # -----------------------------------------------------------------
        # Step 3: Decay old memories
        # -----------------------------------------------------------------
        self._decay_old_memories()

        # -----------------------------------------------------------------
        # Step 4: Require minimum observation years before switching
        # -----------------------------------------------------------------
        # Avoids noisy decisions based on single-year fluctuations
        # Typical value: 3 years (allows trends to stabilize)
        # With historic data initialization, farmers start with enough history

        n_obs = self.current_state["n"]
        min_obs = self._get_aft_param("min_observation_years")

        if n_obs < min_obs:
            self._tpb = 0.0
            self._proposed_bundle = None
            return

        # -----------------------------------------------------------------
        # Step 5: Check fallback condition
        # -----------------------------------------------------------------
        # If performance has declined for FALLBACK_YEARS consecutive years,
        # propose reverting to the previous bundle (adaptive management)

        if self._check_fallback():
            return  # Fallback sets _proposed_bundle and _tpb internally

        # -----------------------------------------------------------------
        # Step 6: Find target bundle (neighbour imitation or exploration)
        # -----------------------------------------------------------------

        # First, try to imitate best-performing neighbour
        target_bundle = self._most_promising_bundle()

        # If no better neighbour, maybe explore randomly
        if target_bundle is None:
            target_bundle = self._maybe_explore_bundle()

        # No change proposed
        if target_bundle is None or target_bundle == self._practice_bundle:
            self._tpb = 0.0
            self._proposed_bundle = None
            return

        # -----------------------------------------------------------------
        # Step 7: Adjust for affordability
        # -----------------------------------------------------------------
        # If farmer can't afford full target bundle, find affordable subset

        affordable_bundle = self._affordable_bundle(target_bundle)

        # If affordable bundle is same as current, no change
        if affordable_bundle == self._practice_bundle:
            self._tpb = 0.0
            self._proposed_bundle = None
            return

        # -----------------------------------------------------------------
        # Step 8: Compute TPB scores for proposed bundle
        # -----------------------------------------------------------------
        self._proposed_bundle = affordable_bundle
        self._compute_tpb_for_bundle(affordable_bundle)

    # =========================================================================
    # FALLBACK MECHANISM (Adaptive Management)
    # =========================================================================

    def _check_fallback(self):
        """Check if farmer should revert to previous bundle due to sustained decline.

        Implements adaptive management: if outcomes have declined for
        fallback_years consecutive years, propose reverting to the
        previous practice bundle.

        Returns
        -------
        bool
            True if fallback is triggered (sets _proposed_bundle internally).
        """
        # Can't fall back if no previous bundle
        if self._previous_bundle is None:
            return False

        # Allow grace period for new practices to show effects
        # Grace period = min_observation_years (reuse existing param)
        n_obs = self.current_state["n"]
        grace_period = self._get_aft_param("min_observation_years")
        if n_obs < grace_period:
            return False

        # -----------------------------------------------------------------
        # Compare current performance to baseline at switch time
        # -----------------------------------------------------------------
        # baseline_score captures the trend score at the time of switch.
        # If current score is worse than baseline, we're declining.
        baseline = self.current_state.get("baseline_score", 0.0)
        current_score = self._weighted_score(self.current_trend)

        # Track consecutive years of decline
        if current_score < baseline:
            self._decline_years += 1
        else:
            self._decline_years = 0  # Reset if performance improves

        # -----------------------------------------------------------------
        # Trigger fallback after sustained decline
        # -----------------------------------------------------------------
        fallback_years = self._get_aft_param("fallback_years")
        if self._decline_years >= fallback_years:
            # Mark current bundle as "failed" (reduces future exploration)
            self.bundle_memory[self._practice_bundle]["failure_count"] += 1

            # Propose reverting to previous bundle
            self._proposed_bundle = self._previous_bundle

            # Fallback bypasses TPB: this is an emergency response, not a
            # planned behavior change. Set intention directly to 1.0.
            # TPB factors are left unchanged (not meaningful for fallback).
            self._tpb = 1.0

            return True

        return False

    # =========================================================================
    # BUNDLE REASONABLENESS CHECK
    # =========================================================================

    def _is_reasonable_bundle(self, bundle):
        """Check if bundle is agronomically reasonable.

        No-till without residue cover is problematic:
        - Soil needs protection from erosion and crusting
        - Residue provides this protection
        - If residue is cheap (low opportunity cost), farmer should keep it

        Parameters
        ----------
        bundle : tuple
            (tillage, cover_crop, residue_on_field)

        Returns
        -------
        bool
            True if bundle is reasonable, False if it should be avoided.
        """
        t, c, r = bundle

        exploration = self.agent.model.config.coupled_config.exploration
        residue_threshold = getattr(exploration, "residue_cost_threshold", 30.0)

        # -----------------------------------------------------------------
        # No-till without residue is risky
        # -----------------------------------------------------------------
        # t=0 is no-till, t=1 is conventional tillage
        # No-till alone (no cover, no residue): soil exposed
        if t == 0 and c == 0 and r == 0:
            # Only unreasonable if residue is cheap (could easily retain it)
            if self.agent.residue_opportunity_cost_per_ha < residue_threshold:
                return False

        # No-till + cover but no residue: still risky in off-season
        if t == 0 and c == 1 and r == 0:
            if self.agent.residue_opportunity_cost_per_ha < residue_threshold:
                return False

        return True

    # =========================================================================
    # RANDOM EXPLORATION (Innovation Diffusion)
    # =========================================================================

    def _maybe_explore_bundle(self):
        """Randomly explore a new bundle (innovation diffusion).

        Some farmers try new practices without neighbour influence:
        - "Pioneers" explore more frequently
        - Poor performers explore more (searching for better options)
        - Longer experience increases willingness to try new things
        - Bundles that failed before are avoided

        Returns
        -------
        tuple or None
            Random valid bundle to try, or None if no exploration.
        """
        # -----------------------------------------------------------------
        # Determine exploration probability
        # -----------------------------------------------------------------

        # Base probability depends on farmer type (from config)
        # Pioneers (innovators) are more willing to experiment
        base_prob = self._get_aft_param("exploration_base_prob")

        # Poor performers explore more (searching for better options)
        # Threshold and multiplier are configurable
        current_score = self._weighted_score(self.current_trend)
        normalized = self._normalize_score(current_score)
        poor_performance_threshold = self._get_aft_param("poor_performance_threshold")
        poor_performance_multiplier = self._get_aft_param("poor_performance_multiplier")
        if normalized < poor_performance_threshold:
            base_prob *= poor_performance_multiplier

        # Experience affects willingness to explore (smooth ramp based on confidence_years)
        # Early: 0.5x exploration (cautious), Experienced: 1.5x exploration (confident)
        n_obs = self.current_state["n"]
        confidence_years = self._get_aft_param("confidence_years")
        experience_factor = min(1.0, n_obs / confidence_years)
        exploration_modifier = 0.5 + experience_factor * 1.0  # Ramps from 0.5 to 1.5
        base_prob *= exploration_modifier

        # Cap exploration probability (configurable)
        max_exploration_prob = self._get_aft_param("max_exploration_prob")
        explore_prob = min(base_prob, max_exploration_prob)

        # -----------------------------------------------------------------
        # Random draw: explore or not?
        # -----------------------------------------------------------------
        if np.random.random() > explore_prob:
            return None  # No exploration this year

        # -----------------------------------------------------------------
        # Select valid bundle to explore
        # -----------------------------------------------------------------
        exploration = self.agent.model.config.coupled_config.exploration
        max_failures = getattr(exploration, "max_failures", 2)

        # Generate all possible bundles
        all_bundles = [(t, c, r) for t in (0, 1) for c in (0, 1) for r in (0, 1)]

        # Filter to valid options:
        # - Not current bundle
        # - Not failed too many times
        # - Agronomically reasonable
        valid = [
            b for b in all_bundles
            if b != self._practice_bundle
            and self.bundle_memory[b]["failure_count"] < max_failures
            and self._is_reasonable_bundle(b)
        ]

        if not valid:
            return None

        # Random selection from valid bundles
        return valid[np.random.randint(len(valid))]

    # =========================================================================
    # SWITCH DECISION
    # =========================================================================

    def should_switch(self):
        """Determine if farmer should switch to proposed bundle.

        Compares TPB intention score to threshold. Higher threshold
        when reverting (to avoid oscillation).

        Returns
        -------
        bool
            True if TPB exceeds threshold and switch should occur.
        """
        # Higher threshold for reverting (avoid flip-flopping)
        if self._proposed_bundle == self._previous_bundle:
            threshold = self._get_aft_param("revert_threshold")
        else:
            threshold = self._get_aft_param("switch_threshold")

        return self._tpb > threshold

    # =========================================================================
    # APPLY BUNDLE TO AGENT
    # =========================================================================

    def apply_bundle(self, bundle):
        """Apply practice bundle to agent and push to LPJmL.

        Sets agent's tillage, cover_crop, and residue_on_field attributes
        and sends updates to the coupled LPJmL model.

        Parameters
        ----------
        bundle : tuple of (int, int, int)
            (tillage, cover_crop, residue_on_field), each 0 or 1.
        """
        # -----------------------------------------------------------------
        # Tillage: 0 = no-till (CA), 1 = conventional tillage
        # Directly maps to LPJmL's with_tillage
        # -----------------------------------------------------------------
        self.agent.tillage = bundle[0]

        # -----------------------------------------------------------------
        # Cover crop: 0 = none, 1/2 = type based on conditions
        # -----------------------------------------------------------------
        if bundle[1] == 1:
            # Determine cover crop type based on environmental conditions
            # 1 = non-legume (catch crop), 2 = legume (N-fixing)
            self.agent.cover_crop = self.agent._indicate_cover_crop_type()
        else:
            self.agent.cover_crop = 0

        # -----------------------------------------------------------------
        # Residue retention: 0 = baseline, 1 = retain (capital-constrained)
        # -----------------------------------------------------------------
        if bundle[2] == 1:
            # Retention level depends on capital vs opportunity cost
            opp_cost = self.agent.residue_opportunity_cost
            if opp_cost > 0:
                affordable = min(1.0, self.agent.capital / opp_cost)
            else:
                affordable = 1.0
            self.agent.residue_on_field = max(self.agent.residue_baseline, affordable)
        else:
            self.agent.residue_on_field = self.agent.residue_baseline

        # -----------------------------------------------------------------
        # Push changes to LPJmL
        # -----------------------------------------------------------------
        for attr in ["tillage", "cover_crop", "residue_on_field"]:
            self.agent.set_lpjml(attribute=attr)

    # =========================================================================
    # NEIGHBOUR COMPARISON METHODS
    # =========================================================================

    def _most_promising_bundle(self):
        """Find best-performing neighbour's bundle (if better than self).

        Social learning: farmers observe neighbours and may imitate those
        who are performing better (Bandura 1977).

        Returns
        -------
        tuple or None
            Best neighbour's bundle, or None if no neighbour is better.
        """
        best_neighbour = None
        best_gap = 0.0

        for neighbour in self.agent.neighbourhood:
            # Skip neighbours who aren't performing better
            if not self._is_better_performing(neighbour):
                continue

            # Track neighbour with largest performance gap
            gap = self._performance_gap(neighbour)
            if gap > best_gap:
                best_gap = gap
                best_neighbour = neighbour

        if best_neighbour is None:
            return None

        return best_neighbour.behaviour._practice_bundle

    def _weighted_score(self, trend):
        """Combine soil, moisture, yield trends into single utility score.

        Farmers weight different outcomes based on their priorities.
        This weighted sum represents overall "performance" of a bundle.

        Parameters
        ----------
        trend : dict
            Keys: 'soil', 'moisture', 'yield'. Values: annual rates of change.

        Returns
        -------
        float
            Weighted performance score.
        """
        return (
            self.agent.weight_soil * trend["soil"]
            + self.agent.weight_moisture * trend["moisture"]
            + self.agent.weight_yield * trend["yield"]
        )

    def _is_better_performing(self, neighbour):
        """Check if neighbour has higher weighted score than self.

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        bool
            True if neighbour's score exceeds self's score.
        """
        neighbour_score = self._weighted_score(neighbour.behaviour.current_trend)
        own_score = self._weighted_score(self.current_trend)
        return neighbour_score > own_score

    def _performance_gap(self, neighbour):
        """Compute positive performance difference (neighbour - self).

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        float
            Positive gap (0 if neighbour is worse).
        """
        neighbour_score = self._weighted_score(neighbour.behaviour.current_trend)
        own_score = self._weighted_score(self.current_trend)
        return max(0.0, neighbour_score - own_score)

    # =========================================================================
    # SCORE NORMALIZATION
    # =========================================================================

    def _neighbourhood_score_range(self):
        """Get min/max weighted scores in neighbourhood (for normalization).

        Returns
        -------
        tuple of (float, float)
            (min_score, max_score) in neighbourhood including self.
        """
        if not self.agent.neighbourhood:
            return (0.0, 1.0)

        # Collect all scores (neighbours + self)
        scores = [
            self._weighted_score(n.behaviour.current_trend)
            for n in self.agent.neighbourhood
        ]
        scores.append(self._weighted_score(self.current_trend))

        min_s, max_s = min(scores), max(scores)

        # Avoid division by zero when all scores are identical
        if max_s - min_s < 1e-6:
            return (min_s - 0.5, max_s + 0.5)

        return (min_s, max_s)

    def _normalize_score(self, score):
        """Map score to [0, 1] relative to neighbourhood range.

        Parameters
        ----------
        score : float
            Raw weighted score.

        Returns
        -------
        float
            Normalized score in [0, 1].
        """
        min_s, max_s = self._neighbourhood_score_range()
        normalized = (score - min_s) / (max_s - min_s)
        return max(0.0, min(1.0, normalized))

    # =========================================================================
    # BUNDLE SIMILARITY
    # =========================================================================

    def _bundle_similarity(self, a, b):
        """Compute fraction of practices that match between two bundles.

        Parameters
        ----------
        a, b : tuple
            Practice bundles to compare.

        Returns
        -------
        float
            Similarity in [0, 1]. 1.0 = identical, 0.0 = completely different.
        """
        matches = sum(x == y for x, y in zip(a, b))
        return matches / 3.0

    # =========================================================================
    # CROP SIMILARITY
    # =========================================================================

    def _crop_similarity(self, neighbour):
        """Check if neighbour grows the same dominant crop at similar scale.

        Fast O(n) comparison using only argmax (no sorting).
        Farmers with similar crop portfolios face similar conditions,
        making their experiences more relevant for social learning.

        Parameters
        ----------
        neighbour : Farmer
            Neighbour farmer to compare with.

        Returns
        -------
        float
            1.0: Same dominant crop with similar share (within 50% relative diff)
            0.5: Same dominant crop but different share
            0.0: Different dominant crops or neighbour doesn't grow it
        """
        # Get crop fractions via farmer's _get_from_earth (handles multi-year data)
        cft_self = self.agent._get_from_earth("cftfrac").values.flatten()
        cft_neighbour = neighbour._get_from_earth("cftfrac").values.flatten()

        # Find own dominant crop (single argmax - very fast)
        dominant_idx = np.argmax(cft_self)
        share_self = cft_self[dominant_idx]

        # Check if neighbour grows the same crop significantly
        share_neighbour = cft_neighbour[dominant_idx]

        if share_neighbour < 0.01:
            return 0.0  # Neighbour doesn't grow this crop

        # Check if shares are similar (within 50% relative difference)
        max_share = max(share_self, share_neighbour)
        relative_diff = abs(share_self - share_neighbour) / max_share

        if relative_diff < 0.5:
            return 1.0  # Same crop, similar share
        else:
            return 0.5  # Same crop, different share

    def _total_similarity(self, new_bundle, neighbour):
        """Combined bundle + crop similarity for social learning.

        Weighted average of practice bundle similarity and crop portfolio
        similarity. Allows tuning relative importance via config.

        Parameters
        ----------
        new_bundle : tuple
            Practice bundle being evaluated.
        neighbour : Farmer
            Neighbour farmer to compare with.

        Returns
        -------
        float
            Combined similarity in [0, 1].
        """
        bundle_sim = self._bundle_similarity(
            new_bundle, neighbour.behaviour._practice_bundle
        )
        crop_sim = self._crop_similarity(neighbour)

        # Weights from config (default: 60% bundle, 40% crop)
        w_bundle = self._get_aft_param("weight_bundle_similarity")
        w_crop = self._get_aft_param("weight_crop_similarity")

        return w_bundle * bundle_sim + w_crop * crop_sim

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Social Learning)
    # =========================================================================

    def _attitude_social_learning(self, new_bundle):
        """Compute attitude from neighbours using the proposed bundle.

        Social learning (Bandura 1977): farmers learn from observing
        neighbours who use similar practices.

        Combines:
        - Absolute performance comparison (ratio - 1): "Is neighbour doing better?"
        - Slope adjustment: "Is neighbour's trajectory sustainable?"

        The slope factor (via sigmoid) discounts neighbours who are declining,
        even if their absolute values are currently high (trap avoidance).

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        if not self.agent.neighbourhood:
            return 0.5  # Neutral without neighbours

        # Accumulate weighted comparisons
        weighted_yield = 0.0
        weighted_soil = 0.0
        weighted_moisture = 0.0
        total_weight = 0.0

        # Get confidence_years from config
        confidence_years = self._get_aft_param("confidence_years")

        # My current absolute values (avoid division by zero)
        my_yield = max(self.agent.cropyield, 1e-6)
        my_soil = max(self.agent.soilc, 1e-6)
        my_moisture = max(self.agent.root_moisture, 1e-6)

        for neighbour in self.agent.neighbourhood:
            # How similar is neighbour? (bundle + crop similarity)
            similarity = self._total_similarity(new_bundle, neighbour)

            if similarity == 0:
                continue  # Skip completely different neighbours

            # Confidence: more observations → more reliable information
            n_obs = neighbour.behaviour.current_state["n"]
            confidence = min(1.0, n_obs / confidence_years)

            # -----------------------------------------------------------------
            # Absolute comparisons (ratio - 1, like old tillage_farmer.py)
            # -----------------------------------------------------------------
            # Positive if neighbour is better, negative if worse
            yield_cmp = neighbour.cropyield / my_yield - 1
            soil_cmp = neighbour.soilc / my_soil - 1
            moisture_cmp = neighbour.root_moisture / my_moisture - 1

            # -----------------------------------------------------------------
            # Slope adjustment: discount declining neighbours
            # -----------------------------------------------------------------
            # sigmoid maps slope to (0, 1): <0.5 if declining, >0.5 if improving
            neighbour_slope = self._weighted_score(neighbour.behaviour.current_trend)
            slope_factor = sigmoid(neighbour_slope)

            # Adjust comparisons by slope factor
            yield_adj = yield_cmp * slope_factor
            soil_adj = soil_cmp * slope_factor
            moisture_adj = moisture_cmp * slope_factor

            # Weight = similarity × confidence
            weight = similarity * confidence

            # Accumulate
            weighted_yield += weight * yield_adj
            weighted_soil += weight * soil_adj
            weighted_moisture += weight * moisture_adj
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Neutral if no relevant neighbours

        # Normalize by total weight
        avg_yield = weighted_yield / total_weight
        avg_soil = weighted_soil / total_weight
        avg_moisture = weighted_moisture / total_weight

        # Weighted sum of comparisons (like old model)
        raw_score = (
            self.agent.weight_yield * avg_yield
            + self.agent.weight_soil * avg_soil
            + self.agent.weight_moisture * avg_moisture
        )

        # Final sigmoid (maps to (0, 1), consistent with old model)
        return sigmoid(raw_score)

    # =========================================================================
    # TPB COMPONENT: SOCIAL NORM
    # =========================================================================

    def _compute_social_norm(self, new_bundle):
        """Compute social norm based on neighbourhood practice distribution.

        Social norm reflects "what others are doing" (descriptive norm).
        Higher if:
        - Many neighbours use similar bundles and grow similar crops
        - The bundle is the most common in neighbourhood

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Social norm score in [0, 1].
        """
        if not self.agent.neighbourhood:
            return 0.5  # Neutral without neighbours

        # -----------------------------------------------------------------
        # Base norm: average total similarity to neighbours
        # -----------------------------------------------------------------
        # Uses combined bundle + crop similarity
        total_similarity = sum(
            self._total_similarity(new_bundle, n)
            for n in self.agent.neighbourhood
        )
        base_norm = total_similarity / len(self.agent.neighbourhood)

        # -----------------------------------------------------------------
        # Homogeneity bonus/penalty
        # -----------------------------------------------------------------
        # If neighbourhood is homogeneous, conformity pressure is stronger

        neighbour_bundles = [n.behaviour._practice_bundle for n in self.agent.neighbourhood]
        unique_bundles = len(set(neighbour_bundles))

        # Homogeneity: 1.0 if all same, lower if diverse
        homogeneity = 1.0 - (unique_bundles - 1) / max(len(neighbour_bundles), 1)

        # Find most common bundle
        most_common = max(set(neighbour_bundles), key=neighbour_bundles.count)

        # Boost if adopting majority bundle; penalize if adopting minority
        # Asymmetry reflects that social approval is stronger than disapproval
        # (Cialdini et al. 1990). Values are configurable per AFT.
        conformity_bonus = self._get_aft_param("conformity_bonus")
        conformity_penalty = self._get_aft_param("conformity_penalty")

        if new_bundle == most_common:
            boost = homogeneity * conformity_bonus
        else:
            boost = -homogeneity * conformity_penalty

        return max(0.0, min(1.0, base_norm + boost))

    # =========================================================================
    # COST CALCULATIONS
    # =========================================================================

    def _get_bundle_direct_cost(self, bundle):
        """Compute annual direct cost of a practice bundle.

        Parameters
        ----------
        bundle : tuple
            Practice bundle.

        Returns
        -------
        float
            Annual direct cost (scaled by farm size).
        """
        costs = self.agent.practice_costs
        total = 0.0

        # Tillage cost (if no-till, may have different cost structure)
        if bundle[0] == 1:
            total += costs.get("tillage", {}).get("direct", 0)

        # Cover crop cost (seeds, planting)
        if bundle[1] == 1:
            total += costs.get("cover_crop", {}).get("direct", 0)

        # Residue retention cost (foregone income from selling)
        if bundle[2] == 1:
            total += costs.get("residue_on_field", {}).get("direct", 0)

        return total * self.agent.farm_size

    def _total_transition_cost(self, old_bundle, new_bundle):
        """Compute one-time transition cost for changing practices.

        Parameters
        ----------
        old_bundle : tuple
            Current practice bundle.
        new_bundle : tuple
            Target practice bundle.

        Returns
        -------
        float
            Total transition cost (scaled by farm size).
        """
        costs = self.agent.practice_costs
        practices = ["tillage", "cover_crop", "residue_on_field"]

        # Sum transition costs for practices that change
        total = sum(
            costs.get(p, {}).get("transition", 0)
            for i, p in enumerate(practices)
            if old_bundle[i] != new_bundle[i]
        )

        return total * self.agent.farm_size

    # =========================================================================
    # AFFORDABILITY ADJUSTMENT
    # =========================================================================

    def _affordable_bundle(self, target_bundle):
        """Find affordable subset of target bundle.

        If farmer can't afford full target bundle, add changes cheapest-first
        to maximize what can be adopted within capital constraints.

        Parameters
        ----------
        target_bundle : tuple
            Desired practice bundle.

        Returns
        -------
        tuple
            Affordable bundle (may equal current if nothing affordable).
        """
        current = self._practice_bundle
        total_cost = self._total_transition_cost(current, target_bundle)

        # -----------------------------------------------------------------
        # Check if full target is affordable and reasonable
        # -----------------------------------------------------------------
        if self.agent.capital >= total_cost and self._is_reasonable_bundle(target_bundle):
            return target_bundle

        # -----------------------------------------------------------------
        # Build affordable subset: add changes cheapest-first
        # -----------------------------------------------------------------
        practices = ["tillage", "cover_crop", "residue_on_field"]

        # List changes needed: (index, new_value, cost)
        changes = [
            (i, target_bundle[i], self.agent.practice_costs.get(p, {}).get("transition", 0) * self.agent.farm_size)
            for i, p in enumerate(practices)
            if current[i] != target_bundle[i]
        ]

        # Sort by cost (cheapest first)
        changes.sort(key=lambda x: x[2])

        # Greedily add affordable changes
        result = list(current)
        remaining_capital = self.agent.capital

        for idx, new_val, cost in changes:
            if remaining_capital >= cost:
                # Check if adding this change keeps bundle reasonable
                candidate = list(result)
                candidate[idx] = new_val

                if self._is_reasonable_bundle(tuple(candidate)):
                    result[idx] = new_val
                    remaining_capital -= cost

        return tuple(result)

    # =========================================================================
    # TPB COMPONENT: PERCEIVED BEHAVIORAL CONTROL (PBC)
    # =========================================================================

    def _compute_risk_factor(self):
        """Compute risk factor from AFT type.

        Risk aversion (Chavas & Holt 1996): farmers weight potential losses
        more heavily than equivalent gains. Higher risk factor means more
        cautious behavior and lower PBC.

        Currently uses only AFT base risk aversion. Traditionalists are more
        risk-averse than pioneers.

        Returns
        -------
        float
            Risk factor in [0, 1]. Higher = more risk-averse = lower PBC.

        References
        ----------
        Chavas, J.P. & Holt, M.T. (1996). Economic behavior under uncertainty.

        TODO: Optional extension - add capital volatility component
        --------------------------------------------------------------
        Could combine base risk with capital volatility (CV over recent years):

            capital_history = self.agent.capital_history  # needs tracking
            if len(capital_history) >= 3:
                cv = np.std(capital_history) / max(np.mean(capital_history), 1e-6)
                volatility_risk = min(cv, 1.0)
            else:
                volatility_risk = 0.0

            weight_base = self._get_aft_param("weight_risk_base")
            weight_volatility = self._get_aft_param("weight_risk_volatility")
            risk_factor = weight_base * base_risk + weight_volatility * volatility_risk

        This would require:
        - Adding capital_history tracking in ca_farmer.py
        - Adding weight_risk_base, weight_risk_volatility to config.yaml
        """
        return self._get_aft_param("risk_aversion")

    def _pbc_for_bundle(self, new_bundle):
        """Compute Perceived Behavioral Control for a bundle.

        PBC reflects "can I actually do this?" - lower when costs are
        high relative to available capital, and when farmer is risk-averse.

        Formula: PBC = pbc_base × cost_factor × (1 - risk_factor)

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            PBC score in [0, pbc_base].
        """
        # -----------------------------------------------------------------
        # Calculate cost impact of switching
        # -----------------------------------------------------------------

        # One-time transition cost
        transition_cost = self._total_transition_cost(self._practice_bundle, new_bundle)

        # Change in annual direct costs
        current_direct = self._get_bundle_direct_cost(self._practice_bundle)
        new_direct = self._get_bundle_direct_cost(new_bundle)
        direct_cost_increase = max(0, new_direct - current_direct)

        # Total cost impact
        cost_impact = transition_cost + direct_cost_increase

        # -----------------------------------------------------------------
        # Calculate disposable capital (above minimum threshold)
        # -----------------------------------------------------------------
        disposable = max(self.agent.capital - self.agent.min_capital, 1e-6)

        # -----------------------------------------------------------------
        # PBC decreases as cost approaches disposable capital
        # -----------------------------------------------------------------
        cost_factor = 1.0 / (1.0 + cost_impact / disposable)

        # -----------------------------------------------------------------
        # Risk aversion reduces PBC (Chavas & Holt 1996)
        # -----------------------------------------------------------------
        # Risk-averse farmers are less confident in their ability to adopt
        # new practices, especially when capital has been volatile
        risk_factor = self._compute_risk_factor()

        return self.agent.pbc_base * cost_factor * (1.0 - risk_factor)

    # =========================================================================
    # COMPUTE FULL TPB SCORE
    # =========================================================================

    def _compute_tpb_for_bundle(self, new_bundle):
        """Compute all TPB components and overall intention for a bundle.

        TPB formula (multiplicative):
            TPB = (w_att × Attitude + w_norm × SocialNorm) × PBC

        This means PBC acts as a gate: low PBC blocks adoption regardless
        of positive attitude/norms.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.
        """
        # -----------------------------------------------------------------
        # Attitude: own experience + social learning
        # -----------------------------------------------------------------
        #
        # att_own: "Am I doing poorly with my current practices?"
        # Based on current_trend (regression over all observations).
        # Declining performance → high attitude → more willing to switch
        # Improving performance → low attitude → less willing to switch
        #
        # Structure matches old tillage_farmer.py:
        # - Weighted sum of individual trend components
        # - Negate (so decline → positive)
        # - Final sigmoid

        trend = self.current_trend
        raw_own = (
            self.agent.weight_yield * (-trend["yield"])
            + self.agent.weight_soil * (-trend["soil"])
            + self.agent.weight_moisture * (-trend["moisture"])
        )
        att_own = sigmoid(raw_own)

        # Social learning component: how are neighbours with proposed bundle
        # doing compared to me?
        att_social = self._attitude_social_learning(new_bundle)

        # -----------------------------------------------------------------
        # Weighted combination (same approach as tillage_farmer.py)
        # -----------------------------------------------------------------
        self._attitude = (
            self.agent.weight_own_land * att_own
            + self.agent.weight_social_learning * att_social
        )

        # -----------------------------------------------------------------
        # Social Norm: what neighbours are doing
        # -----------------------------------------------------------------
        self._social_norm = self._compute_social_norm(new_bundle)
        # -----------------------------------------------------------------
        # PBC: can I afford this?
        # -----------------------------------------------------------------
        self._pbc = self._pbc_for_bundle(new_bundle)

        # -----------------------------------------------------------------
        # TPB Intention: combine components
        # -----------------------------------------------------------------
        # Multiplicative formula: PBC gates the attitude/norm contribution
        self._tpb = (
            self.agent.weight_attitude * self._attitude
            + self.agent.weight_norm * self._social_norm
        ) * self._pbc
