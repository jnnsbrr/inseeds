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

from inseeds.components.farming.farmer import sigmoid, NON_CROPS

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

# Transition blocker codes - indicates WHY a proposed bundle was not adopted
# These follow the decision flow order (first blocker hit is the primary reason)
BLOCKER_NONE = 0                    # No blocker - transition happened or maintaining current
BLOCKER_MIN_OBS_YEARS = 1           # Not enough observation years yet
BLOCKER_FALLBACK_TRIGGERED = 2      # Reverting to previous bundle (adaptive management)
BLOCKER_NO_TARGET = 3               # No better neighbour + exploration didn't trigger
BLOCKER_TARGET_SAME = 4             # Target bundle same as current (already optimal)
BLOCKER_TARGET_UNAFFORDABLE = 5     # Target reduced to current due to cost
BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND = 6         # TPB low - own land attitude is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL = 7     # TPB low - LOCAL social learning is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY = 14  # TPB low - COUNTRY social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL = 8         # TPB low - LOCAL social norm is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY = 15      # TPB low - COUNTRY social norm is limiting
BLOCKER_TPB_LOW_PBC = 9             # TPB below threshold - PBC (cost affordability) is limiting
BLOCKER_TRANSITION_UNAFFORDABLE = 10 # Can't afford transition cost
BLOCKER_CAPITAL_SURVIVAL = 11       # Capital below survival threshold
BLOCKER_CONTROL_RUN = 12            # Control run - no CA dynamics
BLOCKER_EVALUATION_TIME = 13        # Not yet time to re-evaluate (commitment period)

BLOCKER_NAMES = {
    BLOCKER_NONE: "none",
    BLOCKER_MIN_OBS_YEARS: "min_obs_years",
    BLOCKER_FALLBACK_TRIGGERED: "fallback_triggered",
    BLOCKER_NO_TARGET: "no_target",
    BLOCKER_TARGET_SAME: "target_same",
    BLOCKER_TARGET_UNAFFORDABLE: "target_unaffordable",
    BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND: "tpb_low_attitude_own_land",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL: "tpb_low_attitude_social_local",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY: "tpb_low_attitude_social_country",
    BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL: "tpb_low_social_norm_local",
    BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY: "tpb_low_social_norm_country",
    BLOCKER_TPB_LOW_PBC: "tpb_low_pbc",
    BLOCKER_TRANSITION_UNAFFORDABLE: "transition_unaffordable",
    BLOCKER_CAPITAL_SURVIVAL: "capital_survival",
    BLOCKER_CONTROL_RUN: "control_run",
    BLOCKER_EVALUATION_TIME: "evaluation_time",
}

# Transition driver codes - indicates WHY a transition succeeded (mirror of blocker)
# These identify which pathway and TPB component (with local/country distinction) that
# contributed most to enabling the transition. 3 pathways × 6 components = 18 codes.
DRIVER_NONE = 0                              # No transition - maintaining current bundle
DRIVER_FALLBACK = 1                          # Reverted to previous bundle (adaptive management)

# LOCAL pathway - learned from better-performing LOCAL neighbor
DRIVER_LOCAL_ATTITUDE_OWN_LAND = 2          # TPB passed, attitude_own_land was strongest
DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL = 3      # TPB passed, local social learning was strongest
DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY = 4    # TPB passed, country social learning was strongest
DRIVER_LOCAL_SOCIAL_NORM_LOCAL = 5          # TPB passed, local social norm was strongest
DRIVER_LOCAL_SOCIAL_NORM_COUNTRY = 6        # TPB passed, country social norm was strongest
DRIVER_LOCAL_PBC = 7                        # TPB passed, PBC (cost affordability) was strongest

# COUNTRY pathway - inspired by country-level best performer
DRIVER_COUNTRY_ATTITUDE_OWN_LAND = 8         # TPB passed, attitude_own_land was strongest
DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL = 9     # TPB passed, local social learning was strongest
DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY = 10  # TPB passed, country social learning was strongest
DRIVER_COUNTRY_SOCIAL_NORM_LOCAL = 11        # TPB passed, local social norm was strongest
DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY = 12      # TPB passed, country social norm was strongest
DRIVER_COUNTRY_PBC = 13                      # TPB passed, PBC (cost affordability) was strongest

# EXPLORATION pathway - discovered via random exploration
DRIVER_EXPLORATION_ATTITUDE_OWN_LAND = 14    # TPB passed, attitude_own_land was strongest
DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL = 15  # TPB passed, local social learning was strongest
DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY = 16  # TPB passed, country social learning was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL = 17    # TPB passed, local social norm was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY = 18  # TPB passed, country social norm was strongest
DRIVER_EXPLORATION_PBC = 19                  # TPB passed, PBC (cost affordability) was strongest

DRIVER_NAMES = {
    DRIVER_NONE: "none",
    DRIVER_FALLBACK: "fallback",
    # Social pathway (local neighbor inspiration)
    DRIVER_LOCAL_ATTITUDE_OWN_LAND: "local_attitude_own_land",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL: "local_attitude_social_local",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY: "local_attitude_social_country",
    DRIVER_LOCAL_SOCIAL_NORM_LOCAL: "local_social_norm_local",
    DRIVER_LOCAL_SOCIAL_NORM_COUNTRY: "local_social_norm_country",
    DRIVER_LOCAL_PBC: "local_pbc",
    # Country pathway (country-level inspiration)
    DRIVER_COUNTRY_ATTITUDE_OWN_LAND: "country_attitude_own_land",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL: "country_attitude_social_local",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY: "country_attitude_social_country",
    DRIVER_COUNTRY_SOCIAL_NORM_LOCAL: "country_social_norm_local",
    DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY: "country_social_norm_country",
    DRIVER_COUNTRY_PBC: "country_pbc",
    # Exploration pathway (random exploration)
    DRIVER_EXPLORATION_ATTITUDE_OWN_LAND: "exploration_attitude_own_land",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL: "exploration_attitude_social_local",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY: "exploration_attitude_social_country",
    DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL: "exploration_social_norm_local",
    DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY: "exploration_social_norm_country",
    DRIVER_EXPLORATION_PBC: "exploration_pbc",
}


# =============================================================================
# ABSTRACT DECISION MODEL BASE CLASS
# =============================================================================

class DecisionModel(ABC):
    """Abstract base for farmer decision models.

    Tracks:
    - Current practice bundle (tillage, cover_crop, residue)
    - Memory of past outcomes for each bundle tried
    - Current state snapshot for computing trends since last transition

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
        #   residue: 1 if litter cover >= CA threshold (30%), 0 otherwise

        # Get CA cover threshold from config
        ca_threshold = agent.model.config.coupled_config.practice_dimensions.residue.ca_cover_threshold  # noqa: E501

        self._practice_bundle = (
            int(agent.tillage),
            1 if agent.cover_crop > 0 else 0,
            1 if agent.litter_cover >= ca_threshold else 0,
        )

        # Proposed bundle for potential transition (set by update())
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
            "t_start": t_start,           # Year of last practice transition
            "baseline_score": 0.0,        # Weighted score at transition (for fallback)
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
        to imitate, not stale data from the last transition.
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

    @property
    def transition_blocker(self):
        """Primary reason why proposed bundle was not adopted (numeric code).
        
        See BLOCKER_* constants for codes. 0 = no blocker (transition happened
        or no transition needed).
        """
        return getattr(self, '_transition_blocker', BLOCKER_NONE)

    @property
    def transition_blocker_name(self):
        """Human-readable name of transition blocker."""
        return BLOCKER_NAMES.get(self.transition_blocker, "unknown")

    @property
    def transition_driver(self):
        """Primary reason why a transition succeeded (numeric code).
        
        See DRIVER_* constants for codes. 0 = no transition (maintaining current).
        This is the mirror of transition_blocker - identifies which pathway and
        TPB component enabled the transition when it succeeds.
        """
        return getattr(self, '_transition_driver', DRIVER_NONE)

    @property
    def transition_driver_name(self):
        """Human-readable name of transition driver."""
        return DRIVER_NAMES.get(self.transition_driver, "unknown")

    # -------------------------------------------------------------------------
    # Trend computation
    # -------------------------------------------------------------------------

    @property
    def current_trend(self):
        """Annual change in soil C, moisture, and yield since last transition.

        Computes trends using linear regression over all observations since
        the last practice transition, not just start and end points. This provides
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

    def record_transition(self, new_bundle):
        """Store outcome of current bundle and prepare for transition to new_bundle.

        Called when farmer commits to transitioning practices. Records the
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

        # Transition to new bundle
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
    def should_transition(self):
        """Return True if farmer should transition to proposed bundle."""
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
        self._attitude = 0.0     # Attitude toward behavior (combined)
        self._attitude_own_land = 0.0    # Attitude from own land performance
        self._attitude_social_learning = 0.0  # Attitude from social learning
        self._social_norm = 0.0  # Subjective norm
        self._pbc = 0.0          # Perceived behavioral control

        # Evaluation time: farmers don't reconsider every year
        # Randomize initial evaluation time to desynchronize farmers
        # (avoids artificial waves of simultaneous evaluation)
        interval = self.agent.model.config.coupled_config.tpb_thresholds.evaluation_interval
        self._years_until_evaluation = np.random.randint(0, interval + 1)

    # -------------------------------------------------------------------------
    # Properties for external access to TPB components
    # -------------------------------------------------------------------------

    @property
    def tpb(self):
        """Overall TPB intention score."""
        return self._tpb

    @property
    def attitude(self):
        """Attitude component of TPB (combined)."""
        return self._attitude

    @property
    def attitude_own_land(self):
        """Attitude from own land performance."""
        return self._attitude_own_land

    @property
    def attitude_social_learning(self):
        """Attitude from social learning."""
        return self._attitude_social_learning

    @property
    def social_norm(self):
        """Social norm component of TPB."""
        return self._social_norm

    @property
    def pbc(self):
        """Perceived behavioral control component of TPB."""
        return self._pbc

    # =========================================================================
    # EVALUATION TIMING
    # =========================================================================

    def should_evaluate(self) -> bool:
        """Check if farmer should evaluate practice transition this year.

        Farmers don't reconsider practices every year. They evaluate when
        the time period has ended (randomized around evaluation_interval).

        This saves computation and is more realistic - farmers commit to
        observing results before reconsidering.

        Returns
        -------
        bool
            True if farmer should run TPB evaluation this year.
        """
        return self._years_until_evaluation <= 0

    def reset_evaluation_time(self):
        """Reset evaluation time after a transition decision (transition or stay).

        Uses fixed evaluation_interval from config. Randomization only happens
        at initialization to desynchronize farmers - after that, each farmer
        evaluates at consistent intervals (like real-world planning horizons).

        Called after TPB evaluation completes, regardless of whether transition
        happened.
        """
        self._years_until_evaluation = (
            self.agent.model.config.coupled_config.tpb_thresholds.evaluation_interval  # noqa: E501
        )

    def decrement_evaluation_time(self):
        """Decrement the evaluation time counter by one year.

        Called each year when farmer doesn't evaluate.
        """
        if self._years_until_evaluation > 0:
            self._years_until_evaluation -= 1

    # =========================================================================
    # MAIN UPDATE LOGIC
    # =========================================================================

    def _reevaluate_residue_status(self):
        """Re-evaluate residue component of bundle based on actual litter cover.

        Farmers may organically cross the CA residue threshold (30% soil cover)
        through their management practices. This method "certifies" or
        "de-certifies" their CA residue status based on actual outcomes.

        - If litter_cover >= threshold AND bundle residue = 0 → upgrade to 1
        - If litter_cover < threshold AND bundle residue = 1 → downgrade to 0

        This ensures the bundle reflects actual field conditions, not just
        intended practices.
        """
        ca_threshold = self.agent.model.config.coupled_config.practice_dimensions.residue.ca_cover_threshold  # noqa: E501
        current_residue = self._practice_bundle[2]
        litter_cover = self.agent.litter_cover

        # Check if status should change
        if litter_cover >= ca_threshold and current_residue == 0:
            # Farmer achieved CA residue threshold - certify!
            new_bundle = (
                self._practice_bundle[0],
                self._practice_bundle[1],
                1,  # Upgrade residue status
            )
            self._practice_bundle = new_bundle
        elif litter_cover < ca_threshold and current_residue == 1:
            # Farmer fell below CA threshold - de-certify
            new_bundle = (
                self._practice_bundle[0],
                self._practice_bundle[1],
                0,  # Downgrade residue status
            )
            self._practice_bundle = new_bundle

    def update(self):
        """Compute proposed bundle and TPB scores for this timestep.

        Decision flow:
        0. Re-evaluate residue status based on actual litter cover
        1. Add current observation to regression accumulators
        2. Update bundle_memory with current trends (for neighbour visibility)
        3. Decay old memories (bounded rationality)
        4. Check if minimum observation period has passed
        5. Check fallback condition (sustained decline → revert)
        6. Find best-performing neighbour's bundle OR explore randomly
        7. Adjust target bundle for affordability
        8. Compute TPB scores for the proposed bundle
        
        Sets _transition_blocker to indicate why transition didn't happen (if applicable).
        Sets _transition_driver to indicate why transition succeeded (if applicable).
        """
        # Reset blocker, driver, and pathway at start of each update
        self._transition_blocker = BLOCKER_NONE
        self._transition_driver = DRIVER_NONE
        self._target_pathway = None

        # -----------------------------------------------------------------
        # Step 0: Re-evaluate residue status based on actual litter cover
        # -----------------------------------------------------------------
        # Farmers may cross CA threshold organically - certify/de-certify
        self._reevaluate_residue_status()

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
        # Step 4: Require minimum observation years before transitioning
        # -----------------------------------------------------------------
        # Avoids noisy decisions based on single-year fluctuations
        # Typical value: 3 years (allows trends to stabilize)
        # With historic data initialization, farmers start with enough history

        n_obs = self.current_state["n"]
        min_obs = self._get_aft_param("min_observation_years")

        if n_obs < min_obs:
            self._tpb = 0.0
            self._proposed_bundle = None
            self._transition_blocker = BLOCKER_MIN_OBS_YEARS
            return

        # -----------------------------------------------------------------
        # Step 5: Check fallback condition
        # -----------------------------------------------------------------
        # If performance has declined for FALLBACK_YEARS consecutive years,
        # propose reverting to the previous bundle (adaptive management)

        if self._check_fallback():
            # Fallback sets _proposed_bundle and _tpb internally
            # Blocker will be set by should_transition() if TPB too low
            self._transition_blocker = BLOCKER_FALLBACK_TRIGGERED
            self._target_pathway = "fallback"  # Track pathway for transition_driver
            return

        # -----------------------------------------------------------------
        # Step 6: Find target bundle (neighbour imitation or exploration)
        # -----------------------------------------------------------------

        # First, try to imitate best-performing neighbour (local)
        target_bundle = self._most_promising_bundle()
        self._target_pathway = "social"  # Track pathway for transition_driver

        # If no better local neighbour, try country-level inspiration
        if target_bundle is None:
            target_bundle = self._most_promising_bundle_country()
            self._target_pathway = "country"

        # If no country inspiration, maybe explore randomly
        if target_bundle is None:
            target_bundle = self._maybe_explore_bundle()
            self._target_pathway = "exploration"

        # No change proposed
        if target_bundle is None:
            self._tpb = 0.0
            self._proposed_bundle = None
            self._transition_blocker = BLOCKER_NO_TARGET
            return

        if target_bundle == self._practice_bundle:
            self._tpb = 0.0
            self._proposed_bundle = None
            self._transition_blocker = BLOCKER_TARGET_SAME
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
            self._transition_blocker = BLOCKER_TARGET_UNAFFORDABLE
            return

        # -----------------------------------------------------------------
        # Step 8: Compute TPB scores for proposed bundle
        # -----------------------------------------------------------------
        self._proposed_bundle = affordable_bundle
        self._compute_tpb_for_bundle(affordable_bundle)
        
        # Blocker will be set to BLOCKER_TPB_BELOW_THRESHOLD by CAFarmer
        # if should_transition() returns False

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
        # Compare current performance to baseline at transition time
        # -----------------------------------------------------------------
        # baseline_score captures the trend score at the time of transition.
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
    # TRANSITION DECISION
    # =========================================================================

    def should_transition(self):
        """Determine if farmer should transition to proposed bundle.

        Compares TPB intention score to threshold. Higher threshold
        when reverting (to avoid oscillation).

        Thresholds are non-AFT-specific (from config.tpb_thresholds).
        Behavioral differences between AFTs come from weights and PBC.

        Returns
        -------
        bool
            True if TPB exceeds threshold and transition should occur.
        """
        # Get thresholds from config (non-AFT-specific)
        tpb_cfg = self.agent.model.config.coupled_config.tpb_thresholds
        transition_threshold = tpb_cfg.transition_threshold
        revert_threshold = tpb_cfg.revert_threshold

        # Higher threshold for reverting (avoid flip-flopping)
        if self._proposed_bundle == self._previous_bundle:
            threshold = revert_threshold
        else:
            threshold = transition_threshold

        return self._tpb > threshold

    def set_tpb_transition_blocker(self):
        """Set transition_blocker to indicate which TPB component is most limiting.

        Uses a simple recursive approach:
        1. First identify which main component (attitude, social_norm, pbc) is lowest
        2. Then drill down into that component's sub-parts to identify the specific blocker

        This is simpler and more interpretable than weighted drag calculations.
        """
        if self._proposed_bundle is None:
            return

        # Level 1: Which main component is the blocker?
        main_components = {
            'attitude': self._attitude,
            'social_norm': self._social_norm,
            'pbc': self._pbc,
        }
        main_blocker = min(main_components, key=main_components.get)

        # Level 2: Drill down into sub-components
        if main_blocker == 'attitude':
            # Compare own_land vs social_learning
            if self._attitude_own_land <= self._attitude_social_learning:
                self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND
            else:
                # Compare local vs country social learning
                if self._attitude_social_learning_local <= self._attitude_social_learning_country:
                    self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL
                else:
                    self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY

        elif main_blocker == 'social_norm':
            # Compare local vs country
            if self._social_norm_local <= self._social_norm_country:
                self._transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL
            else:
                self._transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY

        else:  # pbc
            # PBC is driven by cost affordability (pbc_base captures AFT risk differences)
            self._transition_blocker = BLOCKER_TPB_LOW_PBC

    def set_tpb_component_driver(self, pathway: str):
        """Set transition_driver to indicate which TPB component enabled the transition.

        Uses a simple recursive approach (mirror of set_tpb_transition_blocker):
        1. First identify which main component (attitude, social_norm, pbc) is highest
        2. Then drill down into that component's sub-parts to identify the specific driver

        Parameters
        ----------
        pathway : str
            One of "social" (learned from local neighbor), "country" (inspired
            by country-level data), or "exploration" (random exploration).
        """
        if self._proposed_bundle is None:
            return

        # Level 1: Which main component is the strongest driver?
        main_components = {
            'attitude': self._attitude,
            'social_norm': self._social_norm,
            'pbc': self._pbc,
        }
        main_driver = max(main_components, key=main_components.get)

        # Level 2: Drill down into sub-components
        if main_driver == 'attitude':
            # Compare own_land vs social_learning
            if self._attitude_own_land >= self._attitude_social_learning:
                sub_driver = 'attitude_own_land'
            else:
                # Compare local vs country social learning
                if self._attitude_social_learning_local >= self._attitude_social_learning_country:
                    sub_driver = 'attitude_social_local'
                else:
                    sub_driver = 'attitude_social_country'

        elif main_driver == 'social_norm':
            # Compare local vs country
            if self._social_norm_local >= self._social_norm_country:
                sub_driver = 'social_norm_local'
            else:
                sub_driver = 'social_norm_country'

        else:  # pbc
            # PBC is driven by cost affordability (pbc_base captures AFT risk differences)
            sub_driver = 'pbc'

        # Map based on pathway and sub-component
        driver_maps = {
            "social": {
                'attitude_own_land': DRIVER_LOCAL_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY,
                'social_norm_local': DRIVER_LOCAL_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_LOCAL_SOCIAL_NORM_COUNTRY,
                'pbc': DRIVER_LOCAL_PBC,
            },
            "country": {
                'attitude_own_land': DRIVER_COUNTRY_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY,
                'social_norm_local': DRIVER_COUNTRY_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY,
                'pbc': DRIVER_COUNTRY_PBC,
            },
            "exploration": {
                'attitude_own_land': DRIVER_EXPLORATION_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY,
                'social_norm_local': DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY,
                'pbc': DRIVER_EXPLORATION_PBC,
            },
        }

        driver_map = driver_maps.get(pathway, driver_maps["exploration"])
        self._transition_driver = driver_map[sub_driver]

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
        # Residue retention: 0 = baseline, 1 = retain to reach CA threshold
        # -----------------------------------------------------------------
        # CA threshold from config (default 30% soil cover per FAO definition)
        # If already at/above threshold, maintain current level
        # If below, increase retention to reach threshold (if affordable)
        ca_threshold = self.agent.model.config.coupled_config.practice_dimensions.residue.ca_cover_threshold  # noqa: E501

        if bundle[2] == 1:
            # Want CA residue retention
            if self.agent.litter_cover < ca_threshold:
                # Below threshold - need to increase retention if affordable
                opp_cost = self.agent.residue_opportunity_cost
                can_afford = opp_cost <= 0 or self.agent.capital >= opp_cost
                if can_afford:
                    self.agent.residue_on_field = 1.0
            # If already at threshold, keep current residue_on_field unchanged
        else:
            # Not pursuing CA residue retention - use baseline
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

    def _most_promising_bundle_country(self):
        """Find best-performing bundle at COUNTRY level (if better than current).

        Non-local social learning: when no local neighbour is better, farmers
        may look to successful practices used elsewhere in their country.

        Uses cached country statistics for O(1) lookup.

        Returns
        -------
        tuple or None
            Best country-level bundle, or None if no bundle performs better
            than current practice or if country data is unavailable.
        """
        # Access country cache
        country = self.agent.cell.country
        cache = getattr(country, "_country_stats_cache", {})

        if not cache:
            return None

        bundle_performance = cache.get("bundle_performance", {})
        if not bundle_performance:
            return None

        # Get own weighted score from current trend
        own_score = self._weighted_score(self.current_trend)

        # Find best performing bundle at country level
        best_bundle = None
        best_score = own_score  # Must beat our current performance

        for bundle, perf in bundle_performance.items():
            # Skip our own bundle
            if bundle == self._practice_bundle:
                continue

            # Calculate weighted score from country averages
            country_score = (
                self.agent.weight_yield * perf.get("avg_yield_slope", 0.0)
                + self.agent.weight_soil * perf.get("avg_soil_slope", 0.0)
                + self.agent.weight_moisture * perf.get("avg_moisture_slope", 0.0)
            )

            if country_score > best_score:
                best_score = country_score
                best_bundle = bundle

        return best_bundle

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
        # Exclude managed grassland - it's not a crop for similarity comparison
        cft_self = self.agent._get_from_earth(
            "cftfrac", drop_band=NON_CROPS
        ).values.flatten()
        cft_neighbour = neighbour._get_from_earth(
            "cftfrac", drop_band=NON_CROPS
        ).values.flatten()

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
    # TPB COMPONENT: ATTITUDE (Own Land)
    # =========================================================================

    def _compute_attitude_own_land(self):
        """Compute attitude from own land performance trends.

        "Am I doing poorly with my current practices?"
        Based on current_trend (regression over all observations).
        Declining performance → high attitude → more willing to transition
        Improving performance → low attitude → less willing to transition

        Structure matches old tillage_farmer.py:
        - Weighted sum of individual trend components
        - Negate (so decline → positive)
        - Final sigmoid

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        trend = self.current_trend
        raw_own = (
            self.agent.weight_yield * (-trend["yield"])
            + self.agent.weight_soil * (-trend["soil"])
            + self.agent.weight_moisture * (-trend["moisture"])
        )
        return sigmoid(raw_own)

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Social Learning)
    # =========================================================================

    def _compute_attitude_social_learning_local(self, new_bundle):
        """Compute attitude from LOCAL neighbours using the proposed bundle.

        Social learning (Bandura 1977): farmers learn from observing
        neighbours who use similar practices.

        Combines:
        - Absolute performance comparison (ratio - 1): "Is neighbour doing better?"
        - Slope adjustment: "Is neighbour's trajectory sustainable?"

        The slope factor (via sigmoid) discounts neighbours who are declining,
        even if their absolute values are currently high (trap avoidance).

        This is the LOCAL component - uses direct neighbour comparisons with
        full similarity weighting. See _compute_attitude_social_learning_country()
        for the country-level component.

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

    def _compute_attitude_social_learning_country(self, new_bundle):
        """Compute attitude from COUNTRY-LEVEL bundle performance.

        Uses cached country statistics to compare own performance against
        average performance of farmers using the proposed bundle across
        the entire country.

        No similarity weighting at country level - uses simpler bundle-based
        grouping for computational efficiency (O(1) vs O(n²)).

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        # Access country cache
        country = self.agent.cell.country
        cache = getattr(country, "_country_stats_cache", {})

        if not cache:
            return 0.5  # Neutral if no country data

        bundle_performance = cache.get("bundle_performance", {})

        # Get average performance for the proposed bundle
        bundle_perf = bundle_performance.get(new_bundle)
        if bundle_perf is None or bundle_perf.get("n_farmers", 0) == 0:
            return 0.5  # Neutral if no data for this bundle

        # My current absolute values (avoid division by zero)
        my_yield = max(self.agent.cropyield, 1e-6)
        my_soil = max(self.agent.soilc, 1e-6)
        my_moisture = max(self.agent.root_moisture, 1e-6)

        # Compare my performance to country average for this bundle
        avg_yield = bundle_perf["avg_yield"]
        avg_soil = bundle_perf["avg_soil"]
        avg_moisture = bundle_perf["avg_moisture"]

        # Absolute comparisons (ratio - 1)
        yield_cmp = avg_yield / my_yield - 1 if my_yield > 0 else 0.0
        soil_cmp = avg_soil / my_soil - 1 if my_soil > 0 else 0.0
        moisture_cmp = avg_moisture / my_moisture - 1 if my_moisture > 0 else 0.0

        # Slope adjustment using country-level average slopes
        avg_yield_slope = bundle_perf.get("avg_yield_slope", 0.0)
        avg_soil_slope = bundle_perf.get("avg_soil_slope", 0.0)
        avg_moisture_slope = bundle_perf.get("avg_moisture_slope", 0.0)

        # Weighted slope for the bundle's average trajectory
        weighted_slope = (
            self.agent.weight_yield * avg_yield_slope
            + self.agent.weight_soil * avg_soil_slope
            + self.agent.weight_moisture * avg_moisture_slope
        )
        slope_factor = sigmoid(weighted_slope)

        # Adjust comparisons by slope factor
        yield_adj = yield_cmp * slope_factor
        soil_adj = soil_cmp * slope_factor
        moisture_adj = moisture_cmp * slope_factor

        # Weighted sum of comparisons
        raw_score = (
            self.agent.weight_yield * yield_adj
            + self.agent.weight_soil * soil_adj
            + self.agent.weight_moisture * moisture_adj
        )

        return sigmoid(raw_score)

    # =========================================================================
    # TPB COMPONENT: SOCIAL NORM
    # =========================================================================

    def _compute_social_norm_local(self, new_bundle):
        """Compute social norm based on LOCAL neighbourhood practice distribution.

        Social norm reflects "what others are doing" (descriptive norm).
        Uses similarity-weighted average: higher if neighbours use similar
        bundles and grow similar crops.

        Follows the simple approach from tillage_farmer.py: fraction of
        neighbours using the practice → sigmoid transformation.

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

        # Average similarity to neighbours for this bundle
        # High similarity = neighbours use similar bundles and crops
        total_similarity = sum(
            self._total_similarity(new_bundle, n)
            for n in self.agent.neighbourhood
        )
        avg_similarity = total_similarity / len(self.agent.neighbourhood)

        # Shifted sigmoid: adoption threshold acts as the neutrality point
        # (Granovetter 1978 heterogeneous-threshold diffusion model).
        # Below threshold -> drag, above -> boost, smoothly transitioning.
        # Threshold is AFT-specific: pioneers feel "normed" at lower adoption,
        # traditionalists require broader local uptake to feel normative.
        threshold = self._get_aft_param("threshold_social_norm_local")
        return sigmoid(avg_similarity - threshold)

    def _compute_social_norm_country(self, new_bundle):
        """Compute social norm based on COUNTRY-LEVEL practice distribution.

        Uses cached country statistics for O(1) lookup. This reflects
        "what farmers in my country are doing" as a broader social influence.

        Follows the simple approach from tillage_farmer.py: fraction of
        farmers using the practice → sigmoid transformation.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Social norm score in [0, 1].
        """
        # Access country cache
        country = self.agent.cell.country
        cache = getattr(country, "_country_stats_cache", {})

        if not cache or cache.get("total_farmers", 0) < 2:
            return 0.5  # Neutral if no country data or only self

        bundle_counts = cache.get("bundle_counts", {})
        total_farmers = cache.get("total_farmers", 1)

        # Fraction of farmers using this bundle at country level
        bundle_count = bundle_counts.get(new_bundle, 0)
        bundle_fraction = bundle_count / total_farmers

        # Shifted sigmoid: threshold = country-adoption level at which this
        # bundle feels normative (Granovetter 1978). Thresholds differ per
        # AFT (pioneers adopt the country signal earlier) and are typically
        # higher than the local threshold, because country adoption is more
        # abstract/statistical than direct observation of neighbours.
        threshold = self._get_aft_param("threshold_social_norm_country")
        return sigmoid(bundle_fraction - threshold)

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

    def _pbc_for_bundle(self, new_bundle):
        """Compute Perceived Behavioral Control for a bundle.

        PBC reflects "can I actually do this?" - lower when costs are
        high relative to available capital.

        Formula: PBC = pbc_base × cost_factor

        Where:
        - pbc_base: AFT-specific baseline (pioneers higher, traditionalists lower)
        - cost_factor: decreases as transition cost approaches disposable capital

        AFT differences in risk aversion are captured via pbc_base, not as a
        separate multiplicative factor (avoids double-counting).

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
        # Calculate cost impact of transitioning
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

        return self.agent.pbc_base * cost_factor

    # =========================================================================
    # COMPUTE FULL TPB SCORE
    # =========================================================================

    def _compute_tpb_for_bundle(self, new_bundle):
        """Compute all TPB components and overall intention for a bundle.

        TPB formula (multiplicative):
            TPB = (w_att × Attitude + w_norm × SocialNorm) × PBC

        This means PBC acts as a gate: low PBC blocks adoption regardless
        of positive attitude/norms.

        Components are computed at both local (neighbour) and country levels,
        then combined with configurable weights. Local effects typically
        dominate (neighbours have more influence), but country-level trends
        provide broader social signals.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.
        """
        # -----------------------------------------------------------------
        # Attitude: own experience + social learning (local + country)
        # -----------------------------------------------------------------
        # Own land attitude (same for local/country - it's your own observation)
        self._attitude_own_land = self._compute_attitude_own_land()

        # Social learning: local (neighbours) and country-level
        self._attitude_social_learning_local = (
            self._compute_attitude_social_learning_local(new_bundle)
        )
        self._attitude_social_learning_country = (
            self._compute_attitude_social_learning_country(new_bundle)
        )

        # Combine local and country social learning with configurable weights
        self._attitude_social_learning = (
            self.agent.weight_attitude_local * self._attitude_social_learning_local  # noqa: E501
            + self.agent.weight_attitude_country * self._attitude_social_learning_country  # noqa: E501
        )

        # Full attitude: own land + combined social learning
        self._attitude = (
            self.agent.weight_own_land * self._attitude_own_land
            + self.agent.weight_social_learning * self._attitude_social_learning
        )

        # -----------------------------------------------------------------
        # Social Norm: local (neighbours) + country-level
        # -----------------------------------------------------------------
        self._social_norm_local = self._compute_social_norm_local(new_bundle)
        self._social_norm_country = self._compute_social_norm_country(new_bundle)

        # Combine with configurable weights
        self._social_norm = (
            self.agent.weight_social_norm_local * self._social_norm_local
            + self.agent.weight_social_norm_country * self._social_norm_country
        )

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
