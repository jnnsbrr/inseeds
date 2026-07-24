"""Theory of Planned Behaviour (TPB) decision model for Conservation Agriculture.

This module implements how farmers decide whether to adopt or abandon
agricultural practices. It uses the Theory of Planned Behaviour framework
combined with social learning and adaptive management.

Overview
--------
Each year, farmers may reconsider their practices based on:
1. **Attitude**: "Do I think this practice is good?" (own experience + social learning)
2. **Social Norm**: "Are my neighbors doing it?" (local + country + cluster)
3. **Perceived Behavioral Control (PBC)**: "Can I afford it?"

These combine into a TPB intention score. If it exceeds a threshold, the
farmer transitions to the new practice.

Decision Flow
-------------
    ┌─────────────────────────────────────────────────────────────┐
    │                    Annual Update                             │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Check if evaluation time (not every year)               │
    │  2. Check fallback (sustained decline → revert)             │
    │  3. Find target bundle:                                     │
    │     a. Local neighbor with better performance?              │
    │     b. Country-level best bundle?                           │
    │     c. Cluster-level best bundle?                           │
    │     d. Random exploration?                                  │
    │  4. Adjust for affordability                                │
    │  5. Compute TPB score                                       │
    │  6. If TPB > threshold → transition                         │
    └─────────────────────────────────────────────────────────────┘

Key Classes
-----------
DecisionModel
    Abstract base class for farmer decision models.
TPB
    Full TPB implementation with social learning at three spatial scales.

Transition Blockers and Drivers
-------------------------------
The model tracks WHY transitions happen (or don't). See BLOCKER_* and DRIVER_*
constants for the full list of codes. This enables analysis of adoption barriers.

Spatial Scales of Social Learning
---------------------------------
- **Local**: Direct neighbor comparison (highest weight)
- **Country**: Average performance across the country
- **Cluster**: Average across agroecologically similar countries

References
----------
.. [1] Ajzen, I. (1991). The theory of planned behavior. Organizational Behavior
       and Human Decision Processes, 50(2), 179-211.
.. [2] Bandura, A. (1977). Social Learning Theory. Prentice Hall.
.. [3] Holling, C.S. (1978). Adaptive Environmental Assessment and Management.

See Also
--------
ca_management : Practice bundles and performance tracking
ca_country : Country-level aggregation
ca_agroecology : Cluster-level aggregation
"""

from abc import ABC, abstractmethod
import logging

import numpy as np

from inseeds.components.farming.farmer import sigmoid, NON_CROPS

logger = logging.getLogger(__name__)
from inseeds.components.farming.ca_management import (
    ManagementBundle,
    ManagementPerformanceMemory,
    ManagementPerformanceTracker,
    PRACTICE_FIELDS,
    RegionManagementPerformanceStore,
    compute_performance_score,
)


def _country_performance_store(country) -> RegionManagementPerformanceStore | None:
    """Return country management performance store if available, None otherwise."""
    store = country.statistic.get("management_performance")
    if isinstance(store, RegionManagementPerformanceStore):
        return store
    return None


def redistribute_weights(
    weight_local: float,
    weight_country: float,
    weight_cluster: float,
    enable_local: bool = True,
    enable_country: bool = True,
    enable_cluster: bool = True,
) -> tuple[float, float, float]:
    """Redistribute weights among enabled spreading levels.

    When a level is disabled, its weight is redistributed proportionally
    to the enabled levels so the total remains the same. This ensures
    that disabling a level (e.g., cluster for single-country runs) doesn't
    drag down scores with neutral values.

    Parameters
    ----------
    weight_local : float
        Weight for local (neighbour) level.
    weight_country : float
        Weight for country level.
    weight_cluster : float
        Weight for agroecological cluster level.
    enable_local : bool
        Whether local spreading is enabled.
    enable_country : bool
        Whether country spreading is enabled.
    enable_cluster : bool
        Whether cluster spreading is enabled.

    Returns
    -------
    tuple[float, float, float]
        Redistributed weights (local, country, cluster).

    Example
    -------
    >>> redistribute_weights(0.7, 0.2, 0.1, True, True, False)
    (0.7778, 0.2222, 0.0)  # Cluster weight redistributed to local + country
    """
    weights = [
        (weight_local, enable_local),
        (weight_country, enable_country),
        (weight_cluster, enable_cluster),
    ]

    enabled_sum = sum(w for w, enabled in weights if enabled)

    if enabled_sum == 0:
        return (0.0, 0.0, 0.0)

    original_total = weight_local + weight_country + weight_cluster
    scale = original_total / enabled_sum

    return (
        weight_local * scale if enable_local else 0.0,
        weight_country * scale if enable_country else 0.0,
        weight_cluster * scale if enable_cluster else 0.0,
    )


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

# Transition blocker codes - indicates WHY a proposed bundle was not adopted
# These follow the decision flow order (first blocker hit is the primary reason)
BLOCKER_NONE = 0                    # No blocker - transition happened or maintaining current
BLOCKER_OBSERVATION_YEARS = 1       # Observation period not complete (commitment period)
BLOCKER_FALLBACK_TRIGGERED = 2      # Reverting to previous bundle (adaptive management)
BLOCKER_NO_TARGET = 3               # No better neighbour + exploration didn't trigger
BLOCKER_TARGET_SAME = 4             # Target bundle same as current (already optimal)
BLOCKER_TARGET_UNAFFORDABLE = 5     # Target reduced to current due to cost
BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND = 6         # TPB low - own land attitude is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL = 7     # TPB low - LOCAL social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL = 8         # TPB low - LOCAL social norm is limiting
BLOCKER_TPB_LOW_PBC = 9             # TPB below threshold - PBC (cost affordability) is limiting
BLOCKER_TRANSITION_UNAFFORDABLE = 10 # Can't afford transition cost
BLOCKER_CAPITAL_SURVIVAL = 11       # Capital below survival threshold
BLOCKER_CONTROL_RUN = 12            # Control run - no CA dynamics
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY = 13  # TPB low - COUNTRY social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY = 14      # TPB low - COUNTRY social norm is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER = 15  # TPB low - CLUSTER social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER = 16      # TPB low - CLUSTER social norm is limiting
BLOCKER_AFFORDABILITY_FORCED = 17   # Practices deselected due to unaffordable direct costs

BLOCKER_NAMES = {
    BLOCKER_NONE: "none",
    BLOCKER_FALLBACK_TRIGGERED: "fallback_triggered",
    BLOCKER_NO_TARGET: "no_target",
    BLOCKER_TARGET_SAME: "target_same",
    BLOCKER_TARGET_UNAFFORDABLE: "target_unaffordable",
    BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND: "tpb_low_attitude_own_land",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL: "tpb_low_attitude_social_local",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY: "tpb_low_attitude_social_country",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER: "tpb_low_attitude_social_cluster",
    BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL: "tpb_low_social_norm_local",
    BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY: "tpb_low_social_norm_country",
    BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER: "tpb_low_social_norm_cluster",
    BLOCKER_TPB_LOW_PBC: "tpb_low_pbc",
    BLOCKER_TRANSITION_UNAFFORDABLE: "transition_unaffordable",
    BLOCKER_CAPITAL_SURVIVAL: "capital_survival",
    BLOCKER_CONTROL_RUN: "control_run",
    BLOCKER_OBSERVATION_YEARS: "observation_years",
    BLOCKER_AFFORDABILITY_FORCED: "affordability_forced",
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
DRIVER_EXPLORATION_ATTITUDE_SOCIAL_CLUSTER = 20  # TPB passed, cluster social learning was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL = 17    # TPB passed, local social norm was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY = 18  # TPB passed, country social norm was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_CLUSTER = 21  # TPB passed, cluster social norm was strongest
DRIVER_EXPLORATION_PBC = 19                  # TPB passed, PBC (cost affordability) was strongest

# Additional cluster-level drivers for LOCAL and COUNTRY pathways
DRIVER_LOCAL_ATTITUDE_SOCIAL_CLUSTER = 22    # TPB passed, cluster social learning was strongest (local pathway)
DRIVER_LOCAL_SOCIAL_NORM_CLUSTER = 23        # TPB passed, cluster social norm was strongest (local pathway)
DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER = 24  # TPB passed, cluster social learning was strongest (country pathway)
DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER = 25      # TPB passed, cluster social norm was strongest (country pathway)

# Forced transitions (bypass TPB)
DRIVER_AFFORDABILITY_FORCED = 26             # Practices deselected due to unaffordable direct costs

DRIVER_NAMES = {
    DRIVER_NONE: "none",
    DRIVER_FALLBACK: "fallback",
    # Social pathway (local neighbor inspiration)
    DRIVER_LOCAL_ATTITUDE_OWN_LAND: "local_attitude_own_land",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL: "local_attitude_social_local",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY: "local_attitude_social_country",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_CLUSTER: "local_attitude_social_cluster",
    DRIVER_LOCAL_SOCIAL_NORM_LOCAL: "local_social_norm_local",
    DRIVER_LOCAL_SOCIAL_NORM_COUNTRY: "local_social_norm_country",
    DRIVER_LOCAL_SOCIAL_NORM_CLUSTER: "local_social_norm_cluster",
    DRIVER_LOCAL_PBC: "local_pbc",
    # Country pathway (country-level inspiration)
    DRIVER_COUNTRY_ATTITUDE_OWN_LAND: "country_attitude_own_land",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL: "country_attitude_social_local",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY: "country_attitude_social_country",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER: "country_attitude_social_cluster",
    DRIVER_COUNTRY_SOCIAL_NORM_LOCAL: "country_social_norm_local",
    DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY: "country_social_norm_country",
    DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER: "country_social_norm_cluster",
    DRIVER_COUNTRY_PBC: "country_pbc",
    # Exploration pathway (random exploration)
    DRIVER_EXPLORATION_ATTITUDE_OWN_LAND: "exploration_attitude_own_land",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL: "exploration_attitude_social_local",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY: "exploration_attitude_social_country",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_CLUSTER: "exploration_attitude_social_cluster",
    DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL: "exploration_social_norm_local",
    DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY: "exploration_social_norm_country",
    DRIVER_EXPLORATION_SOCIAL_NORM_CLUSTER: "exploration_social_norm_cluster",
    DRIVER_EXPLORATION_PBC: "exploration_pbc",
    # Forced transitions
    DRIVER_AFFORDABILITY_FORCED: "affordability_forced",
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

    def get_aft_param(self, param_name):
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
            If the parameter is not defined on the farmer agent (missing from config).
        """
        if not hasattr(self.farmer, param_name):
            aft_name = getattr(self.farmer.aft, "name", "unknown")
            raise AttributeError(
                f"Missing AFT parameter '{param_name}' for AFT '{aft_name}'. "
                f"Add it to config.yaml under aftpar.{aft_name}.{param_name}"
            )
        return getattr(self.farmer, param_name)

    def __init__(self, farmer):
        """Initialize decision model from farmer's current practices.

        Parameters
        ----------
        farmer : ConservationAgricultureFarmer
            The farmer agent owning this behaviour.
        """
        self.farmer = farmer

        # Initialize transition blocker and driver to none
        self.transition_blocker = BLOCKER_NONE
        self.transition_driver = DRIVER_NONE

        # -----------------------------------------------------------------
        # Initialize current practice bundle from farmer agent state
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
        ca_threshold = farmer.model.config.coupled_config.practice_dimensions.residue.ca_cover_threshold  # noqa: E501

        self.practice_bundle = ManagementBundle.from_practices(
            int(farmer.tillage),
            1 if farmer.cover_crop > 0 else 0, # TODO: change to cover_crop type: 0 = none, 1 = catch crop, 2 = legume
            1 if farmer.litter_cover >= ca_threshold else 0,
        )

        # Proposed bundle for potential transition (set by update())
        self.proposed_bundle = None

        # Previous bundle (for fallback mechanism)
        self.previous_bundle = None

        # Counter for consecutive years of declining performance
        self.decline_years = 0

        # Online regression state for trend computation since last transition
        self.performance_tracker = ManagementPerformanceTracker.from_farmer(farmer)

        # -----------------------------------------------------------------
        # Initialize management bundle memory
        # -----------------------------------------------------------------
        # Farmers remembers performance from each bundle they've tried
        # Memory includes: trends, duration used, when last updated, failure count

        current_year = self.farmer.model.lpjml.sim_year
        self.performance_memory = ManagementPerformanceMemory(
            self.practice_bundle,
            self.performance_tracker,
            current_year,
        )

        # baseline_score will be set on first update (avoids init-order issues
        # with reference_scales not being computed yet)

    # -------------------------------------------------------------------------
    # Properties for external access
    # -------------------------------------------------------------------------

    @property
    def practice_bundle_id(self):
        """Current practice bundle as numeric ID (0-7).

        This property is settable for Dask actor synchronization.
        Setting it converts the ID back to a ManagementBundle enum.
        """
        return self.practice_bundle.id if self.practice_bundle else -1

    @practice_bundle_id.setter
    def practice_bundle_id(self, value):
        """Set practice bundle from numeric ID.

        Used by Dask actor sync to restore bundle state on driver.
        """
        if value is not None and value >= 0:
            self.practice_bundle = ManagementBundle.from_id(int(value))

    @property
    def proposed_bundle_label(self):
        """Proposed bundle as human-readable name."""
        return self.proposed_bundle.label if self.proposed_bundle else ""

    @property
    def proposed_bundle_id(self):
        """Proposed bundle as numeric ID (-1 if none)."""
        return self.proposed_bundle.id if self.proposed_bundle else -1

    @property
    def transition_blocker(self):
        """Primary reason why proposed bundle was not adopted (numeric code).
        
        See BLOCKER_* constants for codes. 0 = no blocker (transition happened
        or no transition needed).
        """
        return getattr(self, '_transition_blocker', BLOCKER_NONE)

    @transition_blocker.setter
    def transition_blocker(self, value):
        """Set transition_blocker to a specific value.
        Parameters
        ----------
        value : int
            The blocker value to set.
        """
        self._transition_blocker = value
        if value not in BLOCKER_NAMES.keys():
            raise ValueError(f"Invalid transition blocker value: {value}. Must be one of: {BLOCKER_NAMES.keys()}")

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

    @transition_driver.setter
    def transition_driver(self, value):
        """Set transition_driver to a specific value.
        Parameters
        ----------
        value : int
            The driver value to set.
        """
        self._transition_driver = value
        if value not in DRIVER_NAMES.keys():
            raise ValueError(f"Invalid transition driver value: {value}. Must be one of: {DRIVER_NAMES.keys()}")

    @property
    def transition_driver_name(self):
        """Human-readable name of transition driver."""
        return DRIVER_NAMES.get(self.transition_driver)

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
        current_year = self.farmer.model.lpjml.sim_year
        n_obs = self.performance_tracker.n

        # -----------------------------------------------------------------
        # Store outcome of outgoing bundle (if used for at least 2 years)
        # -----------------------------------------------------------------
        if n_obs > 1:
            self.performance_memory.record_performance(
                self.practice_bundle,
                self.performance_tracker.trend,
                self.performance_tracker.level,
                n_obs,
                current_year,
            )

        # -----------------------------------------------------------------
        # Prepare for new bundle: reset regression state
        # -----------------------------------------------------------------

        # Remember previous bundle (for potential fallback)
        self.previous_bundle = self.practice_bundle

        # Capture baseline score before resetting (for fallback comparison)
        # This is the performance level we expect to maintain or exceed
        baseline = (
            compute_performance_score(self.performance_tracker, self.farmer)
            if n_obs > 1 else 1.0  # Neutral score
        )

        # Reset regression accumulators for new bundle
        self.performance_tracker = ManagementPerformanceTracker.reset_for_transition(
            farmer=self.farmer,
            current_year=current_year,
            baseline_score=baseline
        )

        # Transition to new bundle
        self.practice_bundle = new_bundle

        # Reset decline counter
        self.decline_years = 0

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

    def __init__(self, farmer):
        """Initialize TPB model with zero scores."""
        super().__init__(farmer)

        # TPB components (updated each timestep)
        self._tpb = 0.0          # Overall intention score
        self._attitude = 0.0     # Attitude toward behavior (combined)
        self._attitude_own_land = 0.0    # Attitude from own land performance
        self._attitude_social_learning = 0.0  # Attitude from social learning
        self._social_norm = 0.0  # Subjective norm
        self._pbc = 0.0          # Perceived behavioral control

        # Observation years counter: randomize initial timing to desynchronize farmers
        # (avoids artificial waves of simultaneous evaluation)
        min_obs = self.get_aft_param("min_observation_years")
        self._observation_years = np.random.randint(1, min_obs)

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
        the observation period has ended (randomized around min_observation_years).

        Returns
        -------
        bool
            True if farmer should run TPB evaluation this year.
        """
        return self._observation_years <= 0

    def reset_observation_years(self):
        """Reset observation years after a transition decision (transition or stay).

        Uses normal distribution around min_observation_years (mean=min_obs,
        std=min_obs/2) to create heterogeneity in re-evaluation timing.
        This reflects that some farmers re-evaluate sooner (more proactive)
        while others wait longer (more conservative). Minimum is 1 year.

        Called after TPB evaluation completes, regardless of whether transition
        happened.
        """
        min_obs = self.get_aft_param("min_observation_years")
        if min_obs > 0:
            # Normal distribution around min_obs, minimum 1 year
            self._observation_years = max(
                1, int(np.random.normal(min_obs, min_obs / 2))
            )
        else:
            self._observation_years = 0

    def decrement_observation_years(self):
        """Decrement the observation years counter by one year.

        Called each year when farmer doesn't evaluate.
        """
        if self._observation_years > 0:
            self._observation_years -= 1

    # =========================================================================
    # MAIN UPDATE LOGIC
    # =========================================================================

    def reevaluate_residue_status(self):
        """Re-evaluate residue component of bundle based on actual litter cover.

        Only runs if config.practice_dimensions.residue.reevaluate_residue_status
        is True (default: False).

        When enabled, farmers who organically cross the CA residue threshold
        (30% soil cover) get their bundle upgraded to residue=1.

        - If litter_cover >= threshold AND bundle residue = 0 → upgrade to 1

        Note: We do NOT downgrade farmers who drop below threshold. If a farmer
        intends to retain residue (residue=1) but has low litter cover due to
        poor yields, setting residue=0 would tell LPJmL to actively REMOVE
        residue, which contradicts their intent and worsens the situation.
        Stopping retention should be an explicit farmer decision (via TPB),
        not an automatic consequence of low yields.

        When disabled (default), the bundle reflects the farmer's explicit
        decision only. CA certification can be derived in post-processing
        by combining bundle + litter_cover data.
        """
        # Check if reevaluation by litter cover is enabled (default: False)
        # When disabled, bundle reflects farmer's explicit decision only.
        residue_config = self.farmer.model.config.coupled_config.practice_dimensions.residue
        if not residue_config.residue_status_by_cover:
            return

        # Evaluate residue status by actual litter cover
        ca_threshold = residue_config.ca_cover_threshold
        current_residue = self.practice_bundle.residue_on_field
        litter_cover = self.farmer.litter_cover

        # Upgrade to CA status if threshold is met and not already retaining
        # Note: No downgrade - stopping retention should be an explicit decision
        if litter_cover >= ca_threshold and current_residue == 0:
            self.practice_bundle = self.practice_bundle.change_practices(
                residue_on_field=1
            )
        elif litter_cover < ca_threshold and current_residue == 1:
            self.practice_bundle = self.practice_bundle.change_practices(
                residue_on_field=0
            )

    def reevaluate_cover_crop_type(self):
        """Re-evaluate cover crop type based on fertilization and leaching.

        Cover crop *type* (legume vs non-legume) is stored on the farmer,
        not the bundle. The bundle only tracks whether cover crops are ON/OFF.

        Type values:
        - 1 = non-legume (catch crop, captures excess nutrients)
        - 2 = legume (N-fixing, reduces fertilizer need)
        """
        # Only re-evaluate if cover crops are enabled
        if self.practice_bundle.cover_crop == 0:
            return

        indicated_type = self.farmer.indicate_cover_crop_type()
        current_type = self.farmer.cover_crop

        if indicated_type != current_type:
            self.farmer.cover_crop = indicated_type

    def update(self):
        """Compute proposed bundle and TPB scores for this timestep.

        Decision flow:
        0. Re-evaluate residue status based on actual litter cover
        1. Add current observation to regression accumulators
        2. Update performance memory with current trends (for neighbour visibility)
        3. Decay old memories (bounded rationality)
        4. Check if minimum observation period has passed
        5. Check fallback condition (sustained decline → revert)
        6. Find best-performing neighbour's bundle OR explore randomly
        7. Adjust target bundle for affordability
        8. Compute TPB scores for the proposed bundle
        
        Sets transition_blocker to indicate why transition didn't happen (if applicable).
        Sets transition_driver to indicate why transition succeeded (if applicable).
        """
        # Reset blocker, driver, and pathway at start of each update
        self.transition_blocker = BLOCKER_NONE
        self.transition_driver = DRIVER_NONE
        self.target_pathway = None

        # Reset TPB components so they show 0.0 if not computed this year
        # (e.g., farmer doesn't evaluate due to observation_years)
        self._tpb = 0.0
        self._attitude = 0.0
        self._attitude_own_land = 0.0
        self._attitude_social_learning = 0.0
        self._attitude_social_learning_local = 0.0
        self._attitude_social_learning_country = 0.0
        self._attitude_social_learning_cluster = 0.0
        self._social_norm = 0.0
        self._social_norm_local = 0.0
        self._social_norm_country = 0.0
        self._social_norm_cluster = 0.0
        self._pbc = 0.0

        # -----------------------------------------------------------------
        # Step 0: Re-evaluate residue status and cover crop type
        # -----------------------------------------------------------------
        # Farmers may cross CA threshold organically - certify/de-certify
        self.reevaluate_residue_status()

        self.reevaluate_cover_crop_type()

        # -----------------------------------------------------------------
        # Step 1: Add current year's observation to regression
        # -----------------------------------------------------------------
        # Every metric is fed as a self-referenced fractional rate
        # (value / own_baseline - 1), so all five are dimensionless and centred
        # at zero - the same footing as profit.
        self.performance_tracker.add_observation(
            self.farmer.soilc_rate,
            self.farmer.moisture_rate,
            self.farmer.yield_rate,
            self.farmer.model.lpjml.sim_year,
            profit=self.farmer.profit,
            leaching=self.farmer.leaching_rate,
        )

        # Initialize baseline_score on first update (deferred from init to avoid
        # chicken-and-egg with reference_scales)
        if self.performance_tracker.baseline_score is None and self.performance_tracker.n > 1:
            self.performance_tracker.baseline_score = compute_performance_score(
                self.performance_tracker, self.farmer
            )

        # -----------------------------------------------------------------
        # Step 2: Update performance memory with current trends
        # -----------------------------------------------------------------
        # Keep performance memory up-to-date so neighbours see current performance
        self.performance_memory.update_current(
            self.practice_bundle,
            self.performance_tracker.trend,
            self.performance_tracker.level,
            self.performance_tracker.n,
            self.farmer.model.lpjml.sim_year,
        )

        # -----------------------------------------------------------------
        # Step 3: Decay old memories
        # -----------------------------------------------------------------
        self.performance_memory.decay(
            self.farmer.model.lpjml.sim_year,
            self.get_aft_param("memory_decay_years"),
        )

        # -----------------------------------------------------------------
        # Step 4: Require minimum observation years before transitioning
        # -----------------------------------------------------------------
        # Farmers record observations every year (above), but only evaluate
        # TPB once the observation period is complete. This avoids noisy
        # decisions based on single-year fluctuations.
        # After a transition, the counter resets via reset_observation_years().

        if not self.should_evaluate():
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_OBSERVATION_YEARS
            self.decrement_observation_years()
            return

        # -----------------------------------------------------------------
        # Step 5: Check affordability deselection
        # -----------------------------------------------------------------
        # If current practices cost more than available capital, propose
        # a bundle with costly practices deselected (survival mechanism)

        if self.check_affordability_deselection():
            # Deselection sets proposed_bundle and _tpb internally
            self.transition_blocker = BLOCKER_AFFORDABILITY_FORCED
            self.target_pathway = "affordability"
            return

        # -----------------------------------------------------------------
        # Step 6: Check fallback condition
        # -----------------------------------------------------------------
        # If performance has declined for FALLBACK_YEARS consecutive years,
        # propose reverting to the previous bundle (adaptive management)

        if self.check_fallback():
            # Fallback sets proposed_bundle and _tpb internally
            # Blocker will be set by should_transition() if TPB too low
            self.transition_blocker = BLOCKER_FALLBACK_TRIGGERED
            self.target_pathway = "fallback"  # Track pathway for transition_driver
            return

        # -----------------------------------------------------------------
        # Step 7: Find target bundle (neighbour imitation or exploration)
        # -----------------------------------------------------------------

        # First, try to imitate best-performing neighbour (local)
        target_bundle = self.most_promising_bundle_local()
        self.target_pathway = "local"

        # If no better local neighbour, try country-level inspiration
        if target_bundle is None:
            target_bundle = self.most_promising_bundle_country()
            self.target_pathway = "country"

        # If no country inspiration, try cluster-level (agroecologically similar countries)
        if target_bundle is None:
            target_bundle = self.most_promising_bundle_cluster()
            self.target_pathway = "cluster"

        # If no cluster inspiration, maybe explore randomly
        if target_bundle is None:
            target_bundle = self.maybe_explore_bundle()
            self.target_pathway = "exploration"

        # No change proposed
        if target_bundle is None:
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_NO_TARGET
            return

        if target_bundle == self.practice_bundle:
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_TARGET_SAME
            return

        # -----------------------------------------------------------------
        # Step 8: Adjust for affordability
        # -----------------------------------------------------------------
        # If farmer can't afford full target bundle, find affordable subset

        affordable_bundle = self.affordable_bundle(target_bundle)

        # If affordable bundle is same as current, no change
        if affordable_bundle == self.practice_bundle:
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_TARGET_UNAFFORDABLE
            return

        # -----------------------------------------------------------------
        # Step 9: Compute TPB scores for proposed bundle
        # -----------------------------------------------------------------
        self.proposed_bundle = affordable_bundle
        self.compute_tpb_for_bundle(affordable_bundle)

    # =========================================================================
    # AFFORDABILITY DESELECTION (Survival Mechanism)
    # =========================================================================

    def check_affordability_deselection(self):
        """Check if practices must be deselected due to unaffordable costs.

        When capital is too low to sustain current practices, propose a bundle
        with practices deselected in order of cost (most expensive first) until
        direct costs are within the affordable range.

        Like fallback, this is an emergency response that bypasses TPB (sets
        _tpb = 1.0) because the farmer has no choice but to abandon costly
        practices to survive. However, the transition still goes through the
        normal pathway for proper recording.

        Returns
        -------
        bool
            True if deselection is needed (sets proposed_bundle internally).
        """
        farmer = self.farmer
        available_capital = farmer.capital - farmer.min_capital

        # Calculate current annual direct costs
        bundle = self.practice_bundle
        current_direct_costs = farmer.get_bundle_direct_costs(bundle=bundle)

        # No action needed if costs are within budget
        if current_direct_costs <= available_capital:
            return False

        # Deselection order (most expensive first)
        deselect_order = ["cover_crop", "residue_on_field", "tillage"]

        # Find affordable bundle by deselecting practices
        for practice_name in deselect_order:
            if getattr(bundle, practice_name) == 0:
                continue

            # Try deselecting this practice
            new_bundle = bundle.change_practices(**{practice_name: 0})
            new_costs = farmer.get_bundle_direct_costs(bundle=new_bundle)

            # Only deselect if it actually reduces costs (or provides savings)
            if new_costs < current_direct_costs:
                bundle = new_bundle
                current_direct_costs = new_costs

                # Check if costs are now within budget
                if current_direct_costs <= available_capital:
                    break

        # No change needed if bundle didn't change
        if bundle == self.practice_bundle:
            return False

        # Propose the deselected bundle
        self.proposed_bundle = bundle

        # Bypass TPB: this is an emergency survival response, not a planned
        # behavior change. Set intention directly to 1.0.
        self._tpb = 1.0

        return True

    # =========================================================================
    # FALLBACK MECHANISM (Adaptive Management)
    # =========================================================================

    def check_fallback(self):
        """Check if farmer should revert to previous bundle due to sustained decline.

        Implements adaptive management: if outcomes have declined for
        fallback_years consecutive years, propose reverting to the
        previous practice bundle.

        Returns
        -------
        bool
            True if fallback is triggered (sets proposed_bundle internally).
        """
        # Can't fall back if no previous bundle
        if self.previous_bundle is None:
            return False

        # -----------------------------------------------------------------
        # Compare current performance to baseline at transition time
        # -----------------------------------------------------------------
        # baseline_score captures the performance score at time of transition.
        # If current score is worse than baseline, we're declining.
        baseline = self.performance_tracker.baseline_score
        if baseline is None:
            return False  # Can't compare without baseline

        current_score = compute_performance_score(self.performance_tracker, self.farmer)

        # Track consecutive years of decline
        if current_score < baseline:
            self.decline_years += 1
        else:
            self.decline_years = 0  # Reset if performance improves

        # -----------------------------------------------------------------
        # Trigger fallback after sustained decline
        # -----------------------------------------------------------------
        fallback_years = self.get_aft_param("fallback_years")
        if self.decline_years >= fallback_years:
            # Mark current bundle as "failed" (reduces future exploration)
            self.performance_memory.record_failure(self.practice_bundle)

            # Propose reverting to previous bundle
            self.proposed_bundle = self.previous_bundle

            # Fallback bypasses TPB: this is an emergency response, not a
            # planned behavior change. Set intention directly to 1.0.
            # TPB factors are left unchanged (not meaningful for fallback).
            self._tpb = 1.0

            return True

        return False

    # =========================================================================
    # RANDOM EXPLORATION (Innovation Diffusion)
    # =========================================================================

    def maybe_explore_bundle(self):
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
        base_prob = self.get_aft_param("exploration_base_prob")

        # Poor performers explore more (searching for better options)
        # Score < baseline_score means declining from when we started this practice
        baseline = self.performance_tracker.baseline_score
        if baseline is not None:
            current_score = compute_performance_score(self.performance_tracker, self.farmer)
            if current_score < baseline:
                base_prob *= self.get_aft_param("poor_performance_multiplier")

        # Cap exploration probability
        explore_prob = min(base_prob, self.get_aft_param("max_exploration_prob"))

        # -----------------------------------------------------------------
        # Random draw: explore or not?
        # -----------------------------------------------------------------
        if np.random.random() > explore_prob:
            return None  # No exploration this year

        # -----------------------------------------------------------------
        # Select valid bundle to explore (biased toward incremental changes)
        # -----------------------------------------------------------------
        exploration = self.farmer.model.config.coupled_config.exploration
        max_failures = getattr(exploration, "max_failures", 2)

        # Generate all possible bundles with their "distance" from current
        current = self.practice_bundle
        valid = []
        for b in ManagementBundle.all_bundles():
            if b == current:
                continue
            if self.performance_memory[b].failure_count >= max_failures:
                continue
            # Distance = number of practice changes (0-3)
            dist = (
                int(b.tillage != current.tillage)
                + int(b.cover_crop != current.cover_crop)
                + int(b.residue_on_field != current.residue_on_field)
            )
            valid.append((b, dist))

        if not valid:
            return None

        # Weight by inverse distance: 1-change bundles are 3x more likely than 3-change
        # weights: dist=1 -> 3, dist=2 -> 2, dist=3 -> 1
        # Additionally penalize bundles already tried (in memory) - favor novel options
        weights = []
        for b, dist in valid:
            w = 4 - dist  # base weight from distance
            if self.performance_memory[b].duration > 0:
                w *= 0.25  # already tried -> 4x less likely
            weights.append(w)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        idx = np.random.choice(len(valid), p=weights)
        return valid[idx][0]

    # =========================================================================
    # TRANSITION DECISION
    # =========================================================================

    def should_transition(self):
        """Determine if farmer should transition to proposed bundle.

        Compares TPB intention score to threshold.
        Hysteresis (avoiding flip-flopping) is handled by min_observation_years:
        after transitioning, the performance tracker resets (n=1), so farmers
        need min_observation_years of new data before reconsidering.

        Thresholds are non-AFT-specific (from config.tpb).
        Behavioral differences between AFTs come from weights and PBC.

        Returns
        -------
        bool
            True if TPB exceeds threshold and transition should occur.
        """
        tpb_config = self.farmer.model.config.coupled_config.tpb
        threshold = tpb_config.transition_threshold

        return self._tpb > threshold

    def _get_enabled_spreading_levels(self):
        """Get which spreading levels are enabled from config.

        Returns
        -------
        tuple[bool, bool, bool]
            (enable_local, enable_country, enable_cluster)
        """
        spreading_config = getattr(
            self.farmer.model.config.coupled_config, "tpb", None
        )
        enable_local = getattr(spreading_config, "enable_local", True) if spreading_config else True
        enable_country = getattr(spreading_config, "enable_country", True) if spreading_config else True
        enable_cluster = getattr(spreading_config, "enable_cluster", True) if spreading_config else True
        return enable_local, enable_country, enable_cluster



    def set_tpb_transition_blocker(self):
        """Set transition_blocker to indicate which TPB component is most limiting.

        Uses a simple recursive approach:
        1. First identify which main component (attitude, social_norm, pbc) is lowest
        2. Then drill down into that component's sub-parts to identify the specific blocker

        Only compares enabled spreading levels (respects tpb config).
        """
        if self.proposed_bundle is None:
            return

        # Get enabled levels to avoid blaming disabled components
        enable_local, enable_country, enable_cluster = self._get_enabled_spreading_levels()

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
                self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND
            else:
                # Compare only ENABLED social learning levels
                social_components = {}
                if enable_local:
                    social_components['local'] = self._attitude_social_learning_local
                if enable_country:
                    social_components['country'] = self._attitude_social_learning_country
                if enable_cluster:
                    social_components['cluster'] = self._attitude_social_learning_cluster

                if social_components:
                    min_social = min(social_components, key=social_components.get)
                    if min_social == 'local':
                        self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL
                    elif min_social == 'country':
                        self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY
                    else:
                        self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER
                else:
                    # Fallback if no social levels enabled (shouldn't happen)
                    self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND

        elif main_blocker == 'social_norm':
            # Compare only ENABLED social norm levels
            norm_components = {}
            if enable_local:
                norm_components['local'] = self._social_norm_local
            if enable_country:
                norm_components['country'] = self._social_norm_country
            if enable_cluster:
                norm_components['cluster'] = self._social_norm_cluster

            if norm_components:
                min_norm = min(norm_components, key=norm_components.get)
                if min_norm == 'local':
                    self.transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL
                elif min_norm == 'country':
                    self.transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY
                else:
                    self.transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER
            else:
                # Fallback if no social levels enabled (shouldn't happen)
                self.transition_blocker = BLOCKER_TPB_LOW_PBC

        else:  # pbc
            # PBC is driven by cost affordability (pbc_base captures AFT risk differences)
            self.transition_blocker = BLOCKER_TPB_LOW_PBC

    def set_tpb_component_driver(self, pathway: str):
        """Set transition_driver to indicate which TPB component enabled the transition.

        Uses a simple recursive approach (mirror of set_tpb_transition_blocker):
        1. First identify which main component (attitude, social_norm, pbc) is highest
        2. Then drill down into that component's sub-parts to identify the specific driver

        Only compares enabled spreading levels (respects tpb config).

        Parameters
        ----------
        pathway : str
            One of "local" (learned from local neighbor), "country" (inspired
            by country-level data), "cluster" (agroecological cluster learning),
            or "exploration" (random exploration).
        """
        if self.proposed_bundle is None:
            return

        # Get enabled levels to avoid crediting disabled components
        enable_local, enable_country, enable_cluster = self._get_enabled_spreading_levels()

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
                # Compare only ENABLED social learning levels
                social_components = {}
                if enable_local:
                    social_components['local'] = self._attitude_social_learning_local
                if enable_country:
                    social_components['country'] = self._attitude_social_learning_country
                if enable_cluster:
                    social_components['cluster'] = self._attitude_social_learning_cluster

                if social_components:
                    max_social = max(social_components, key=social_components.get)
                    if max_social == 'local':
                        sub_driver = 'attitude_social_local'
                    elif max_social == 'country':
                        sub_driver = 'attitude_social_country'
                    else:
                        sub_driver = 'attitude_social_cluster'
                else:
                    # Fallback if no social levels enabled
                    sub_driver = 'attitude_own_land'

        elif main_driver == 'social_norm':
            # Compare only ENABLED social norm levels
            norm_components = {}
            if enable_local:
                norm_components['local'] = self._social_norm_local
            if enable_country:
                norm_components['country'] = self._social_norm_country
            if enable_cluster:
                norm_components['cluster'] = self._social_norm_cluster

            if norm_components:
                max_norm = max(norm_components, key=norm_components.get)
                if max_norm == 'local':
                    sub_driver = 'social_norm_local'
                elif max_norm == 'country':
                    sub_driver = 'social_norm_country'
                else:
                    sub_driver = 'social_norm_cluster'
            else:
                # Fallback if no social levels enabled
                sub_driver = 'pbc'

        else:  # pbc
            # PBC is driven by cost affordability (pbc_base captures AFT risk differences)
            sub_driver = 'pbc'

        # Map based on pathway and sub-component
        # Pathways: "local" (learned from local neighbor), "country" (inspired by
        # country-level stats), "cluster" (agroecological cluster), "exploration"
        driver_maps = {
            "local": {
                'attitude_own_land': DRIVER_LOCAL_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_LOCAL_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_LOCAL_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_LOCAL_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_LOCAL_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_LOCAL_PBC,
            },
            "country": {
                'attitude_own_land': DRIVER_COUNTRY_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_COUNTRY_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_COUNTRY_PBC,
            },
            "cluster": {
                'attitude_own_land': DRIVER_COUNTRY_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_COUNTRY_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_COUNTRY_PBC,
            },
            "exploration": {
                'attitude_own_land': DRIVER_EXPLORATION_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_EXPLORATION_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_EXPLORATION_PBC,
            },
        }

        driver_map = driver_maps.get(pathway, driver_maps["exploration"])
        self.transition_driver = driver_map[sub_driver]

    # =========================================================================
    # APPLY BUNDLE TO AGENT
    # =========================================================================

    def apply_bundle(self, bundle):
        """Apply practice bundle to farmer and push to LPJmL.

        Sets farmer's tillage, cover_crop, and residue_on_field attributes
        and sends updates to the coupled LPJmL model.

        Parameters
        ----------
        bundle : Bundle
            Practice bundle (tillage, cover_crop, residue_on_field), each 0 or 1.
        """
        # -----------------------------------------------------------------
        # Tillage: 0 = no-till (CA), 1 = conventional tillage
        # Directly maps to LPJmL's with_tillage
        # -----------------------------------------------------------------
        self.farmer.tillage = bundle.tillage

        # -----------------------------------------------------------------
        # Cover crop: 0 = none, 1/2 = type based on conditions
        # -----------------------------------------------------------------
        if bundle.cover_crop == 1:
            # Determine cover crop type based on environmental conditions
            # 1 = non-legume (catch crop), 2 = legume (N-fixing)
            self.farmer.cover_crop = self.farmer.indicate_cover_crop_type()
        else:
            self.farmer.cover_crop = 0

        # -----------------------------------------------------------------
        # Residue retention: 0 = baseline, 1 = retain all residue
        # -----------------------------------------------------------------
        # When farmer decides to retain (bundle.residue_on_field == 1), they
        # leave all residue on field (residue_on_field = 1.0 for LPJmL).
        # Whether this meets CA threshold (30% cover) is determined separately.
        if bundle.residue_on_field == 1:
            # Farmer decided to retain residue - leave everything on field
            self.farmer.residue_on_field = 1.0
        else:
            # Not retaining - use baseline residue behavior
            self.farmer.residue_on_field = self.farmer.residue_baseline

        # -----------------------------------------------------------------
        # Push changes to LPJmL
        # -----------------------------------------------------------------
        for attr in ["tillage", "cover_crop", "residue_on_field"]:
            self.farmer.set_lpjml(attribute=attr)

    # =========================================================================
    # COMPARISON METHODS
    # =========================================================================

    def most_promising_bundle_local(self):
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

        for neighbour in self.farmer.neighbourhood:
            # Skip neighbours who aren't performing better
            if not self.is_better_performing(neighbour):
                continue

            # Track neighbour with largest performance gap
            gap = self.performance_gap(neighbour)
            if gap > best_gap:
                best_gap = gap
                best_neighbour = neighbour

        if best_neighbour is None:
            return None

        return best_neighbour.behaviour.practice_bundle

    def most_promising_bundle_country(self):
        """Find best-performing bundle at COUNTRY level (if better than current).

        Non-local social learning: when no local neighbour is better, farmers
        may look to successful practices used elsewhere in their country.
        Country stats are already merged with neighbouring countries (configured
        via neighbour_country_weight) at the country level.

        Uses unified performance scoring with proper normalization and
        metric-specific trend/level weighting.

        Returns
        -------
        tuple or None
            Best country-level bundle, or None if no bundle performs better
            than current practice or if country data is unavailable.
        """
        country = self.farmer.cell.country
        store = _country_performance_store(country)
        if store is None:
            return None

        # Compute own performance using unified scoring
        own_score = compute_performance_score(
            self.performance_tracker, self.farmer
        )

        # Find best performing bundle at country level
        best_bundle = None
        best_score = own_score  # Must beat our current score

        for bundle, perf in store.items():
            # Skip our own bundle
            if bundle == self.practice_bundle:
                continue

            bundle_score = compute_performance_score(perf, self.farmer)

            # Track the best bundle (highest unified score)
            if bundle_score > best_score:
                best_score = bundle_score
                best_bundle = bundle

        return best_bundle

    def most_promising_bundle_cluster(self):
        """Find best-performing bundle at CLUSTER level (if better than current).

        Tele-coupled social learning: when no local or country-level neighbour
        is better, farmers may look to successful practices in agroecologically
        similar countries (same temperature, precipitation, PET patterns).

        Uses unified performance scoring with proper normalization and
        metric-specific trend/level weighting.

        Returns
        -------
        tuple or None
            Best cluster-level bundle, or None if no bundle performs better
            than current practice or if cluster data is unavailable.
        """
        country = self.farmer.cell.country
        cluster_id = getattr(country, "agroecological_cluster", -1)

        if cluster_id < 0:
            return None

        # Access via country._world which works on both driver and workers
        cluster_stats = self.farmer.cell.country._world.statistic.get(
            "cluster_management_performance", {}
        )
        store = cluster_stats.get(cluster_id)
        if not isinstance(store, RegionManagementPerformanceStore):
            return None

        # Compute own performance using unified scoring
        own_score = compute_performance_score(
            self.performance_tracker, self.farmer
        )

        # Find best performing bundle at cluster level
        best_bundle = None
        best_score = own_score  # Must beat our current score

        for bundle, perf in store.items():
            # Skip our own bundle
            if bundle == self.practice_bundle:
                continue

            bundle_score = compute_performance_score(perf, self.farmer)

            # Track the best bundle (highest unified score)
            if bundle_score > best_score:
                best_score = bundle_score
                best_bundle = bundle

        return best_bundle

    def is_better_performing(self, neighbour):
        """Check if neighbour is performing better than self.

        Uses unified performance scoring with proper normalization and
        metric-specific trend/level weighting. A neighbor is better if their
        unified score exceeds ours.

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        bool
            True if neighbour's unified score exceeds self's score.
        """

        my_score = compute_performance_score(
            self.performance_tracker, self.farmer
        )
        their_score = compute_performance_score(
            neighbour.behaviour.performance_tracker, self.farmer
        )

        # if self.farmer.cell.output.cell.item() == 8:
        #     breakpoint()
        # self.farmer.world.statistic.get("reference_scales")
        return their_score > my_score

    def performance_gap(self, neighbour):
        """Compute positive performance score difference (neighbour - self).

        Uses unified performance scoring with proper normalization and
        metric-specific trend/level weighting.

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        float
            Positive score gap (0 if neighbour has lower or equal score).
        """

        my_score = compute_performance_score(
            self.performance_tracker, self.farmer
        )
        their_score = compute_performance_score(
            neighbour.behaviour.performance_tracker, self.farmer
        )

        # Return positive gap only (0 if neighbor is not better)
        return max(0.0, their_score - my_score)

    # =========================================================================
    # CROP SIMILARITY
    # =========================================================================

    def crop_similarity(self, neighbour):
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
        # Get crop fractions via farmer's get_from_earth (handles multi-year data)
        # Exclude managed grassland - it's not a crop for similarity comparison
        cft_self = self.farmer.get_from_earth(
            "cftfrac", drop_band=NON_CROPS, time_idx=-1
        ).values.flatten()
        cft_neighbour = neighbour.get_from_earth(
            "cftfrac", drop_band=NON_CROPS, time_idx=-1
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

    def total_similarity(self, new_bundle, neighbour):
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
        bundle_sim = new_bundle.similarity(neighbour.behaviour.practice_bundle)
        crop_sim = self.crop_similarity(neighbour)

        # Weights from config (default: 60% bundle, 40% crop)
        w_bundle = self.get_aft_param("weight_bundle_similarity")
        w_crop = self.get_aft_param("weight_crop_similarity")

        return w_bundle * bundle_sim + w_crop * crop_sim

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Own Land)
    # =========================================================================

    def compute_attitude_own_land(self):
        """Compute attitude from own land performance vs baseline.

        "Am I doing worse than when I started this practice?"
        Compares current performance score to baseline_score (captured at transition).
        Declining performance → high attitude → more willing to transition
        Improving performance → low attitude → less willing to transition

        Uses compute_performance_score for consistency with social learning.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        # Baseline not yet set (first update not happened) → neutral
        baseline = self.performance_tracker.baseline_score

        # Get sensitivity from config
        tpb_config = self.farmer.model.config.coupled_config.tpb
        sensitivity = tpb_config.attitude_sensitivity

        # Compare current score to baseline (score at time of transition)
        current_score = compute_performance_score(self.performance_tracker, self.farmer)

        # score_diff: positive = improving, negative = declining
        score_diff = current_score - baseline

        # For attitude: declining → high attitude (want to change)
        # So we negate: -score_diff → declining becomes positive → high sigmoid
        return sigmoid(-score_diff * sensitivity)

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Social Learning)
    # =========================================================================

    def compute_attitude_social_learning_local(self, new_bundle):
        """Compute attitude from LOCAL neighbours using the proposed bundle.

        Evaluates neighbours using unified performance scoring with proper
        normalization and metric-specific trend/level weighting, weighted by
        similarity and confidence.

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Homophily (McPherson et al. 2001): similarity-weighted learning
        - Adaptive learning (Boyd & Richerson 1985): filter out worse performers
        - Yield gap analysis (van Ittersum et al. 2013): normalized comparisons

        This is the LOCAL component - uses direct neighbour comparisons with
        full similarity weighting. See compute_attitude_social_learning_country()
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
        if not self.farmer.neighbourhood:
            return 0.5  # Neutral without neighbours

        # Get parameters from config
        confidence_years = self.get_aft_param("confidence_years")

        # My unified performance score
        my_score = compute_performance_score(
            self.performance_tracker, self.farmer
        )

        # Accumulate weighted score differences
        weighted_score_diff = 0.0
        total_weight = 0.0

        for neighbour in self.farmer.neighbourhood:
            # How similar is neighbour? (bundle + crop similarity)
            similarity = self.total_similarity(new_bundle, neighbour)

            if similarity == 0:
                continue  # Skip completely different neighbours

            n_tracker = neighbour.behaviour.performance_tracker

            # Confidence: more observations → more reliable information
            n_obs = n_tracker.n
            confidence = min(1.0, n_obs / confidence_years)

            # Compute neighbour's unified performance score
            neighbour_score = compute_performance_score(
                n_tracker, self.farmer
            )

            # Score difference: positive if neighbour is better
            score_diff = neighbour_score - my_score

            # if neighbour.cell.output.cell.item() == 8:
            #     breakpoint()

            # Weight = similarity × confidence
            weight = similarity * confidence

            # Accumulate weighted score differences
            weighted_score_diff += weight * score_diff
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Neutral if no relevant neighbours

        # Normalize by total weight to get average score difference
        avg_score_diff = weighted_score_diff / total_weight

        # Scale by attitude_sensitivity before sigmoid
        # This single parameter controls how strongly score differences translate
        # to attitudes (both level and trend are already normalized to country refs)
        tpb_config = self.farmer.model.config.coupled_config.tpb
        sensitivity = tpb_config.attitude_sensitivity

        # Sigmoid maps to (0, 1) attitude score
        return sigmoid(avg_score_diff * sensitivity)

    def compute_attitude_social_learning_country(self, new_bundle):
        """Compute attitude from COUNTRY-LEVEL bundle performance.

        Uses unified performance scoring with proper normalization and
        metric-specific trend/level weighting to compare own performance
        against average performance of farmers using the proposed bundle.

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Adaptive learning (Boyd & Richerson 1985): filter out worse strategies
        - Yield gap analysis (van Ittersum et al. 2013): normalized comparisons

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
        country = self.farmer.cell.country
        store = _country_performance_store(country)
        if store is None:
            return 0.5  # Neutral if no country data

        perf = store.get(new_bundle)
        if perf.count == 0:
            return 0.5  # Neutral if no data for this bundle

        # Compute performance scores using unified scoring
        my_score = compute_performance_score(
            self.performance_tracker, self.farmer
        )
        bundle_score = compute_performance_score(perf, self.farmer)

        # Score difference as attitude input
        # Positive if bundle is better (higher score), negative if worse
        score_diff = bundle_score - my_score

        # Scale by attitude_sensitivity before sigmoid
        tpb_config = self.farmer.model.config.coupled_config.tpb
        sensitivity = getattr(tpb_config, "attitude_sensitivity", 4.0)

        # Sigmoid maps to (0, 1) attitude score
        return sigmoid(score_diff * sensitivity)

    def compute_attitude_social_learning_cluster(self, new_bundle):
        """Compute attitude from AGROECOLOGICAL CLUSTER-level bundle performance.

        Cross-border social learning: farmers learn from countries with similar
        agroecological conditions. Uses unified performance scoring with proper
        normalization and metric-specific trend/level weighting.

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Adaptive learning (Boyd & Richerson 1985): filter out worse strategies
        - Yield gap analysis (van Ittersum et al. 2013): normalized comparisons

        Uses aggregated cluster statistics from world.statistic (computed from
        peer countries in the same agroecological cluster).

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        country = self.farmer.cell.country
        cluster_id = getattr(country, "agroecological_cluster", -1)

        if cluster_id < 0:
            return 0.5  # Neutral if no cluster

        # Access via country._world which works on both driver and workers
        cluster_stats = country._world.statistic.get("cluster_management_performance", {})
        store = cluster_stats.get(cluster_id)
        if not isinstance(store, RegionManagementPerformanceStore):
            return 0.5  # Neutral if no cluster data

        perf = store.get(new_bundle)
        if perf.count == 0:
            return 0.5  # Neutral if no data for this bundle

        # Compute performance scores using unified scoring
        my_score = compute_performance_score(
            self.performance_tracker, self.farmer
        )
        bundle_score = compute_performance_score(perf, self.farmer)

        # Score difference as attitude input
        # Positive if bundle is better (higher score), negative if worse
        score_diff = bundle_score - my_score

        # Scale by attitude_sensitivity before sigmoid
        tpb_config = self.farmer.model.config.coupled_config.tpb
        sensitivity = getattr(tpb_config, "attitude_sensitivity", 4.0)

        # Sigmoid maps to (0, 1) attitude score
        return sigmoid(score_diff * sensitivity)

    # =========================================================================
    # TPB COMPONENT: SOCIAL NORM
    # =========================================================================

    def compute_social_norm_local(self, new_bundle):
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
        if not self.farmer.neighbourhood:
            return 0.5  # Neutral without neighbours

        # Average similarity to neighbours for this bundle
        # High similarity = neighbours use similar bundles and crops
        total_similarity = sum(
            self.total_similarity(new_bundle, n)
            for n in self.farmer.neighbourhood
        )
        avg_similarity = total_similarity / len(self.farmer.neighbourhood)

        # Shifted sigmoid: adoption threshold acts as the neutrality point
        # (Granovetter 1978 heterogeneous-threshold diffusion model).
        # Below threshold -> drag, above -> boost, smoothly transitioning.
        # Threshold is AFT-specific: pioneers feel "normed" at lower adoption,
        # traditionalists require broader local uptake to feel normative.
        threshold = self.get_aft_param("threshold_social_norm_local")
        return sigmoid(avg_similarity - threshold)

    def compute_social_norm_country(self, new_bundle):
        """Compute social norm based on COUNTRY-LEVEL practice distribution.

        Uses cached country statistics for O(1) lookup. This reflects
        "what farmers in my country are doing" as a broader social influence.
        Country stats are already merged with neighbouring countries
        (configured via neighbour_country_weight) at the country level.

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
        store = _country_performance_store(self.farmer.cell.country)
        if store is None or store.total_farmers < 2:
            return 0.5  # Neutral if no country data

        bundle_count = store.bundle_counts.get(new_bundle, 0)
        total_farmers = store.total_farmers

        # Fraction of farmers using this bundle
        bundle_fraction = bundle_count / total_farmers

        # Shifted sigmoid: threshold = country-adoption level at which this
        # bundle feels normative (Granovetter 1978). Thresholds differ per
        # AFT (pioneers adopt the country signal earlier) and are typically
        # higher than the local threshold, because country adoption is more
        # abstract/statistical than direct observation of neighbours.
        threshold = self.get_aft_param("threshold_social_norm_country")
        return sigmoid(bundle_fraction - threshold)

    def compute_social_norm_cluster(self, new_bundle):
        """Compute social norm from AGROECOLOGICAL CLUSTER-level practice distribution.

        Tele-coupled social norm: farmers are influenced by adoption patterns
        in countries with similar agroecological conditions. This captures how
        agricultural practices diffuse across countries within similar
        agro-ecological zones.

        Uses aggregated cluster statistics from world.statistic (computed from
        peer countries in the same agroecological cluster).

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Social norm score in [0, 1].
        """
        country = self.farmer.cell.country
        cluster_id = getattr(country, "agroecological_cluster", -1)

        if cluster_id < 0:
            return 0.5  # Neutral if no cluster

        # Access via country._world which works on both driver and workers
        cluster_stats = country._world.statistic.get("cluster_management_performance", {})
        store = cluster_stats.get(cluster_id)
        if not isinstance(store, RegionManagementPerformanceStore):
            return 0.5  # Neutral if no cluster data

        total_in_cluster = store.total_farmers
        bundle_adopters = store.bundle_counts.get(new_bundle, 0)

        if total_in_cluster == 0:
            return 0.5  # Neutral if no peer farmers

        # Adoption rate in cluster
        adoption_rate = bundle_adopters / total_in_cluster

        # Shifted sigmoid with AFT-specific threshold
        threshold = getattr(self.farmer, "threshold_social_norm_cluster", 0.1)
        return sigmoid(adoption_rate - threshold)


    # =========================================================================
    # AFFORDABILITY ADJUSTMENTMENT BASED ON CAPITAL
    # =========================================================================

    def affordable_bundle(self, target_bundle):
        """Find affordable subset of target bundle.

        If farmer can't afford full target bundle, add changes cheapest-first
        to maximize what can be adopted within capital constraints.

        See farmer.get_practice_transition_cost() for per-practice cost logic.

        Parameters
        ----------
        target_bundle : Bundle
            Desired practice bundle.

        Returns
        -------
        Bundle
            Affordable bundle (may equal current if nothing affordable).
        """
        current = self.practice_bundle
        total_cost = self.farmer.get_bundle_transition_costs(
            self.practice_bundle,
            target_bundle,
        )

        # -----------------------------------------------------------------
        # Check if full target is affordable
        # -----------------------------------------------------------------
        if self.farmer.capital >= total_cost:
            return target_bundle

        # -----------------------------------------------------------------
        # Build affordable subset: add changes cheapest-first
        # -----------------------------------------------------------------
        changes = []
        for field in PRACTICE_FIELDS:
            old_val = getattr(current, field)
            new_val = getattr(target_bundle, field)

            if old_val == new_val:
                continue

            cost = self.farmer.get_practice_transition_cost(field, old_val, new_val)
            changes.append((field, new_val, cost))

        # Sort by cost (cheapest first)
        changes.sort(key=lambda x: x[2])

        # Greedily add affordable changes
        result = current
        remaining_capital = self.farmer.capital

        for field, new_val, cost in changes:
            if remaining_capital >= cost:
                result = result.change_practices(**{field: new_val})
                remaining_capital -= cost

        return result

    # =========================================================================
    # TPB COMPONENT: PERCEIVED BEHAVIORAL CONTROL (PBC)
    # =========================================================================

    def anticipated_delta_yield(self, current_bundle, target_bundle):
        """Estimate yield change from switching bundles based on peer experience.

        Searches for farmers who have experience with BOTH bundles and computes
        the average yield difference. Searches in order: cluster → country → local,
        using the first level with sufficient data.

        Parameters
        ----------
        current_bundle : ManagementBundle
            The bundle currently being used.
        target_bundle : ManagementBundle
            The bundle being considered.

        Returns
        -------
        float
            Estimated yield change (same units as farmer.cropyield), positive if
            target yields more. Returns 0.0 if no farmers have experience with
            both bundles.

        Notes
        -----
        Remembered yields are stored as self-referenced fractional rates
        (yield / own_baseline - 1). The peer-averaged rate difference is
        converted back to absolute units using the EVALUATING farmer's own
        baseline yield, so downstream revenue anticipation stays in absolute
        yield units.
        """
        if current_bundle == target_bundle:
            return 0.0

        # Get enabled spreading levels
        enable_local, enable_country, enable_cluster = self._get_enabled_spreading_levels()

        # Search cluster first (most data), then country, then local
        farmer_lists = []

        if enable_cluster:
            # Get farmers in same agroecological cluster
            cluster_farmers = self._get_cluster_farmers()
            if cluster_farmers:
                farmer_lists.append(cluster_farmers)

        if enable_country:
            # Get farmers in same country
            country_farmers = list(self.farmer.cell.country.farmers)
            if country_farmers:
                farmer_lists.append(country_farmers)

        if enable_local:
            # Get local neighbors
            local_farmers = list(self.farmer.neighbourhood)
            if local_farmers:
                farmer_lists.append(local_farmers)

        # Search each level until we find sufficient data
        for farmers in farmer_lists:
            delta_yields = []
            for f in farmers:
                if f is self.farmer:
                    continue
                if not hasattr(f, 'behaviour') or not hasattr(f.behaviour, 'performance_memory'):
                    continue

                mem = f.behaviour.performance_memory
                current_exp = mem[current_bundle]
                target_exp = mem[target_bundle]

                # Both must have experience (duration > 0)
                if current_exp.duration > 0 and target_exp.duration > 0:
                    # Difference of fractional rates (dimensionless)
                    delta = target_exp.level_yield - current_exp.level_yield
                    delta_yields.append(delta)

            # If we found at least 3 farmers with dual experience, use this level
            if len(delta_yields) >= 3:
                mean_delta_rate = float(sum(delta_yields) / len(delta_yields))
                # Convert relative rate difference to absolute yield units using
                # the evaluating farmer's own baseline yield.
                return mean_delta_rate * self.farmer.baseline_yield

        # No sufficient data at any level
        return 0.0

    def _get_cluster_farmers(self):
        """Get all farmers in the same agroecological cluster."""
        country = self.farmer.cell.country
        cluster_id = getattr(country, 'cluster_id', -1)
        if cluster_id < 0:
            return []

        # Get all countries in this cluster
        world = country._world
        cluster_countries = getattr(world, '_cluster_countries', {}).get(cluster_id, [])

        farmers = []
        for c in cluster_countries:
            farmers.extend(list(c.farmers))
        return farmers

    def pbc_for_bundle(self, new_bundle):
        """Compute Perceived Behavioral Control for a bundle.

        PBC reflects "can I actually do this?" - lower when net financial
        impact is high relative to annual income (baseline revenue).

        Formula: PBC = pbc_base × sigmoid(ratio × pbc_sensitivity)

        Where:
        - pbc_base: AFT-specific baseline (pioneers higher, traditionalists lower)
        - ratio = -net_financial_impact / (baseline_revenue × planning_horizon)
          (positive when profitable, negative when costly)
        - pbc_sensitivity: config parameter controlling response steepness

        Using revenue over the planning horizon as reference ensures consistency:
        net_financial_impact includes planning_horizon, so we compare to income
        over the same period. Farmers assess "can I afford this over 5 years?"
        by comparing total costs/gains to expected income over those 5 years.

        Sigmoid maps the financial ratio to [0, 1]:
        - ratio = 0 (neutral): PBC = pbc_base × 0.5
        - ratio >> 0 (profitable): PBC → pbc_base
        - ratio << 0 (costly): PBC → 0

        Net financial impact considers:
        - One-time transition costs
        - Change in annual direct costs over planning horizon
        - Anticipated change in revenue (from yield differences)

        AFT differences in risk aversion are captured via pbc_base, not as a
        separate multiplicative factor (avoids double-counting).

        References
        ----------
        Li, X., Dai, J., Zhu, X., Li, J., He, J., Huang, Y., ... & Shen, Q. (2023).
        Mechanism of attitude, subjective norms, and perceived behavioral control
        influence the green development behavior of construction enterprises.
        Humanities and Social Sciences Communications, 10(1), 266.

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

        # One-time transition costs
        transition_costs = self.farmer.get_bundle_transition_costs(
            current_bundle=self.practice_bundle,
            target_bundle=new_bundle,
        )

        # Change in annual direct costs (new bundle vs current bundle)
        # Positive = new bundle costs more, Negative = new bundle saves money
        delta_direct_costs = (
            self.farmer.get_bundle_direct_costs(bundle=new_bundle)
            - self.farmer.get_bundle_direct_costs(bundle=self.practice_bundle)
        )

        # -----------------------------------------------------------------
        # Calculate anticipated revenue change from yield difference
        # -----------------------------------------------------------------
        # Peer experience gives the anticipated absolute yield change (in
        # cropyield units, gC/m2). Convert it to a whole-farm USD revenue change
        # on the SAME footing as baseline_revenue and the cost terms below:
        # express it as a fraction of the farmer's own baseline yield, then
        # apply that fraction to baseline revenue. This implicitly carries the
        # farmer's real crop/price mix and area (already embedded in
        # baseline_revenue), avoiding a separate gC->tonnes and area conversion
        # and a plain price average.
        delta_yield = self.anticipated_delta_yield(self.practice_bundle, new_bundle)
        if self.farmer.baseline_yield > 0:
            delta_revenue = (
                delta_yield / self.farmer.baseline_yield
            ) * self.farmer.baseline_revenue  # $/farm/yr
        else:
            delta_revenue = 0.0
        # -----------------------------------------------------------------
        # Total financial impact over planning horizon
        # -----------------------------------------------------------------
        # Positive = net outflow (costs exceed revenue gain)
        # Negative = net inflow (revenue gain exceeds costs)
        planning_horizon = self.farmer.behaviour.get_aft_param("min_observation_years")
        net_financial_impact = (
            transition_costs
            + delta_direct_costs * planning_horizon
            - delta_revenue * planning_horizon  # Revenue is subtracted (inflow)
        )

        # -----------------------------------------------------------------
        # PBC: smooth, bounded affordability factor from financial impact
        # -----------------------------------------------------------------
        # Scale financial impact by revenue over the planning horizon for a
        # dimensionless ratio (net_financial_impact already includes the
        # horizon). Farmers assess affordability by comparing total costs/gains
        # to expected income over that period.
        #
        #   ratio = net_financial_impact / revenue_over_horizon
        #   ratio > 0 → costly switch,  ratio < 0 → profitable switch
        #
        # The affordability factor uses the tanh sigmoid, anchored so a
        # cost-neutral switch (ratio = 0) gives factor = 1 (PBC = pbc_base):
        #
        #   financial_factor = 2 · sigmoid(-ratio) = 1 - tanh(ratio)  ∈ (0, 2)
        #
        # This matches the previous 1/(1+ratio) response to first order in the
        # normal operating range, but is bounded and monotone everywhere: it has
        # no pole and no sign flip when the anticipated gain is very large (the
        # old form went negative and collapsed PBC to 0 for gains > ~100%).
        # - costly (ratio ≫ 0):     factor → 0,  PBC → 0
        # - neutral (ratio = 0):    factor = 1,  PBC = pbc_base
        # - profitable (ratio < 0): factor > 1,  PBC → pbc_base·2, capped at 1.0
        # -----------------------------------------------------------------
        # Guard against division by zero when baseline_revenue is 0 (e.g., no crops)
        revenue_over_horizon = max(self.farmer.baseline_revenue * planning_horizon, 1.0)
        ratio = net_financial_impact / revenue_over_horizon
        financial_factor = 2.0 * sigmoid(-ratio)

        return max(0.0, min(1.0, self.farmer.pbc_base * financial_factor))  # Cap to [0, 1]


    # =========================================================================
    # COMPUTE FULL TPB SCORE
    # =========================================================================

    def compute_tpb_for_bundle(self, new_bundle):
        """Compute all TPB components and overall intention for a bundle.

        TPB formula (multiplicative):
            TPB = (w_att × Attitude + w_norm × SocialNorm) × PBC

        This means PBC acts as a gate: low PBC blocks adoption regardless
        of positive attitude/norms.

        Components are computed at local (neighbour), country, and agroecological
        cluster levels, then combined with configurable weights. Levels can be
        disabled via config (tpb), and their weights are redistributed
        to enabled levels.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.
        """
        # -----------------------------------------------------------------
        # Get spreading level config (which levels are enabled)
        # -----------------------------------------------------------------
        enable_local, enable_country, enable_cluster = self._get_enabled_spreading_levels()

        # -----------------------------------------------------------------
        # Redistribute weights for attitude social learning
        # -----------------------------------------------------------------
        raw_att_local = self.farmer.weight_attitude_local
        raw_att_country = self.farmer.weight_attitude_country
        raw_att_cluster = self.farmer.weight_attitude_cluster

        att_w_local, att_w_country, att_w_cluster = redistribute_weights(
            raw_att_local, raw_att_country, raw_att_cluster,
            enable_local, enable_country, enable_cluster,
        )

        # -----------------------------------------------------------------
        # Redistribute weights for social norm
        # -----------------------------------------------------------------
        raw_norm_local = self.farmer.weight_social_norm_local
        raw_norm_country = self.farmer.weight_social_norm_country
        raw_norm_cluster = self.farmer.weight_social_norm_cluster

        norm_w_local, norm_w_country, norm_w_cluster = redistribute_weights(
            raw_norm_local, raw_norm_country, raw_norm_cluster,
            enable_local, enable_country, enable_cluster,
        )

        # -----------------------------------------------------------------
        # Attitude: own experience + social learning (local + country + cluster)
        # -----------------------------------------------------------------
        # Own land attitude (always computed - it's your own observation)
        self._attitude_own_land = self.compute_attitude_own_land()

        # Social learning: compute only enabled levels
        self._attitude_social_learning_local = (
            self.compute_attitude_social_learning_local(new_bundle)
            if enable_local else 0.0
        )
        self._attitude_social_learning_country = (
            self.compute_attitude_social_learning_country(new_bundle)
            if enable_country else 0.0
        )
        self._attitude_social_learning_cluster = (
            self.compute_attitude_social_learning_cluster(new_bundle)
            if enable_cluster else 0.0
        )

        # Combine with redistributed weights
        self._attitude_social_learning = (
            att_w_local * self._attitude_social_learning_local
            + att_w_country * self._attitude_social_learning_country
            + att_w_cluster * self._attitude_social_learning_cluster
        )

        # Full attitude: own land + combined social learning
        self._attitude = (
            self.farmer.weight_own_land * self._attitude_own_land
            + self.farmer.weight_social_learning * self._attitude_social_learning
        )

        # -----------------------------------------------------------------
        # Social Norm: local (neighbours) + country-level + agroecological cluster
        # -----------------------------------------------------------------
        # Compute only enabled levels
        self._social_norm_local = (
            self.compute_social_norm_local(new_bundle)
            if enable_local else 0.0
        )
        self._social_norm_country = (
            self.compute_social_norm_country(new_bundle)
            if enable_country else 0.0
        )
        self._social_norm_cluster = (
            self.compute_social_norm_cluster(new_bundle)
            if enable_cluster else 0.0
        )

        # Combine with redistributed weights
        self._social_norm = (
            norm_w_local * self._social_norm_local
            + norm_w_country * self._social_norm_country
            + norm_w_cluster * self._social_norm_cluster
        )

        # -----------------------------------------------------------------
        # PBC: can I afford this?
        # -----------------------------------------------------------------
        self._pbc = self.pbc_for_bundle(new_bundle)

        # -----------------------------------------------------------------
        # TPB Intention: combine components
        # -----------------------------------------------------------------
        # Multiplicative formula: PBC gates the attitude/norm contribution
        self._tpb = (
            self.farmer.weight_attitude * self._attitude
            + self.farmer.weight_norm * self._social_norm
        ) * self._pbc
