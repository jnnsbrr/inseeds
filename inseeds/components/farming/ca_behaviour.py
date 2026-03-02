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
# 0 = practice OFF (conventional tillage, no cover crop, baseline residue)
# 1 = practice ON (no-till, cover crop planted, residue retained)

BUNDLE_NAMES = {
    (0, 0, 0): "conventional",        # All conventional practices
    (0, 0, 1): "residue_only",        # Only residue retention
    (0, 1, 0): "cover_crop_only",     # Only cover crops
    (0, 1, 1): "cover_crop_residue",  # Cover crops + residue
    (1, 0, 0): "notill_only",         # Only no-till (risky without residue)
    (1, 0, 1): "notill_residue",      # No-till + residue (common CA entry)
    (1, 1, 0): "notill_cover_crop",   # No-till + cover (needs residue ideally)
    (1, 1, 1): "conservation",        # Full Conservation Agriculture
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
        # Each element is 0 (off) or 1 (on)

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
        # Snapshot current state for trend computation
        # -----------------------------------------------------------------
        # When farmer switches practices, we record soil C, moisture, yield
        # to compute trends (annual rate of change) under the new bundle

        t_start = agent.model.lpjml.sim_year

        self.current_state = {
            "t_start": t_start,              # Year of last switch
            "soilc_start": agent.soilc,      # Soil carbon at switch (gC/m²)
            "moisture_start": agent.root_moisture,  # Root zone moisture
            "yield_start": agent.cropyield,  # Crop yield at switch
            "baseline_score": 0.0,           # Weighted score at switch (for fallback)
        }

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

        Returns
        -------
        dict
            Keys: 'soil', 'moisture', 'yield'. Values: annual rate of change.
        """
        mem = self.current_state
        duration = self.agent.model.lpjml.sim_year - mem["t_start"]

        # Need at least 1 year to compute trend
        if duration < 1:
            return {"soil": 0.0, "moisture": 0.0, "yield": 0.0}

        # Annual rate of change = (current - start) / years
        return {
            "soil": (self.agent.soilc - mem["soilc_start"]) / duration,
            "moisture": (self.agent.root_moisture - mem["moisture_start"]) / duration,
            "yield": (self.agent.cropyield - mem["yield_start"]) / duration,
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
        duration = current_year - self.current_state["t_start"]

        # -----------------------------------------------------------------
        # Store outcome of outgoing bundle (if used for at least 1 year)
        # -----------------------------------------------------------------
        if duration > 0:
            trend = self.current_trend

            # Preserve failure count from previous memory
            old_fc = self.bundle_memory[self._practice_bundle].get("failure_count", 0)

            # Update memory for the bundle we're leaving
            self.bundle_memory[self._practice_bundle] = {
                "trend_soil": trend["soil"],
                "trend_moisture": trend["moisture"],
                "trend_yield": trend["yield"],
                "duration": duration,
                "last_updated": current_year,
                "failure_count": old_fc,
            }

        # -----------------------------------------------------------------
        # Prepare for new bundle
        # -----------------------------------------------------------------

        # Remember previous bundle (for potential fallback)
        self._previous_bundle = self._practice_bundle

        # Reset state snapshot for new bundle
        self.current_state = {
            "t_start": current_year,
            "soilc_start": self.agent.soilc,
            "moisture_start": self.agent.root_moisture,
            "yield_start": self.agent.cropyield,
            "baseline_score": self._weighted_score(self.current_trend),
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
        memory_decay = getattr(self.agent, "memory_decay_years", DEFAULT_MEMORY_DECAY_YEARS)

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

    def _get_valid_memory(self, bundle):
        """Return memory for bundle if recent enough, else None.

        Parameters
        ----------
        bundle : tuple
            Practice bundle to look up.

        Returns
        -------
        dict or None
            Memory dict if valid, None if no memory or too old.
        """
        mem = self.bundle_memory.get(bundle, {})

        # No memory recorded
        if mem.get("duration", 0) == 0:
            return None

        # Memory too old
        memory_decay = getattr(self.agent, "memory_decay_years", DEFAULT_MEMORY_DECAY_YEARS)
        if self.agent.model.lpjml.sim_year - mem.get("last_updated", 0) > memory_decay:
            return None

        return mem

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
        1. Decay old memories (bounded rationality)
        2. Check if minimum observation period has passed
        3. Check fallback condition (sustained decline → revert)
        4. Find best-performing neighbour's bundle OR explore randomly
        5. Adjust target bundle for affordability
        6. Compute TPB scores for the proposed bundle
        """
        # -----------------------------------------------------------------
        # Step 1: Decay old memories
        # -----------------------------------------------------------------
        self._decay_old_memories()

        duration = self.agent.model.lpjml.sim_year - self.current_state["t_start"]

        # -----------------------------------------------------------------
        # Step 2: Need at least 1 year of data
        # -----------------------------------------------------------------
        if duration < 1:
            self._tpb = 0.0
            self._proposed_bundle = None
            return

        # -----------------------------------------------------------------
        # Step 3: Require minimum observation years before switching
        # -----------------------------------------------------------------
        # Avoids noisy decisions based on single-year fluctuations
        # Typical value: 3 years (allows trends to stabilize)

        min_obs = getattr(self.agent, "min_observation_years", 3)

        if duration < min_obs:
            self._tpb = 0.0
            self._proposed_bundle = None
            return

        # -----------------------------------------------------------------
        # Step 4: Check fallback condition
        # -----------------------------------------------------------------
        # If performance has declined for FALLBACK_YEARS consecutive years,
        # propose reverting to the previous bundle (adaptive management)

        if self._check_fallback():
            return  # Fallback sets _proposed_bundle and _tpb internally

        # -----------------------------------------------------------------
        # Step 5: Find target bundle (neighbour imitation or exploration)
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
        # Step 6: Adjust for affordability
        # -----------------------------------------------------------------
        # If farmer can't afford full target bundle, find affordable subset

        affordable_bundle = self._affordable_bundle(target_bundle)

        # If affordable bundle is same as current, no change
        if affordable_bundle == self._practice_bundle:
            self._tpb = 0.0
            self._proposed_bundle = None
            return

        # -----------------------------------------------------------------
        # Step 7: Compute TPB scores for proposed bundle
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
        duration = self.agent.model.lpjml.sim_year - self.current_state["t_start"]
        grace_period = getattr(self.agent, "min_observation_years", 3)
        if duration < grace_period:
            return False

        # -----------------------------------------------------------------
        # Compare current performance to baseline at switch time
        # -----------------------------------------------------------------
        baseline = self.current_state.get("baseline_score", 0)
        current_score = self._weighted_score(self.current_trend)

        # Track consecutive years of decline
        if current_score < baseline:
            self._decline_years += 1
        else:
            self._decline_years = 0  # Reset if performance improves

        # -----------------------------------------------------------------
        # Trigger fallback after sustained decline
        # -----------------------------------------------------------------
        fallback_years = getattr(self.agent, "fallback_years", DEFAULT_FALLBACK_YEARS)
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
        # No-till alone (no cover, no residue): soil exposed
        if t == 1 and c == 0 and r == 0:
            # Only unreasonable if residue is cheap (could easily retain it)
            if self.agent.residue_opportunity_cost_per_ha < residue_threshold:
                return False

        # No-till + cover but no residue: still risky in off-season
        if t == 1 and c == 1 and r == 0:
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

        # Base probability depends on farmer type
        # Pioneers (innovators) are more willing to experiment
        base_prob = 0.05 if self.agent.aft.name == "pioneer" else 0.01

        # Poor performers explore more (searching for better options)
        current_score = self._weighted_score(self.current_trend)
        normalized = self._normalize_score(current_score)
        if normalized < 0.3:
            base_prob *= 2.0

        # Experience affects willingness to explore (smooth ramp based on confidence_years)
        # Early: 0.5x exploration (cautious), Experienced: 1.5x exploration (confident)
        duration = self.agent.model.lpjml.sim_year - self.current_state["t_start"]
        confidence_years = getattr(self.agent, "confidence_years", DEFAULT_CONFIDENCE_YEARS)
        experience_factor = min(1.0, duration / confidence_years)
        exploration_modifier = 0.5 + experience_factor * 1.0  # Ramps from 0.5 to 1.5
        base_prob *= exploration_modifier

        # Cap exploration probability
        explore_prob = min(base_prob, 0.15)

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
            threshold = getattr(self.agent, "revert_threshold", 0.6)
        else:
            threshold = getattr(self.agent, "switch_threshold", 0.5)

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
        # Tillage: 0 = conventional, 1 = no-till
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
        # Get crop fractions
        cft_self = self.agent.cell.from_earth.cftfrac.values.flatten()
        cft_neighbour = neighbour.cell.from_earth.cftfrac.values.flatten()

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
        w_bundle = getattr(self.agent, "weight_bundle_similarity", 0.6)
        w_crop = getattr(self.agent, "weight_crop_similarity", 0.4)

        return w_bundle * bundle_sim + w_crop * crop_sim

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Social Learning)
    # =========================================================================

    def _attitude_social_learning(self, new_bundle):
        """Compute attitude from neighbours using similar bundles and crops.

        Social learning (Bandura 1977): farmers learn from observing
        neighbours who use similar practices. Weight by:
        - Similarity: bundle + crop similarity (more similar = more informative)
        - Confidence: longer use → more reliable signal

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

        weighted_sum = 0.0
        total_weight = 0.0

        # Get confidence_years from config (default 10)
        confidence_years = getattr(self.agent, "confidence_years", 10)

        for neighbour in self.agent.neighbourhood:
            # How similar is neighbour? (bundle + crop similarity)
            similarity = self._total_similarity(new_bundle, neighbour)

            if similarity == 0:
                continue  # Skip completely different neighbours

            # Confidence: longer use → more reliable information
            n_duration = (
                neighbour.behaviour.agent.model.lpjml.sim_year
                - neighbour.behaviour.current_state["t_start"]
            )
            confidence = min(1.0, n_duration / confidence_years)

            # neighbour's performance score
            n_score = self._weighted_score(neighbour.behaviour.current_trend)

            # Weight = similarity × confidence
            weight = similarity * confidence
            weighted_sum += weight * n_score
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Neutral if no relevant neighbours

        # Normalize the weighted average
        return self._normalize_score(weighted_sum / total_weight)

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
        if new_bundle == most_common:
            boost = homogeneity * 0.2  # Conformity bonus
        else:
            boost = -homogeneity * 0.1  # Non-conformity penalty

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
        """Compute risk factor from AFT type and capital volatility.

        Risk aversion (Chavas & Holt 1996): farmers weight potential losses
        more heavily than equivalent gains. Higher risk factor means more
        cautious behavior and lower PBC.

        Components
        ----------
        1. Base risk aversion: AFT parameter (traditionalist > pioneer)
        2. Capital volatility: coefficient of variation over recent years

        Returns
        -------
        float
            Risk factor in [0, 1]. Higher = more risk-averse = lower PBC.

        References
        ----------
        Chavas, J.P. & Holt, M.T. (1996). Economic behavior under uncertainty.
        """
        # AFT base risk aversion (0 = risk-neutral, 1 = very risk-averse)
        base_risk = getattr(self.agent, "risk_aversion", 0.2)

        # Capital volatility: CV = std / mean over recent years
        capital_history = getattr(self.agent, "capital_history", [])

        if len(capital_history) >= 3:
            mean_capital = np.mean(capital_history)
            std_capital = np.std(capital_history)
            cv = std_capital / max(mean_capital, 1e-6)
            volatility_risk = min(cv, 1.0)  # Cap at 1.0
        else:
            volatility_risk = 0.0  # Not enough data

        # Combine: base risk + volatility contribution (weighted by 0.5)
        # Volatility can add up to 50% more risk on top of base
        return min(1.0, base_risk + (1 - base_risk) * volatility_risk * 0.5)

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

        own_memory = self._get_valid_memory(new_bundle)

        # Get confidence_years from config
        confidence_years = getattr(self.agent, "confidence_years", DEFAULT_CONFIDENCE_YEARS)

        if own_memory:
            # Own experience: confidence grows with duration
            duration = own_memory.get("duration", 1)
            confidence = min(1.0, duration / confidence_years)

            # Reconstruct trend from memory
            trend_from_memory = {
                k: own_memory["trend_" + k]
                for k in ["soil", "moisture", "yield"]
            }
            att_own_raw = self._weighted_score(trend_from_memory)

            # Blend own experience with neutral (0.5) based on confidence
            att_own = confidence * self._normalize_score(att_own_raw) + (1 - confidence) * 0.5
        else:
            # No prior experience: neutral attitude
            att_own = 0.5

        # Social learning component
        att_social = self._attitude_social_learning(new_bundle)

        # -----------------------------------------------------------------
        # Weighted combination (same approach as tillage_farmer.py)
        # -----------------------------------------------------------------
        # Weights serve dual purpose: relative importance + variance compensation.
        # If att_own has lower variance than att_social, increase weight_own_land
        # to amplify its contribution. Calibrate weights empirically.
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
