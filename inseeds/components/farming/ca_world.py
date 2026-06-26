"""CA-specific World entity with static reference scales."""

import numpy as np

from inseeds.components.farming.world import World


class CAWorld(World):
    """World entity with static reference scales for performance scoring.

    Computes world-level mean values ONCE at initialization from historic data.
    These serve as fixed scaling factors for the entire simulation, ensuring:
    - Different metrics (yield, soilC, moisture) become comparable
    - Absolute performance is preserved (higher projected endpoint = higher score)
    - No dynamic normalization that distorts farmer-level comparisons
    """

    def compute_reference_scales(self):
        """Compute static reference scales from initial farmer data.

        Called ONCE during model initialization, after farmers are created
        but before simulation starts. Uses world mean of each metric as the
        scaling factor.

        The projected endpoint formula is:
            score = (level + trend × horizon) / scale

        This makes a score of 1.0 correspond to "at initial world average".
        Scores >1.0 mean above average, <1.0 mean below average.

        Results stored in world.statistic["reference_scales"] as fixed constants
        for the entire simulation.
        """
        yields, soilcs, moistures = [], [], []

        for farmer in self.farmers:
            # Collect initial levels from all farmers
            if hasattr(farmer, "cropyield") and farmer.cropyield > 0:
                yields.append(farmer.cropyield)
            if hasattr(farmer, "soilc") and farmer.soilc > 0:
                soilcs.append(farmer.soilc)
            if hasattr(farmer, "root_moisture") and farmer.root_moisture > 0:
                moistures.append(farmer.root_moisture)

        # Compute world means (with fallback to reasonable defaults)
        scale_yield = float(np.mean(yields))
        scale_soilc = float(np.mean(soilcs))
        scale_moisture = float(np.mean(moistures))

        # Store as FIXED constants for entire simulation
        self.statistic.set("reference_scales", {
            "yield": scale_yield,
            "soilc": scale_soilc,
            "moisture": scale_moisture,
        })
