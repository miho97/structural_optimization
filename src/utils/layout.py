"""Observation channel layout shared by the environment and the policy.

Every observation has shape ``(NUM_CHANNELS, height + 1, width + 1)``. Cell
quantities occupy the top-left ``height x width`` block and are zero in the
padded last row and column. Nodal quantities use the full image. The three
"plane" channels hold one constant value everywhere.
"""

LAYOUT_VERSION = 2

# Cell state, one-hot.
UNDECIDED = 0
SOLID = 1
VOID = 2
# Nodal boundary conditions.
SUPPORT_X = 3
SUPPORT_Y = 4
FORCE_X = 5
FORCE_Y = 6
# Cell selected for removal by the first half of a refinement swap.
PENDING = 7
# Element strain energy from the latest solve, log-scaled to [-1, 1].
ENERGY = 8
# Constant planes.
BUDGET_LEFT = 9   # (solid budget - solid cells) / total cells
SWAPS_LEFT = 10   # refinement swaps left / maximum swaps
TARGET = 11       # solid budget / total cells

NUM_CHANNELS = 12

BOUNDARY_CHANNELS = (SUPPORT_X, SUPPORT_Y, FORCE_X, FORCE_Y)
TOPOLOGY_CHANNELS = (
    UNDECIDED,
    SOLID,
    VOID,
    PENDING,
    ENERGY,
    BUDGET_LEFT,
    SWAPS_LEFT,
    TARGET,
)
