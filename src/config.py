# =============================================================================
# config.py - All hyperparameters in one place
# Month 2: Added VELOCITY_BINS and QUALITY_TREND_BINS for mobility extension
# =============================================================================

RANDOM_SEED = 42
SIM_DURATION = 10_000
TASK_ARRIVAL_RATE = 6.0

# ── Seeding scheme ────────────────────────────────────────────────────────
# NOTE (fix, Sep 2026): training episode `ep` of run `seed` previously used
# simulation seed (seed + ep). With SEEDS = [42, 123, 456, 789, 999] and 2,000
# episodes, run 42 covered sim-seeds 42..2041 and run 123 covered 123..2122 -
# an overlap of 1,919 of 2,000 episodes (96%). Every pair of "independent"
# seeds shared between 52% and 96% of its training workload, so the five runs
# were near-duplicates and the paired t-test's independence assumption did not
# hold. Striding the seed space keeps the five training ranges disjoint.
SEED_STRIDE      = 10_000   # > TRAIN_EPISODES, so ranges cannot overlap
EVAL_SEED_OFFSET = 5_000    # > TRAIN_EPISODES, so eval is disjoint from training


def train_sim_seed(seed: int, episode: int) -> int:
    """Simulation seed for training episode `episode` of run `seed`."""
    return seed * SEED_STRIDE + episode


def eval_sim_seed(seed: int) -> int:
    """Evaluation seed for run `seed`, disjoint from its training range."""
    return seed * SEED_STRIDE + EVAL_SEED_OFFSET

TASK_SIZE_MIN = 1_000
TASK_SIZE_MAX = 10_000
TASK_COMPLEXITY_MIN = 100
TASK_COMPLEXITY_MAX = 1_000

# Device (Tier 1)
DEVICE_CPU_SPEED = 500
DEVICE_POWER     = 0.5

# Edge (Tier 2)
EDGE_CPU_SPEED         = 8_000
EDGE_PROPAGATION_DELAY = 0.005
# NOTE: never enforced - simpy.Resource queues are unbounded. Kept only as a
# documented design intent; see edge_server.py. Not referenced by any code.
# EDGE_QUEUE_CAPACITY  = 10

# Cloud (Tier 3)
CLOUD_CPU_SPEED         = 30_000
CLOUD_PROPAGATION_DELAY = 0.08

TRANSMISSION_POWER = 0.3

# ── Wireless channel - Shannon-Hartley model ──────────────────────────────
# Achievable uplink data rate depends on signal quality:
#   rate = B * log2(1 + SNR)        (Shannon-Hartley theorem)
# SNR is derived from network_quality (0.1 = poor signal .. 1.0 = great).
# This makes the network_quality state variable actually affect cost:
#   poor signal -> low rate -> high transmission delay & energy.
CHANNEL_BANDWIDTH_HZ = 1e6     # 1 MHz uplink bandwidth (B)
SNR_DB_MIN           = -10.0   # quality=0.1 -> ~137 kbps  (very poor signal)
SNR_DB_MAX           = 30.0    # quality=1.0 -> ~10 Mbps   (excellent signal)

# ── Network-quality sampling range for TRAINING episodes ──────────────────
# Each training episode draws one network quality from U(MIN, MAX).
#
# NOTE (fix, Aug 2026): this range was previously (0.5, 1.0), which never
# produced a quality below the first NETWORK_QUALITY_BINS boundary (0.4).
# The whole net_bin=0 ("poor signal") slice of the state space was therefore
# NEVER VISITED during training - the agent stored only 18 of 36 states, and
# any greedy query against an unvisited state returned argmax([0,0,0]) = Local.
# That produced a spurious "the agent falls back to local under poor signal"
# reading of the Q-table which was an artefact of zero-initialisation, not
# learned behaviour. Sampling the full range fixes the coverage gap.
QUALITY_TRAIN_MIN = 0.1
QUALITY_TRAIN_MAX = 1.0

# Fixed network quality used for ALL greedy evaluation runs, so that the
# headline comparison, the 5-seed statistics and the Pareto sweep are all
# measured under identical, documented channel conditions.
EVAL_NETWORK_QUALITY = 0.9

# Q-Learning
LEARNING_RATE   = 0.15
DISCOUNT_FACTOR = 0.9
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.998
NUM_EPISODES    = 2_000

# Reward weights
W_LATENCY = 0.7
W_ENERGY  = 0.3

# ── Month 1 state bins ────────────────────────────────────────────────────
# State = (queue_bin, size_bin, net_quality_bin) → 36 states
QUEUE_BINS           = [1, 3, 6]
TASK_SIZE_BINS       = [3_000, 7_000]
NETWORK_QUALITY_BINS = [0.4, 0.75]

# ── Month 2: Mobility extension bins ─────────────────────────────────────
# velocity_bin: how fast device is moving (m/s normalised 0-1)
#   0 = stationary/slow (<0.3)  1 = walking (0.3-0.7)  2 = vehicle (>0.7)
# quality_trend_bin: is network quality improving or degrading?
#   0 = declining (< -0.05 change)  1 = stable  2 = improving (> +0.05)
#
# New state = (queue_bin, size_bin, net_quality_bin, velocity_bin, quality_trend_bin)
# Total states: 4 × 3 × 3 × 3 × 3 = 324  ← still manageable for Q-table
VELOCITY_BINS      = [0.3, 0.7]
QUALITY_TREND_BINS = [-0.05, 0.05]

# Actions
ACTION_LOCAL = 0
ACTION_EDGE  = 1
ACTION_CLOUD = 2
ACTION_NAMES = {0: "Local", 1: "Edge", 2: "Cloud"}
