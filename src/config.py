# =============================================================================
# config.py - All hyperparameters in one place
# Month 2: Added VELOCITY_BINS and QUALITY_TREND_BINS for mobility extension
# =============================================================================

RANDOM_SEED = 42
SIM_DURATION = 10_000
TASK_ARRIVAL_RATE = 6.0

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
EDGE_QUEUE_CAPACITY    = 10

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
