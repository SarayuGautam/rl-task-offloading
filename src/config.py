# =============================================================================
# config.py
# All simulation and learning hyperparameters in one place.
# Change values here — do NOT scatter magic numbers across files.
# =============================================================================

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------
SIM_DURATION = 10_000          # total simulation time (seconds)
TASK_ARRIVAL_RATE = 4.0        # average tasks per second — creates realistic queue pressure

# Task properties (uniform random between min and max)
TASK_SIZE_MIN = 1_000          # bits
TASK_SIZE_MAX = 10_000         # bits
TASK_COMPLEXITY_MIN = 100      # CPU cycles (millions)
TASK_COMPLEXITY_MAX = 1_000    # CPU cycles (millions)

# -----------------------------------------------------------------------------
# Device (Tier 1 — local)
# -----------------------------------------------------------------------------
DEVICE_CPU_SPEED = 500         # million CPU cycles per second
DEVICE_POWER = 0.5             # Watts (active processing power)

# -----------------------------------------------------------------------------
# Edge server (Tier 2)
# -----------------------------------------------------------------------------
EDGE_CPU_SPEED = 5_000         # million CPU cycles per second
EDGE_BANDWIDTH = 20e6          # bits per second (20 Mbps wireless)
EDGE_PROPAGATION_DELAY = 0.005 # seconds (5 ms)
EDGE_QUEUE_CAPACITY = 10       # max tasks waiting in queue

# -----------------------------------------------------------------------------
# Cloud server (Tier 3)
# -----------------------------------------------------------------------------
CLOUD_CPU_SPEED = 10_000       # million CPU cycles per second
CLOUD_BANDWIDTH = 5e6          # bits per second (5 Mbps backhaul — bottleneck)
CLOUD_PROPAGATION_DELAY = 0.08 # seconds (80 ms)

# -----------------------------------------------------------------------------
# Energy model
# Energy = Power * Time  (Joules)
# Transmission energy uses a fixed transmission power constant.
# -----------------------------------------------------------------------------
TRANSMISSION_POWER = 0.3       # Watts (radio transmission power)

# -----------------------------------------------------------------------------
# Q-Learning hyperparameters
# -----------------------------------------------------------------------------
LEARNING_RATE = 0.1            # alpha — how fast to update Q values
DISCOUNT_FACTOR = 0.9          # gamma — how much to value future rewards
EPSILON_START = 1.0            # start fully exploring
EPSILON_END = 0.05             # minimum exploration rate
EPSILON_DECAY = 0.995          # multiply epsilon by this each episode
NUM_EPISODES = 1_000           # training episodes

# -----------------------------------------------------------------------------
# Reward function weights
# reward = -(W_LATENCY * latency + W_ENERGY * energy)
# -----------------------------------------------------------------------------
W_LATENCY = 0.7                # latency is more important
W_ENERGY = 0.3

# -----------------------------------------------------------------------------
# State discretization bins
# The Q-table maps discrete states to Q-values.
# -----------------------------------------------------------------------------
QUEUE_BINS = [0, 3, 7]         # edge queue: [low, medium, high]
TASK_SIZE_BINS = [3_000, 7_000]# task size: [small, medium, large]
NETWORK_QUALITY_BINS = [0.4]   # channel quality 0–1: [poor, good]

# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------
ACTION_LOCAL = 0
ACTION_EDGE  = 1
ACTION_CLOUD = 2
ACTION_NAMES = {0: "Local", 1: "Edge", 2: "Cloud"}
