# =============================================================================
# config.py — All hyperparameters in one place
# =============================================================================

RANDOM_SEED = 42

# Simulation
SIM_DURATION    = 10_000
TASK_ARRIVAL_RATE = 6.0       # tasks/sec — creates 41% edge utilisation

TASK_SIZE_MIN    = 1_000      # bits
TASK_SIZE_MAX    = 10_000
TASK_COMPLEXITY_MIN = 100     # million CPU cycles
TASK_COMPLEXITY_MAX = 1_000

# Device (Tier 1)
DEVICE_CPU_SPEED = 500
DEVICE_POWER     = 0.5        # Watts

# Edge (Tier 2) — fast but single-server queue
# Utilisation = 6 * (550/8000) = 41% → queue forms regularly
EDGE_CPU_SPEED        = 8_000         # mcycles/sec
EDGE_BANDWIDTH        = 20e6          # 20 Mbps
EDGE_PROPAGATION_DELAY = 0.005        # 5 ms
EDGE_QUEUE_CAPACITY   = 10

# Cloud (Tier 3) — very fast CPU, good BW, but 100ms propagation
# Cloud is better when edge queue wait > ~50ms
CLOUD_CPU_SPEED        = 30_000       # mcycles/sec
CLOUD_BANDWIDTH        = 20e6         # 20 Mbps
CLOUD_PROPAGATION_DELAY = 0.1         # 100 ms

# Energy model
TRANSMISSION_POWER = 0.3              # Watts

# Q-Learning
LEARNING_RATE  = 0.15
DISCOUNT_FACTOR = 0.9
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.998                # reaches min ~ep 1500
NUM_EPISODES   = 2_000

# Reward = -(W_LATENCY * latency + W_ENERGY * energy)
W_LATENCY = 0.7
W_ENERGY  = 0.3

# State discretisation: (queue_bin, size_bin, net_bin)
# queue: 0=empty  1-2=light  3-5=busy  6+=saturated
QUEUE_BINS          = [1, 3, 6]
TASK_SIZE_BINS      = [3_000, 7_000]
NETWORK_QUALITY_BINS = [0.4, 0.75]

# Actions
ACTION_LOCAL = 0
ACTION_EDGE  = 1
ACTION_CLOUD = 2
ACTION_NAMES = {0: "Local", 1: "Edge", 2: "Cloud"}
