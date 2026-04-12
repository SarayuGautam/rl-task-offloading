# =============================================================================
# environment/network_model.py
#
# Mathematical models for transmission delay and energy.
# These are the core formulas from Module 1 of the curriculum.
#
# KEY FORMULAS:
#   Transmission delay  = data_size / data_rate          (seconds)
#   Propagation delay   = distance / speed_of_light      (constant per tier)
#   Computation time    = cpu_cycles / cpu_speed          (seconds)
#   Transmission energy = transmission_power * tx_time   (Joules)
#   Computation energy  = compute_power * compute_time   (Joules)
# =============================================================================

import numpy as np
from src.config import (
    EDGE_BANDWIDTH, EDGE_PROPAGATION_DELAY,
    CLOUD_BANDWIDTH, CLOUD_PROPAGATION_DELAY,
    DEVICE_CPU_SPEED, DEVICE_POWER,
    EDGE_CPU_SPEED,
    CLOUD_CPU_SPEED,
    TRANSMISSION_POWER,
)


def transmission_delay(size_bits: float, bandwidth_bps: float) -> float:
    """
    Time to push `size_bits` onto the wire.

    Args:
        size_bits     : data size in bits
        bandwidth_bps : channel bandwidth in bits per second

    Returns:
        transmission delay in seconds
    """
    return size_bits / bandwidth_bps


def compute_time(complexity_mcycles: float, cpu_speed_mcycles_per_sec: float) -> float:
    """
    Time for a server/device to execute `complexity_mcycles` million CPU cycles.

    Args:
        complexity_mcycles         : task workload (million CPU cycles)
        cpu_speed_mcycles_per_sec  : server speed (million CPU cycles/second)

    Returns:
        computation time in seconds
    """
    return complexity_mcycles / cpu_speed_mcycles_per_sec


def transmission_energy(size_bits: float, bandwidth_bps: float) -> float:
    """
    Energy consumed transmitting data wirelessly.

    E = P_tx * t_tx

    Args:
        size_bits     : data size in bits
        bandwidth_bps : wireless channel bandwidth

    Returns:
        energy in Joules
    """
    t_tx = transmission_delay(size_bits, bandwidth_bps)
    return TRANSMISSION_POWER * t_tx


def compute_energy(complexity_mcycles: float, cpu_speed: float, power_watts: float) -> float:
    """
    Energy consumed during local/edge computation.

    E = P_compute * t_compute

    Returns:
        energy in Joules
    """
    t_compute = compute_time(complexity_mcycles, cpu_speed)
    return power_watts * t_compute


# =============================================================================
# Per-tier cost functions
# Each returns (total_latency_seconds, total_energy_joules)
# =============================================================================

def local_cost(task_size_bits: float, complexity_mcycles: float):
    """
    Process task on the local device.
    No transmission — only computation cost.
    """
    latency = compute_time(complexity_mcycles, DEVICE_CPU_SPEED)
    energy  = compute_energy(complexity_mcycles, DEVICE_CPU_SPEED, DEVICE_POWER)
    return latency, energy


def edge_cost(task_size_bits: float, complexity_mcycles: float, queue_wait: float = 0.0):
    """
    Offload task to nearby edge server.

    Total latency = tx_delay + propagation + queue_wait + compute_time

    Args:
        queue_wait : extra waiting time if edge server is busy (seconds)
    """
    tx_delay  = transmission_delay(task_size_bits, EDGE_BANDWIDTH)
    compute   = compute_time(complexity_mcycles, EDGE_CPU_SPEED)
    latency   = tx_delay + EDGE_PROPAGATION_DELAY + queue_wait + compute
    energy    = transmission_energy(task_size_bits, EDGE_BANDWIDTH)
    return latency, energy


def cloud_cost(task_size_bits: float, complexity_mcycles: float):
    """
    Offload task to distant cloud server.
    No queue — cloud has effectively infinite capacity.

    Total latency = tx_delay + propagation (large!) + compute_time
    """
    tx_delay  = transmission_delay(task_size_bits, CLOUD_BANDWIDTH)
    compute   = compute_time(complexity_mcycles, CLOUD_CPU_SPEED)
    latency   = tx_delay + CLOUD_PROPAGATION_DELAY + compute
    energy    = transmission_energy(task_size_bits, CLOUD_BANDWIDTH)
    return latency, energy
