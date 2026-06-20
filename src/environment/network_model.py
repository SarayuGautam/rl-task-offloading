# =============================================================================
# environment/network_model.py
#
# Mathematical models for transmission delay and energy.
# These are the core formulas from Module 1 of the curriculum.
#
# KEY FORMULAS:
#   Shannon data rate   = B * log2(1 + SNR)              (bits/sec)
#   Transmission delay  = data_size / data_rate          (seconds)
#   Propagation delay   = distance / speed_of_light      (constant per tier)
#   Computation time    = cpu_cycles / cpu_speed          (seconds)
#   Transmission energy = transmission_power * tx_time   (Joules)
#   Computation energy  = compute_power * compute_time   (Joules)
#
# The data rate is NOT constant - it follows the Shannon-Hartley theorem and
# depends on signal quality (SNR). A weak signal gives a lower rate, so the
# same task takes longer (and costs more energy) to transmit. This is what
# makes the agent's `network_quality` state actually matter.
# =============================================================================

import numpy as np
from src.config import (
    EDGE_PROPAGATION_DELAY,
    CLOUD_PROPAGATION_DELAY,
    DEVICE_CPU_SPEED, DEVICE_POWER,
    EDGE_CPU_SPEED,
    CLOUD_CPU_SPEED,
    TRANSMISSION_POWER,
    CHANNEL_BANDWIDTH_HZ, SNR_DB_MIN, SNR_DB_MAX,
)


def snr_from_quality(quality: float) -> float:
    """
    Map network quality (0.1 = poor .. 1.0 = great) to a linear SNR.

    Quality is linearly interpolated on a decibel scale between SNR_DB_MIN
    and SNR_DB_MAX, then converted to linear power ratio: SNR = 10^(dB/10).
    """
    q = min(max(quality, 0.1), 1.0)
    snr_db = SNR_DB_MIN + (q - 0.1) / (1.0 - 0.1) * (SNR_DB_MAX - SNR_DB_MIN)
    return 10.0 ** (snr_db / 10.0)


def shannon_rate(quality: float, bandwidth_hz: float = CHANNEL_BANDWIDTH_HZ) -> float:
    """
    Shannon-Hartley achievable data rate (bits/sec) for the given quality:
        rate = B * log2(1 + SNR)
    """
    return bandwidth_hz * np.log2(1.0 + snr_from_quality(quality))


def transmission_delay(size_bits: float, quality: float) -> float:
    """
    Time to transmit `size_bits` over a wireless channel of the given quality.

    Args:
        size_bits : data size in bits
        quality   : network quality 0.1 (poor) .. 1.0 (great)

    Returns:
        transmission delay in seconds (= size_bits / Shannon rate)
    """
    return size_bits / shannon_rate(quality)


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


def transmission_energy(size_bits: float, quality: float) -> float:
    """
    Energy consumed transmitting data wirelessly.

    E = P_tx * t_tx

    A poor signal means a longer transmission, so more energy is spent - the
    energy cost rises with bad network quality, just like the delay.

    Args:
        size_bits : data size in bits
        quality   : network quality 0.1 (poor) .. 1.0 (great)

    Returns:
        energy in Joules
    """
    t_tx = transmission_delay(size_bits, quality)
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
    No transmission - only computation cost.
    """
    latency = compute_time(complexity_mcycles, DEVICE_CPU_SPEED)
    energy  = compute_energy(complexity_mcycles, DEVICE_CPU_SPEED, DEVICE_POWER)
    return latency, energy


def edge_cost(task_size_bits: float, complexity_mcycles: float,
              quality: float, queue_wait: float = 0.0):
    """
    Offload task to nearby edge server over the wireless uplink.

    Total latency = tx_delay(quality) + propagation + queue_wait + compute_time

    Args:
        quality    : network quality 0.1 (poor) .. 1.0 (great)
        queue_wait : extra waiting time if edge server is busy (seconds)
    """
    tx_delay  = transmission_delay(task_size_bits, quality)
    compute   = compute_time(complexity_mcycles, EDGE_CPU_SPEED)
    latency   = tx_delay + EDGE_PROPAGATION_DELAY + queue_wait + compute
    energy    = transmission_energy(task_size_bits, quality)
    return latency, energy


def cloud_cost(task_size_bits: float, complexity_mcycles: float, quality: float):
    """
    Offload task to distant cloud server over the wireless uplink.
    No queue - cloud has effectively infinite capacity.

    Total latency = tx_delay(quality) + propagation (large!) + compute_time
    """
    tx_delay  = transmission_delay(task_size_bits, quality)
    compute   = compute_time(complexity_mcycles, CLOUD_CPU_SPEED)
    latency   = tx_delay + CLOUD_PROPAGATION_DELAY + compute
    energy    = transmission_energy(task_size_bits, quality)
    return latency, energy
