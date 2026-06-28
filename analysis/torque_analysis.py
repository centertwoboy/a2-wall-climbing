"""
Torque analysis — peak, RMS, 95th percentile, and saturation ratio.
"""

import numpy as np


def peak_torque(tau: np.ndarray) -> float:
    """Max absolute torque."""
    return float(np.max(np.abs(tau)))


def rms_torque(tau: np.ndarray) -> float:
    """Root-mean-square torque."""
    return float(np.sqrt(np.mean(tau ** 2)))


def percentile_torque(tau: np.ndarray, p: float = 95.0) -> float:
    """p-th percentile of absolute torque."""
    return float(np.percentile(np.abs(tau), p))


def saturation_ratio(tau_demand: np.ndarray, tau_max: float) -> float:
    """
    Fraction of samples where |tau_demand| >= tau_max.

    Returns ratio in [0, 1].
    """
    if len(tau_demand) == 0:
        return 0.0
    return float(np.sum(np.abs(tau_demand) >= tau_max) / len(tau_demand))


def torque_metrics(tau: np.ndarray, tau_demand: np.ndarray = None,
                   tau_max: float = 180.0) -> dict:
    """
    Compute all torque metrics for one joint.

    Returns dict with keys: peak, rms, p95, [sat_ratio].
    """
    metrics = {
        "peak": peak_torque(tau),
        "rms": rms_torque(tau),
        "p95": percentile_torque(tau, 95),
    }
    if tau_demand is not None:
        metrics["sat_ratio"] = saturation_ratio(tau_demand, tau_max)
    return metrics


def compute_all_joint_metrics(logger, tau_max: float = 180.0) -> dict:
    """
    Compute metrics for all 12 joints from a DataLogger.

    Returns: dict[joint_name] -> {peak, rms, p95, sat_ratio}
    """
    all_metrics = {}
    for col in logger.columns:
        if col.startswith("tau_") and not col.startswith("tau_demand"):
            joint_name = col.replace("tau_", "")
            tau = logger.get_column(col)
            tau_d_col = f"tau_demand_{joint_name}"
            tau_d = logger.get_column(tau_d_col) if tau_d_col in logger.columns else None
            all_metrics[joint_name] = torque_metrics(tau, tau_d, tau_max)
    return all_metrics


def check_warning_levels(peak: float, warning1: float = 126.0,
                         warning2: float = 144.0, limit: float = 180.0) -> str:
    """Classify peak torque into warning level."""
    if peak < warning1:
        return "OK"
    elif peak < warning2:
        return "WARNING_L1"
    elif peak < limit:
        return "WARNING_L2"
    else:
        return "EXCEEDS_LIMIT"
