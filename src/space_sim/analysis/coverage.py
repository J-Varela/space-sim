from __future__ import annotations

from typing import Dict, List, Tuple

from space_sim.physics.visibility import compute_access_windows
from space_sim.simulation.engine import SimulationLog


def access_windows_from_log(log: SimulationLog) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """
    Convert sampled visibility time series into access windows for each (gs_id, sat_id).
    """
    out: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}

    for key, samples in log.visibility.items():
        times = [t for (t, _vis) in samples]
        flags = [vis for (_t, vis) in samples]
        out[key] = compute_access_windows(times, flags)

    return out
