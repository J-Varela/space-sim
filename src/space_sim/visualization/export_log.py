from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from space_sim.simulation.engine import SimulationLog
from space_sim.simulation.scenario import Scenario


def export_log_to_json(log: SimulationLog, out_path: str = "out/simlog.json") -> str:
    """
    Export minimal playback data:
      {
        "sat_positions_eci_km": {
          "SAT-001": [{"t":0.0,"r":[x,y,z]}, ...],
          ...
        }
      }
    """
    data: Dict[str, Any] = {"sat_positions_eci_km": {}}

    for sat_id, samples in log.sat_positions_eci_km.items():
        data["sat_positions_eci_km"][sat_id] = [{"t": t, "r": [r[0], r[1], r[2]]} for (t, r) in samples]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return out_path


def export_playback_bundle(
    scenario: Scenario,
    log: SimulationLog,
    out_path: str = "out/playback_bundle.json",
) -> str:
    """
    Export a bundle for the Three.js viewer:
      - satellites: positions over time (ECI)
      - ground_stations: metadata (lat/lon/alt)
      - times: global time vector
      - visibility: sampled booleans by (gs_id, sat_id) over time

    JSON shape:
    {
      "times_s": [0,10,20,...],
      "sat_positions_eci_km": { "SAT-001": [[x,y,z], ...], ... },
      "ground_stations": { "GS-001": {"name": "...", "lat_deg":..., "lon_deg":..., "alt_km":..., "min_elev_deg":...}, ...},
      "visibility": { "GS-001|SAT-001": [0,1,1,0,...], ... }
    }
    """
    sat_ids = sorted(log.sat_positions_eci_km.keys())
    if not sat_ids:
        raise ValueError("No satellite positions found in log.")

    # Reference times (assume uniform sampling across sats)
    ref_samples = log.sat_positions_eci_km[sat_ids[0]]
    times_s: List[float] = [t for (t, _r) in ref_samples]

    data: Dict[str, Any] = {
        "times_s": times_s,
        "sat_positions_eci_km": {},
        "ground_stations": {},
        "visibility": {},
    }

    # Satellite positions as dense arrays aligned to times_s
    for sat_id in sat_ids:
        samples = log.sat_positions_eci_km[sat_id]
        if len(samples) != len(times_s):
            raise ValueError(f"{sat_id} samples length mismatch.")
        data["sat_positions_eci_km"][sat_id] = [[r[0], r[1], r[2]] for (_t, r) in samples]

    # Ground station metadata from scenario
    for gs_id, gs in scenario.ground_stations.items():
        data["ground_stations"][gs_id] = {
            "name": gs.name,
            "lat_deg": gs.lat_deg,
            "lon_deg": gs.lon_deg,
            "alt_km": gs.alt_km,
            "min_elev_deg": gs.min_elevation_deg,
        }

    # Visibility series: key as "GS|SAT" -> list[int] aligned to times_s
    # log.visibility contains samples [(t, bool), ...]
    for (gs_id, sat_id), samples in log.visibility.items():
        # Make a dense series aligned with times_s (assume same length/order)
        if len(samples) != len(times_s):
            raise ValueError(f"Visibility series length mismatch for {(gs_id, sat_id)}.")
        series = [1 if vis else 0 for (_t, vis) in samples]
        data["visibility"][f"{gs_id}|{sat_id}"] = series

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return out_path
