from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from space_sim.core.constants import R_EARTH_KM
from space_sim.core.frames import eci_to_ecef_km, ecef_to_latlon_deg
from space_sim.simulation.engine import SimulationLog


@dataclass
class CoverageGridSpec:
    """
    Defines a lat/lon grid over Earth.
    """
    lat_step_deg: float = 2.0
    lon_step_deg: float = 2.0

    def lat_bins(self) -> List[float]:
        # centers from -90..+90
        n = int(round(180.0 / self.lat_step_deg)) + 1
        return [-90.0 + i * self.lat_step_deg for i in range(n)]

    def lon_bins(self) -> List[float]:
        # centers from -180..+180
        n = int(round(360.0 / self.lon_step_deg)) + 1
        return [-180.0 + i * self.lon_step_deg for i in range(n)]


def _wrap_lon_deg(lon: float) -> float:
    # wrap to [-180, 180)
    x = ((lon + 180.0) % 360.0) - 180.0
    return x


def eci_to_latlon_deg(r_eci_km: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Spherical Earth approximation.
    NOTE: This treats ECI axes as Earth-fixed at t=0 (no Earth rotation correction).
    For coverage v0, it's acceptable; later we'll incorporate ECI->ECEF using time.
    """
    x, y, z = r_eci_km
    r = math.sqrt(x*x + y*y + z*z)
    if r == 0:
        raise ValueError("Zero position vector.")
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    lon = _wrap_lon_deg(lon)
    return lat, lon


def haversine_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """
    Great-circle distance on a sphere (Earth) in km.
    """
    lat1 = math.radians(lat1_deg); lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg); lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return R_EARTH_KM * c


def ground_swath_radius_km(alt_km: float, fov_half_angle_deg: float) -> float:
    """
    Approximate ground footprint radius (km) for a nadir-pointing sensor.
    Simplified flat approximation: swath ~= alt * tan(fov)
    For better geometry later we’ll use spherical intersection.
    """
    return alt_km * math.tan(math.radians(fov_half_angle_deg))


@dataclass
class CoverageResult:
    spec: CoverageGridSpec
    hits: List[List[int]]                      # [lat_idx][lon_idx] hit counts
    last_seen_s: List[List[Optional[float]]]    # last time seen per cell
    revisit_samples_s: List[List[List[float]]]  # per-cell revisit deltas (can be big)

    def to_compact_json(self) -> Dict:
        lats = self.spec.lat_bins()
        lons = self.spec.lon_bins()

        # Convert revisit samples into summary stats per cell (min/mean/max/count)
        stats = []
        for i in range(len(lats)):
            row = []
            for j in range(len(lons)):
                samples = self.revisit_samples_s[i][j]
                if not samples:
                    row.append(None)
                else:
                    mn = min(samples)
                    mx = max(samples)
                    mean = sum(samples) / len(samples)
                    row.append({"count": len(samples), "min_s": mn, "mean_s": mean, "max_s": mx})
            stats.append(row)

        return {
            "lat_step_deg": self.spec.lat_step_deg,
            "lon_step_deg": self.spec.lon_step_deg,
            "lat_bins": lats,
            "lon_bins": lons,
            "hits": self.hits,
            "revisit_stats": stats,
        }


def compute_earth_coverage_from_log(
    log: SimulationLog,
    sat_id: str,
    dt_s: float,
    grid: CoverageGridSpec = CoverageGridSpec(2.0, 2.0),
    fov_half_angle_deg: float = 15.0,
) -> CoverageResult:
    """
    Compute coverage grid from a satellite trajectory.
    Steps:
      - Convert sat position to lat/lon (subsat point)
      - Compute approximate swath radius from altitude and FOV
      - Mark grid cells within swath as hit
      - Record revisit times per cell
    """
    if sat_id not in log.sat_positions_eci_km:
        raise ValueError(f"sat_id '{sat_id}' not found in log.")

    samples = log.sat_positions_eci_km[sat_id]
    times = [t for (t, _r) in samples]
    rs = [r for (_t, r) in samples]

    lats = grid.lat_bins()
    lons = grid.lon_bins()

    hits = [[0 for _ in range(len(lons))] for _ in range(len(lats))]
    last_seen: List[List[Optional[float]]] = [[None for _ in range(len(lons))] for _ in range(len(lats))]
    revisit_samples: List[List[List[float]]] = [[[] for _ in range(len(lons))] for _ in range(len(lats))]

    for t_s, r_eci in zip(times, rs):
        # altitude approx
        x, y, z = r_eci
        rmag = math.sqrt(x*x + y*y + z*z)
        alt_km = max(0.0, rmag - R_EARTH_KM)

        r_ecef = eci_to_ecef_km(r_eci, t_s)
        sat_lat, sat_lon = ecef_to_latlon_deg(r_ecef)

        swath_km = ground_swath_radius_km(alt_km, fov_half_angle_deg)

        # Quick bounding box in degrees to avoid checking whole Earth
        # 1 deg latitude ~ 111 km; longitude shrink by cos(lat)
        lat_pad = swath_km / 111.0
        coslat = max(0.1, math.cos(math.radians(sat_lat)))
        lon_pad = swath_km / (111.0 * coslat)

        lat_min = sat_lat - lat_pad
        lat_max = sat_lat + lat_pad
        lon_min = sat_lon - lon_pad
        lon_max = sat_lon + lon_pad

        # Iterate only over cells within the rough bbox
        for i, latc in enumerate(lats):
            if latc < lat_min or latc > lat_max:
                continue
            for j, lonc in enumerate(lons):
                # handle wrap at dateline by checking wrapped distance
                lonc_wrapped = lonc
                # simple check: bring lonc near sat_lon
                dlon = _wrap_lon_deg(lonc_wrapped - sat_lon)
                lon_near = sat_lon + dlon

                if lon_near < lon_min or lon_near > lon_max:
                    continue

                d = haversine_km(sat_lat, sat_lon, latc, lon_near)
                if d <= swath_km:
                    hits[i][j] += 1
                    prev = last_seen[i][j]
                    if prev is not None:
                        revisit_samples[i][j].append(t_s - prev)
                    last_seen[i][j] = t_s

    return CoverageResult(spec=grid, hits=hits, last_seen_s=last_seen, revisit_samples_s=revisit_samples)


def compute_constellation_coverage_from_log(
    log: SimulationLog,
    sat_ids: List[str],
    dt_s: float,
    grid: CoverageGridSpec = CoverageGridSpec(2.0, 2.0),
    fov_half_angle_deg: float = 15.0,
) -> CoverageResult:
    """
    Union coverage: cell is hit if ANY satellite covers it at time t.
    Revisit is computed from union hits timeline per cell.
    """
    for sid in sat_ids:
        if sid not in log.sat_positions_eci_km:
            raise ValueError(f"sat_id '{sid}' not found in log.")

    # Reference time vector
    ref_samples = log.sat_positions_eci_km[sat_ids[0]]
    times = [t for (t, _r) in ref_samples]
    lats = grid.lat_bins()
    lons = grid.lon_bins()

    hits = [[0 for _ in range(len(lons))] for _ in range(len(lats))]
    last_seen: List[List[Optional[float]]] = [[None for _ in range(len(lons))] for _ in range(len(lats))]
    revisit_samples: List[List[List[float]]] = [[[] for _ in range(len(lons))] for _ in range(len(lats))]

    # Pre-read trajectories
    traj = {sid: [r for (_t, r) in log.sat_positions_eci_km[sid]] for sid in sat_ids}

    for k, t_s in enumerate(times):
        # For this timestep, compute each sat subsat lat/lon and swath
        sat_state = []
        for sid in sat_ids:
            r_eci = traj[sid][k]
            x, y, z = r_eci
            rmag = math.sqrt(x*x + y*y + z*z)
            alt_km = max(0.0, rmag - R_EARTH_KM)

            r_ecef = eci_to_ecef_km(r_eci, t_s)
            sat_lat, sat_lon = ecef_to_latlon_deg(r_ecef)

            swath_km = ground_swath_radius_km(alt_km, fov_half_angle_deg)
            sat_state.append((sat_lat, sat_lon, swath_km))

        # For each sat, mark cells; but union logic means we should only "hit" once per timestep per cell
        # We'll build a boolean mask per timestep for efficiency at this scale.
        hit_this_step = [[False for _ in range(len(lons))] for _ in range(len(lats))]

        for (sat_lat, sat_lon, swath_km) in sat_state:
            lat_pad = swath_km / 111.0
            coslat = max(0.1, math.cos(math.radians(sat_lat)))
            lon_pad = swath_km / (111.0 * coslat)

            lat_min = sat_lat - lat_pad
            lat_max = sat_lat + lat_pad
            lon_min = sat_lon - lon_pad
            lon_max = sat_lon + lon_pad

            for i, latc in enumerate(lats):
                if latc < lat_min or latc > lat_max:
                    continue
                for j, lonc in enumerate(lons):
                    dlon = _wrap_lon_deg(lonc - sat_lon)
                    lon_near = sat_lon + dlon
                    if lon_near < lon_min or lon_near > lon_max:
                        continue

                    d = haversine_km(sat_lat, sat_lon, latc, lon_near)
                    if d <= swath_km:
                        hit_this_step[i][j] = True

        # Apply union hits + revisit
        for i in range(len(lats)):
            for j in range(len(lons)):
                if hit_this_step[i][j]:
                    hits[i][j] += 1
                    prev = last_seen[i][j]
                    if prev is not None:
                        revisit_samples[i][j].append(t_s - prev)
                    last_seen[i][j] = t_s

    return CoverageResult(spec=grid, hits=hits, last_seen_s=last_seen, revisit_samples_s=revisit_samples)


def save_coverage_json(result: CoverageResult, out_path: str = "out/coverage.json") -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_compact_json(), f)
    return out_path
