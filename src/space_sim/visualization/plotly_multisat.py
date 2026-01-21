from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import plotly.graph_objects as go

from space_sim.core.constants import R_EARTH_KM
from space_sim.simulation.engine import SimulationLog


def _earth_mesh(radius_km: float = R_EARTH_KM, n_lat: int = 30, n_lon: int = 60):
    lats = [(-math.pi / 2) + i * (math.pi / (n_lat - 1)) for i in range(n_lat)]
    lons = [(-math.pi) + j * (2 * math.pi / (n_lon - 1)) for j in range(n_lon)]

    x, y, z = [], [], []
    for lat in lats:
        row_x, row_y, row_z = [], [], []
        for lon in lons:
            row_x.append(radius_km * math.cos(lat) * math.cos(lon))
            row_y.append(radius_km * math.cos(lat) * math.sin(lon))
            row_z.append(radius_km * math.sin(lat))
        x.append(row_x); y.append(row_y); z.append(row_z)
    return x, y, z


def render_multisat_playback(
    log: SimulationLog,
    out_html: str = "out/phase4_multisat.html",
    show_earth: bool = True,
    frame_stride: int = 1,
    trail_len: int = 200,
) -> str:
    """
    Multi-satellite animated playback.
    Assumes all satellites were logged at the same time stamps.
    """
    sat_ids = sorted(log.sat_positions_eci_km.keys())
    if not sat_ids:
        raise ValueError("No satellite positions found in log.")

    # Use first sat as time reference
    ref = log.sat_positions_eci_km[sat_ids[0]]
    times_full = [t for (t, _r) in ref]

    # Stride frames for performance
    idxs = list(range(0, len(times_full), max(1, frame_stride)))
    times = [times_full[i] for i in idxs]

    # Prepack positions for quick frame creation
    pos: Dict[str, Dict[str, List[float]]] = {}
    for sid in sat_ids:
        samples = log.sat_positions_eci_km[sid]
        if len(samples) != len(times_full):
            raise ValueError(f"Satellite {sid} has {len(samples)} samples, expected {len(times_full)}.")
        xs = [samples[i][1][0] for i in idxs]
        ys = [samples[i][1][1] for i in idxs]
        zs = [samples[i][1][2] for i in idxs]
        pos[sid] = {"x": xs, "y": ys, "z": zs}

    fig = go.Figure()

    # Earth
    if show_earth:
        ex, ey, ez = _earth_mesh()
        fig.add_trace(go.Surface(x=ex, y=ey, z=ez, showscale=False, opacity=0.35, name="Earth"))

    # One marker trace per satellite (these will be updated each frame)
    # Also optionally include a short trail trace per satellite
    marker_trace_idxs: Dict[str, int] = {}
    trail_trace_idxs: Dict[str, int] = {}

    for sid in sat_ids:
        # Trail trace (starts empty-ish)
        fig.add_trace(go.Scatter3d(
            x=[pos[sid]["x"][0]],
            y=[pos[sid]["y"][0]],
            z=[pos[sid]["z"][0]],
            mode="lines",
            name=f"{sid} trail",
        ))
        trail_trace_idxs[sid] = len(list(fig.data)) - 1

        # Marker trace
        fig.add_trace(go.Scatter3d(
            x=[pos[sid]["x"][0]],
            y=[pos[sid]["y"][0]],
            z=[pos[sid]["z"][0]],
            mode="markers",
            name=f"{sid}",
            marker=dict(size=6),
        ))
        marker_trace_idxs[sid] = len(list(fig.data)) - 1

    # Build frames: update marker positions + trailing segments
    frames: List[go.Frame] = []
    for fi, t in enumerate(times):
        frame_data = []
        frame_traces = []

        for sid in sat_ids:
            x = pos[sid]["x"][fi]
            y = pos[sid]["y"][fi]
            z = pos[sid]["z"][fi]

            # trail window
            start = max(0, fi - trail_len)
            tx = pos[sid]["x"][start:fi + 1]
            ty = pos[sid]["y"][start:fi + 1]
            tz = pos[sid]["z"][start:fi + 1]

            # Update trail trace
            frame_data.append(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines"))
            frame_traces.append(trail_trace_idxs[sid])

            # Update marker trace
            frame_data.append(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers", marker=dict(size=6)))
            frame_traces.append(marker_trace_idxs[sid])

        frames.append(go.Frame(name=str(fi), data=frame_data, traces=frame_traces))

    fig.frames = frames

    # Slider steps (don’t include every step if huge)
    step_stride = max(1, len(times) // 50)
    slider_steps = [
        dict(
            method="animate",
            args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
            label=f"{int(times[i])}s"
        )
        for i in range(0, len(times), step_stride)
    ]

    fig.update_layout(
        title="Phase 4A — Multi-Satellite Playback (Plotly)",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h"),
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(steps=slider_steps, active=0)],
    )

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, auto_open=False)
    return out_html
