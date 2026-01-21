from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import plotly.graph_objects as go

from space_sim.core.constants import R_EARTH_KM
from space_sim.core.frames import Vector3
from space_sim.simulation.engine import SimulationLog


def _earth_mesh(radius_km: float = R_EARTH_KM, n_lat: int = 30, n_lon: int = 60):
    # Create a sphere mesh (parametric)
    lats = [(-math.pi / 2) + i * (math.pi / (n_lat - 1)) for i in range(n_lat)]
    lons = [(-math.pi) + j * (2 * math.pi / (n_lon - 1)) for j in range(n_lon)]

    x = []
    y = []
    z = []
    for lat in lats:
        row_x = []
        row_y = []
        row_z = []
        for lon in lons:
            row_x.append(radius_km * math.cos(lat) * math.cos(lon))
            row_y.append(radius_km * math.cos(lat) * math.sin(lon))
            row_z.append(radius_km * math.sin(lat))
        x.append(row_x)
        y.append(row_y)
        z.append(row_z)
    return x, y, z


def render_static_scene(
    log: SimulationLog,
    out_html: str = "out/phase4_scene.html",
    show_earth: bool = True,
) -> str:
    """
    Renders a static 3D scene:
      - Earth sphere
      - Orbit tracks for each satellite
      - Last position marker for each satellite
    """
    fig = go.Figure()

    if show_earth:
        ex, ey, ez = _earth_mesh()
        fig.add_trace(
            go.Surface(
                x=ex, y=ey, z=ez,
                showscale=False,
                opacity=0.35,
                name="Earth"
            )
        )

    # Satellite tracks + last position
    for sat_id, samples in log.sat_positions_eci_km.items():
        xs = [r[0] for (_t, r) in samples]
        ys = [r[1] for (_t, r) in samples]
        zs = [r[2] for (_t, r) in samples]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            name=f"{sat_id} track",
        ))

        # last point
        fig.add_trace(go.Scatter3d(
            x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
            mode="markers",
            name=f"{sat_id} now",
            marker=dict(size=5),
        ))

    fig.update_layout(
        title="Phase 4 — Space Sim Playback (Static Scene)",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h"),
    )

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, auto_open=False)
    return out_html


def render_animated_satellite(
    log: SimulationLog,
    sat_id: str,
    out_html: str = "out/phase4_animated.html",
    show_earth: bool = True,
) -> str:
    """
    Renders an animated 3D scene for ONE satellite:
      - Earth
      - Full track
      - A moving marker across timesteps
    """
    if sat_id not in log.sat_positions_eci_km:
        raise ValueError(f"sat_id '{sat_id}' not found in log.sat_positions_eci_km")

    samples = log.sat_positions_eci_km[sat_id]
    times = [t for (t, _r) in samples]
    xs = [r[0] for (_t, r) in samples]
    ys = [r[1] for (_t, r) in samples]
    zs = [r[2] for (_t, r) in samples]

    fig = go.Figure()

    if show_earth:
        ex, ey, ez = _earth_mesh()
        fig.add_trace(go.Surface(x=ex, y=ey, z=ez, showscale=False, opacity=0.35, name="Earth"))

    # Track
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name=f"{sat_id} track",
    ))

    # Initial marker
    fig.add_trace(go.Scatter3d(
        x=[xs[0]], y=[ys[0]], z=[zs[0]],
        mode="markers",
        name=f"{sat_id} marker",
        marker=dict(size=6),
    ))

    # Frames update the marker trace (the last trace)
    frames = []
    for i in range(len(times)):
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(x=[xs[i]], y=[ys[i]], z=[zs[i]], mode="markers", marker=dict(size=6))
            ],
            traces=[2],  # update trace index 2 (marker)
        ))

    fig.frames = frames

    fig.update_layout(
        title=f"Phase 4 — Animated Playback: {sat_id}",
        scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                        label=f"{int(times[i])}s") for i in range(0, len(times), max(1, len(times)//20))],
            active=0
        )]
    )

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, auto_open=False)
    return out_html
