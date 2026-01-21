import math

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements
from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import Engine
from space_sim.simulation.systems.state_recorder import StateRecorderSystem
from space_sim.simulation.systems.visibility_system import VisibilitySystem
from space_sim.visualization.plotly_viewer import render_static_scene, render_animated_satellite

def deg(x): return x * math.pi / 180.0

scenario = Scenario(name="Phase4 Demo")

sat = Satellite(
    sat_id="SAT-001",
    name="DemoSat",
    elements=OrbitalElements(
        a_km=7000.0,
        e=0.001,
        inc_rad=deg(51.6),
        raan_rad=deg(30.0),
        argp_rad=deg(40.0),
        M0_rad=0.0,
    ),
)
scenario.add_satellite(sat)

gs = GroundStation(
    station_id="GS-001",
    name="DemoGS",
    lat_deg=38.8339,
    lon_deg=-104.8214,
    alt_km=1.9,
    min_elevation_deg=10.0,
)
scenario.add_ground_station(gs)

engine = Engine(dt_s=10.0, systems=[StateRecorderSystem(), VisibilitySystem()])
log = engine.run(scenario, t_start_s=0.0, t_end_s=2 * 3600.0)

static_path = render_static_scene(log, out_html="out/phase4_scene.html")
anim_path = render_animated_satellite(log, sat_id="SAT-001", out_html="out/phase4_animated.html")

print("Wrote:")
print(" -", static_path)
print(" -", anim_path)
print("\nOpen these HTML files in your browser.")
