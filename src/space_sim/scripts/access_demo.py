import math

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements

def deg(x): return x * math.pi / 180.0

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
    )
)

# Example ground station (roughly: Colorado Springs-ish)
gs = GroundStation(
    station_id="GS-001",
    name="DemoGS",
    lat_deg=38.8339,
    lon_deg=-104.8214,
    alt_km=1.9,
    min_elevation_deg=10.0
)

times = [float(t) for t in range(0, 6*3600 + 1, 10)]  # 6 hours, 10s step
windows = gs.access_windows(sat, times)

print(f"Found {len(windows)} access windows:")
for a, b in windows:
    print(f"  start={a:8.1f}s  end={b:8.1f}s  duration={(b-a):6.1f}s")
