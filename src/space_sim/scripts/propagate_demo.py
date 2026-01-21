import math
from space_sim.objects.satellite import Satellite
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

for t in [0, 600, 1200, 1800]:
    r, v = sat.state_eci_at(float(t))
    print(t, r)
