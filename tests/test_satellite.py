import math

from space_sim.objects import Satellite
from space_sim.physics.orbit import OrbitalElements
from space_sim.core.constants import MU_EARTH_KM3_S2  # if yours is in core/constants.py adjust import

def norm(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def deg(x): 
    return x * math.pi / 180.0

def test_satellite_state_returns_vectors():
    elements = OrbitalElements(
        a_km=7000.0,
        e=0.001,
        inc_rad=deg(51.6),
        raan_rad=deg(30.0),
        argp_rad=deg(40.0),
        M0_rad=0.0,
    )
    sat = Satellite(sat_id="SAT-001", name="DemoSat", elements=elements)

    r, v = sat.state_eci_at(0.0)
    assert len(r) == 3 and len(v) == 3
    assert norm(r) > 6000.0  # should be above Earth radius-ish in km
    assert norm(v) > 0.0
