import math 

from space_sim.physics.orbit import OrbitalElements, coe_to_rv_eci, mean_motion_rad_s
from space_sim.core.constants import MU_EARTH_KM3_S2

def norm(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def test_circular_orbit_radius_constant():
    # Circular orbit: r should stay ~a for all times
    a = 7000.0  # km
    e = 0.0
    elements = OrbitalElements(
        a_km=a,
        e=e,
        inc_rad=0.0,
        raan_rad=0.0,
        argp_rad=0.0,
        M0_rad=0.0,
    )

    n = mean_motion_rad_s(a)
    period = 2.0 * math.pi / n

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = frac * period
        r, v = coe_to_rv_eci(elements, t)
        assert abs(norm(r) - a) < 1e-6  # tight because e=0 ideal model

        # Check vis-viva: v^2 = mu(2/r - 1/a) = mu/a for circular
        v_expected = math.sqrt(MU_EARTH_KM3_S2 / a)
        assert abs(norm(v) - v_expected) < 1e-6