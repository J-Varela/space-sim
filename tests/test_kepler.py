# Tests (This is what makes it "defense-style")
import math
import pytest

from space_sim.physics.gravity import solve_keplers_equation


def test_kepler_zero_eccentricity():
    # If e=0, E=M exactly
    for M in [0.0, 0.5, 1.0, 2.0, 5.0]:
        E = solve_keplers_equation(M, 0.0)
        assert math.isclose((E - (M % (2*math.pi))) % (2*math.pi), 0.0, abs_tol=1e-12)


def test_kepler_converges_typical():
    E = solve_keplers_equation(M_rad=1.0, e=0.4)
    # Verify equation residual
    res = E - 0.4 * math.sin(E) - (1.0 % (2*math.pi))
    assert abs(res) < 1e-10
