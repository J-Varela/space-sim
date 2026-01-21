# Two-body / 32 model

from __future__ import annotations

import math


def wrap_to_2pi(angle_rad: float) -> float:
    """Wrap angle to [0, 2π)."""
    two_pi = 2.0 * math.pi
    return angle_rad % two_pi


def solve_keplers_equation(M_rad: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation for elliptic orbits:
        M = E - e sin(E)
    using Newton-Raphson.

    Args:
        M_rad: Mean anomaly (rad)
        e: eccentricity (0 <= e < 1)
        tol: convergence tolerance
        max_iter: iteration cap

    Returns:
        E_rad: Eccentric anomaly (rad)
    """
    if not (0.0 <= e < 1.0):
        raise ValueError("Elliptic Kepler solver requires 0 <= e < 1.")

    M = wrap_to_2pi(M_rad)

    # Good initial guess
    if e < 0.8:
        E = M
    else:
        # For higher e, start closer to pi to avoid slow convergence near M~0
        E = math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        if abs(fp) < 1e-15:
            break
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            return wrap_to_2pi(E)

    # If not converged, return best effort but signal via exception
    raise RuntimeError("Kepler solver did not converge within max_iter.")
