
import numpy as np

def descent_step(state, M, a, step_size):
    """
    Performs one Kerr-style interior descent step.
    This is a placeholder model that you can refine later.
    """

    r, theta, phi = state

    # Simple inward radial drift
    dr = -step_size * (1 + a**2 / (r**2 + 1e-9))

    # Small angular evolution
    dtheta = step_size * 0.01 * np.sin(theta)
    dphi = step_size * 0.02

    return np.array([
        r + dr,
        theta + dtheta,
        phi + dphi
    ])
