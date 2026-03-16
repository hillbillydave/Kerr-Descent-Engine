import numpy as np

def compute_metrics(state, previous_state):
    """
    Computes simple diagnostics for the Kerr descent.
    Returns a dictionary of useful metrics.
    """

    r, theta, phi = state
    r_prev, theta_prev, phi_prev = previous_state

    # Step magnitudes
    dr = r - r_prev
    dtheta = theta - theta_prev
    dphi = phi - phi_prev

    # Radial stability indicator
    radial_speed = abs(dr)

    # Angular drift magnitude
    angular_drift = np.sqrt(dtheta**2 + dphi**2)

    return {
        "r": r,
        "theta": theta,
        "phi": phi,
        "dr": dr,
        "dtheta": dtheta,
        "dphi": dphi,
        "radial_speed": radial_speed,
        "angular_drift": angular_drift
    }

