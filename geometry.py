import numpy as np

def delta(r, a, M):
    """Radial function in Kerr metric"""
    return r**2 - 2*M*r + a**2

def sigma(r, theta, a):
    """Angular function in Kerr metric"""
    return r**2 + a**2 * np.cos(theta)**2

def kerr_metric(r, theta, a, M):
    """Returns the Kerr metric tensor components at (r, theta)"""
    Δ = delta(r, a, M)
    Σ = sigma(r, theta, a)

    g_tt = -(1 - (2*M*r)/Σ)
    g_rr = Σ / Δ
    g_thth = Σ
    g_phph = (r**2 + a**2 + (2*M*a**2*r*np.sin(theta)**2)/Σ) * np.sin(theta)**2
    g_tph = -(2*M*a*r*np.sin(theta)**2)/Σ

    return {
        "g_tt": g_tt,
        "g_rr": g_rr,
        "g_thth": g_thth,
        "g_phph": g_phph,
        "g_tph": g_tph
    }
