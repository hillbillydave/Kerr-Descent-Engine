import numpy as np
from src import (
    KerrDescentEngine,
    descent_step,
    compute_metrics,
    kerr_metric
)

def test_engine_initialization():
    engine = KerrDescentEngine(M=1.0, a=0.5, step_size=0.01)
    assert engine.M == 1.0
    assert engine.a == 0.5
    assert engine.step_size == 0.01

def test_single_descent_step():
    state = np.array([10.0, 1.0, 0.0])
    new_state = descent_step(state, M=1.0, a=0.5, step_size=0.01)

    assert new_state.shape == (3,)
    assert np.all(np.isfinite(new_state))
    assert new_state[0] < state[0]  # r should decrease

def test_engine_run():
    engine = KerrDescentEngine(M=1.0, a=0.5, step_size=0.01)
    initial_state = np.array([10.0, 1.0, 0.0])

    trajectory = engine.run(initial_state, steps=50, descent_fn=descent_step)

    assert trajectory.shape == (51, 3)
    assert np.all(np.isfinite(trajectory))
    assert trajectory[-1, 0] < trajectory[0, 0]  # r decreases over time

def test_metrics_computation():
    prev = np.array([10.0, 1.0, 0.0])
    curr = np.array([9.9, 1.01, 0.02])

    metrics = compute_metrics(curr, prev)

    assert "dr" in metrics
    assert "angular_drift" in metrics
    assert metrics["dr"] < 0
    assert metrics["angular_drift"] > 0

def test_kerr_metric():
    r = 10.0
    theta = 1.0
    M = 1.0
    a = 0.5

    g = kerr_metric(r, theta, M, a)

    # Ensure all expected components exist
    for key in ["g_tt", "g_rr", "g_thth", "g_phph", "Sigma", "Delta"]:
        assert key in g
        assert np.isfinite(g[key])
