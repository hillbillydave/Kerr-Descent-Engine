import numpy as np
from src import KerrDescentEngine, descent_step

def test_engine_runs():
    engine = KerrDescentEngine(M=1.0, a=0.5, step_size=0.01)
    initial_state = np.array([10.0, 1.0, 0.0])

    trajectory = engine.run(initial_state, steps=10, descent_fn=descent_step)

    # Basic sanity checks
    assert trajectory.shape[0] == 11
    assert trajectory.shape[1] == 3
    assert np.all(np.isfinite(trajectory))
