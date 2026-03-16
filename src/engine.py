import numpy as np

class KerrDescentEngine:
    """
    Core engine for performing Kerr-style interior descent.
    Handles state evolution, parameter storage, and trajectory output.
    """

    def __init__(self, M=1.0, a=0.5, step_size=0.01):
        self.M = M
        self.a = a
        self.step_size = step_size

    def step(self, state, descent_fn):
        """
        Performs a single descent step using the provided descent function.
        """
        return descent_fn(state, self.M, self.a, self.step_size)

    def run(self, initial_state, steps, descent_fn):
        """
        Evolves the system for a number of steps.
        Returns the full trajectory.
        """
        trajectory = [initial_state]
        state = initial_state

        for _ in range(steps):
            state = self.step(state, descent_fn)
            trajectory.append(state)

        return np.array(trajectory)

