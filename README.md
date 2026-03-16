# Kerr-Descent-Engine

A modular physics engine for exploring interior trajectories in Kerr black holes using simplified ODIM-style descent dynamics.

## 📦 Project Structure
src/
init.py
engine.py
descent.py
metrics.py
geometry.py

## 🚀 Quick Start

```python
import numpy as np
from src import KerrDescentEngine, descent_step

engine = KerrDescentEngine(M=1.0, a=0.5, step_size=0.01)

initial_state = np.array([10.0, 1.0, 0.0])  # r, theta, phi
trajectory = engine.run(initial_state, steps=1000, descent_fn=descent_step)

print(trajectory)
