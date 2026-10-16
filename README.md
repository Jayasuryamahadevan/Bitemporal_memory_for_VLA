# Bitemporal Memory for Embodied AI

This repository contains a queryable memory record for embodied agents, a
research survey, and early MuJoCo arm-control lessons.

The memory package stores sourced observations and facts with two clocks:
**world time** (when a state was true) and **knowledge time** (when the robot
learned it). It supports historical state queries, corrections and retractions,
freshness checks, and text, time, entity, and spatial search.

Start with the [memory guide](06_memory/README.md) and [architecture plan](06_memory/PLAN.md).
Run the demonstration and tests from the repository root:

```bash
python 06_memory/demo.py
python -m unittest discover -s tests -v
```

The [research survey](research.md) motivates the separation of the fast
control loops from the off-loop queryable record. The existing arm lessons are
in `01_get_a_robot/` and `02_build_your_own_arm/`.
