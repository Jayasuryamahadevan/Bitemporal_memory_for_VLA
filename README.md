# Bitemporal Memory for Embodied AI

This repository contains an onboard memory runtime for embodied agents, a
research survey, and early MuJoCo arm-control lessons.

The controller-facing memory read is an immutable in-RAM snapshot. Bounded
background workers store evidence-backed facts in a local bitemporal ledger
with two clocks: **world time** (when a state was true) and **knowledge time**
(when the robot learned it). Historical queries, corrections, freshness
checks, and text, time, entity, and spatial search stay off the action path.
Evidence-backed directed scene relations and bounded retrieval now assemble
facts, episodes, graph neighbors, and provenance for question answering,
planning, and VLA context.

Start with the [onboard memory guide](06_memory/README.md) and
[architecture plan](06_memory/PLAN.md).
Install the core package, then run the demonstration and tests from the
repository root:

```bash
python -m pip install -e .
python 06_memory/demo.py
python -m unittest discover -s tests -v
```

The optional MuJoCo Ant virtual-robot integration test and install steps are
also documented in the [onboard memory guide](06_memory/README.md).

The [research survey](research.md) motivates the separation of the fast
control loops from the off-loop queryable record. The existing arm lessons are
in `01_get_a_robot/` and `02_build_your_own_arm/`.
