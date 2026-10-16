# Embodied memory: queryable record layer

`robomemory` stores observations and evidence-backed state for one or more
robots. It supports two forms of time travel:

- **World time:** “Where was `cup-7` at 10:15?” (`valid_at`)
- **Knowledge time:** “What did the robot believe then, before a late frame or
  correction arrived?” (`known_event_id` or `known_at`)

The record is separate from the reflex/action loops in `research.md`. It never
generates motor commands. Perception, tracking, progress verification, and
coordinate transforms are upstream. This package stores their results and
lets a planner or operator inspect the exact evidence.

## Quick start

From the repository root, run:

```bash
.venv/bin/python 06_memory/demo.py
.venv/bin/python -m unittest discover -s tests -v
```

The Python API is the intended integration point:

```python
from datetime import datetime, timedelta, timezone
from robomemory import MemoryStore

memory = MemoryStore("robot_memory.db")
t = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)

observation_id = memory.record_observation(
    robot_id="arm-1", episode_id="sort-42", observed_at=t,
    kind="vision", text="red cup on left table", source="wrist-camera",
    entity_id="cup-7", frame_id="map", position=(1.2, 0.4, 0.8),
    evidence_uri="mcap://sort-42/camera/0182", metadata={"tracker": "v3"},
    idempotency_key="sort-42/camera/0182",
)
fact_id = memory.assert_fact(
    robot_id="arm-1", subject="cup-7", predicate="location",
    value="left table", valid_from=t, evidence_event_id=observation_id,
    status="verified", confidence=0.96,
)
answer = memory.query_state(
    robot_id="arm-1", subject="cup-7", predicate="location",
    valid_at=t + timedelta(seconds=5), max_age=timedelta(seconds=30),
)
assert answer.status == "verified"
assert answer.fact.evidence_event_id == observation_id
```

`verified` means an upstream check has confirmed the state. A detector guess
should be stored as `hypothesis`; a failed or occluded check can be stored as
`unknown` with a `null` value. The latest assertion blocks older state even
when it is unverified. Set `max_age` for dynamic facts. A returned `verified`
status is evidence provenance, not a guarantee that perception was correct.

To inspect what was known before a later update, pass the earlier `fact_id` as
`known_event_id`. The returned `known_event_id` is the exact replay cursor.
Use `known_at` for wall-clock queries where an exact event cursor is unavailable.
`history()` shows assertions and subsequent retractions.

Search the episode record with any combination of text, time interval, entity,
episode, and radius in a named coordinate frame:

```python
hits = memory.search_observations(
    robot_id="arm-1", text="red cup", entity_id="cup-7",
    observed_from=t, observed_until=t + timedelta(minutes=1),
    near=(1.2, 0.4, 0.8), radius=0.5, frame_id="map",
)
```

Text search uses token matching (all query words) and BM25 ranking. It is not
vector or language-model retrieval. The exact event text and evidence URI are
returned so an operator or slower semantic service can inspect the raw record.

## CLI

```bash
.venv/bin/python -m robomemory --db /tmp/robot.db observe \
  --robot arm-1 --at 2026-10-16T10:00:00Z --kind vision \
  --text "red cup on left table" --source wrist-camera \
  --entity cup-7 --frame map --position 1.2 0.4 0.8

.venv/bin/python -m robomemory --db /tmp/robot.db assert \
  --robot arm-1 --subject cup-7 --predicate location \
  --value '"left table"' --at 2026-10-16T10:00:00Z --evidence 1

.venv/bin/python -m robomemory --db /tmp/robot.db state \
  --robot arm-1 --subject cup-7 --predicate location \
  --at 2026-10-16T10:00:05Z --max-age-seconds 30

.venv/bin/python -m robomemory --db /tmp/robot.db search \
  --robot arm-1 --text "red cup" --near 1.2 0.4 0.8 \
  --radius 0.5 --frame map
```

The CLI emits JSON. `observe` and `assert` return event IDs. `state` can take
`--known-event-id` or `--known-at`; `history` and `search` accept the same
knowledge cutoff. `retract` requires a new observation and reason.

## Operational boundaries

- SQLite WAL permits concurrent readers and serialized writers on **one
  host**. Keep the database on a local filesystem. The schema version is
  checked on startup. Writes are transactional, idempotency keys guard retries,
  and each robot's evidence is isolated.
- Store raw frames, video, and high-rate telemetry in a media recorder such as
  ROS bag/MCAP. Put durable record URIs in `evidence_uri`; do not put images or
  high-rate control samples into this semantic ledger.
- Use a stable tracker ID for `subject` and `entity_id`. `frame_id` is literal:
  spatial search compares only positions already expressed in the **same**
  coordinate frame. Perform transforms at the observation timestamp before
  writing, using the robot's transform system.
- Reconcile contradictory evidence upstream. A retraction preserves the old
  knowledge view. Follow a retraction with a corrected assertion or an
  `unknown` assertion when the physical state remains uncertain.
- A robot fleet or remote service will need server storage, access control,
  replication, schema migrations, backup/restore, and latency monitoring.
  This package is a durable single-host foundation, not a validated autonomy
  stack. The task-level evaluation path is in `PLAN.md`.

Design rationale follows the existing [research survey](../research.md) and
the primary [SQLite WAL](https://www.sqlite.org/wal.html) and
[FTS5](https://www.sqlite.org/fts5.html) documentation.
