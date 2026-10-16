# Onboard memory for embodied AI

This package is designed to run on the robot's onboard computer. It separates
the action path from durable recording:

- The controller reads the latest immutable RAM snapshot.
- Background workers append evidence and facts to the local bitemporal ledger.
- Historical, text, and spatial queries run asynchronously away from motor control.

Python is not a hard real-time runtime. A hard real-time C++ controller should
consume the latest fixed-size snapshot over shared memory or a latest-value
ROS 2 topic. The snapshot read in this package performs no database or network
I/O, but still inherits Python scheduling and garbage-collection jitter.

## Start here

```bash
python 06_memory/demo.py
python -m unittest discover -s tests -v
```

## Virtual robot validation

The optional integration test runs Gymnasium's articulated MuJoCo `Ant-v5`
robot headlessly. Gymnasium ships the Ant MJCF model with its package; install
the simulator and model dependency with:

```bash
python -m pip install -e '.[sim]'
python -m unittest discover -s tests -p 'test_mujoco_integration.py' -v
```

The test steps the simulated robot, records its base pose as a sourced sensor
observation, stores evidence-linked state facts, and checks current cache reads
and bitemporal queries at earlier and later knowledge cursors. The test uses a
temporary local database. It validates the memory API against a simulator; it
does not validate a ROS 2 adapter, perception, robot hardware, or control-loop
timing. Simulation time is mapped onto an isolated UTC interval for this test;
production integrations must carry an explicit clock domain and boot identity
before combining simulation or device clocks with wall time.

`Ant-v5` is a generic quadruped benchmark model, not a robot-specific digital
twin. The memory API has no built-in simulator dependency: a robot or simulator
adapter must map its own timestamps, frames, entities, and sensor output into
the observation and fact interface. A Unitree Go2 Menagerie download was not
available in this environment because direct model downloads could not resolve
the network host; no Go2 hardware/model validation is claimed.

The lower-level MemoryStore API is useful for offline tools, replay, and tests.
Robot processes should use OnboardMemory:

```python
from datetime import datetime, timedelta, timezone
from robomemory import OnboardMemory

memory = OnboardMemory(
    "/var/lib/robot-memory/arm-1.db",
    robot_id="arm-1",
    write_queue_capacity=256,
    reserved_critical_slots=16,
    max_hot_keys=512,
    history_per_key=8,
    max_write_bytes=65536,
)

# This runs in a perception/recorder worker, never in the motor callback.
seen = memory.submit_observation(
    observed_at=datetime.now(timezone.utc),
    kind="vision",
    text="red cup on left table",
    source="wrist-camera",
    episode_id="sort-42",
    entity_id="cup-7",
    frame_id="map",
    position=(1.2, 0.4, 0.8),
    evidence_uri="mcap://sort-42/camera/0182",
    metadata={"tracker": "v3"},
    idempotency_key="sort-42/camera/0182",
)
observation_id = seen.result(timeout=1)  # wait only in this worker
assertion = memory.submit_fact(
    subject="cup-7",
    predicate="location",
    value="left table",
    valid_from=datetime.now(timezone.utc),
    evidence_event_id=observation_id,
    status="verified",
    confidence=0.96,
)
assertion.result(timeout=1)  # still outside the action loop

# Controller-facing call: in-memory lookup, no SQL or RPC.
controller_cycle_time_utc = datetime.now(timezone.utc)  # use the robot clock
state = memory.read_state(
    subject="cup-7",
    predicate="location",
    now=controller_cycle_time_utc,
    max_age=timedelta(seconds=2),
)
if state.status != "verified":
    # Use the robot's existing conservative/sensor-driven fallback.
    pass

# Historical queries are asynchronous and must also stay off the control loop.
past = memory.query_state_async(
    subject="cup-7",
    predicate="location",
    valid_at=datetime.now(timezone.utc),
).result(timeout=2)

# Call during non-real-time lifecycle shutdown.
memory.close(timeout=5)
```

Pass the controller's current UTC timestamp as now; the read does not sample a
clock. read_state returns one of verified, hypothesis, unknown, stale, expired, or
missing. The latest applicable state blocks fallback to an older value when it
is uncertain or expired. Pass a freshness limit for dynamic predicates. A
verified label records upstream verification; it is not a guarantee that the
perception system was correct.

## Time travel and query

The persistent store has two clocks:

- valid_at: when a fact was true in the physical world.
- known_event_id or known_at: what the robot had recorded by then.

Use query_state_async, history_async, or search_observations_async for
historical lookup. A query returns fact and evidence event IDs. Text matching
uses all query terms with BM25 ranking; it is not vector search or an
unconstrained language-model answer. A slower planner may translate a natural
language request into these structured queries, then cite the returned
evidence.

The hot snapshot answers current state only. It is bounded by max_hot_keys;
eviction returns missing so callers fall back safely. It keeps at most
max_hot_keys × history_per_key facts (capped at 4096) and may return missing for
an older state outside that window; use the durable query for time travel. At
startup it loads bounded live histories for keys with the newest live assertions,
including facts whose valid time is ahead of wall time. Do not declare memory
ready until the runtime has been constructed and hydration succeeds.

## Queues and failure handling

- submit_observation, submit_fact, and submit_retraction return a
  WriteReceipt. Do not wait on it from a control callback.
- Each submission detaches caller-owned data before queueing. JSON-compatible
  write fields are limited to max_write_bytes (65,536 by default); oversized
  writes raise MemoryPayloadTooLarge. Submit from a worker, since copying and
  validation consume CPU. Do not mutate an input concurrently with submission.
- If the bounded writer queue is full, submission raises MemoryQueueFull
  immediately and increments writes_rejected. Handle it as an explicit loss:
  low-value telemetry may be sampled or dropped. Reserved slots are available
  to fact/retraction writes and to observations submitted with critical=True.
  Safety-critical handling must trigger the controller's existing safe
  behavior and operator fault path if even that capacity is exhausted.
- Historical query submission is capped. MemoryQueryBusy means retry or
  degrade the non-critical query; never wait for query capacity from a control
  callback.
- Database/write failures fail the receipt. stats() exposes accepted,
  completed, failed, and rejected writes plus queue depth, high-water mark,
  cache size, and a degraded flag. Keep using the last committed snapshot only
  within its freshness limits.
- close() drains accepted writes and waits for in-flight queries up to its
  timeout. Call it from process lifecycle management, not from control.

## Storage boundaries

SQLite WAL is the onboard semantic/event ledger, on a local filesystem. It is
single-host storage, not a fleet database. High-rate sensor streams, images,
and video should go to rosbag2/MCAP or a bounded local media store; the ledger
keeps their durable evidence URIs. Position values must already be transformed
into the named frame at the source timestamp. An upstream tracker owns stable
entity IDs and uncertainty.

One OnboardMemory instance must own writes for a robot while it is running.
Writes made directly through another MemoryStore or runtime do not notify the
hot snapshot. Restart the runtime to hydrate external changes, or route every
write through its queue.

This schema expects one timezone-aware robot clock converted to UTC. Do not
mix wall-clock, simulation-clock, or monotonic device timestamps. A ROS 2
integration must add an explicit clock domain and boot identifier before it
merges event times across clock resets or replay.

For a robot fleet, add authenticated replication, encrypted transport,
retention/rotation, backup and restore, disk-health alarms, schema migration,
and recovery drills. None of those can be validated against this repository's
current MuJoCo-only lessons. Measure on the target computer before making
latency or task-success claims.
