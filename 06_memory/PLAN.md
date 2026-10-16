# Onboard memory architecture for an embodied agent

## Deployment goal

Run memory on the robot's onboard computer. Keep motor control independent of
disk latency, network availability, semantic model latency, and database
recovery. The current repository has no perception stack, object tracker, or
ROS 2 package, so this work builds a usable, ROS-agnostic runtime boundary;
it does not pretend to solve perception or provide hard real-time guarantees
from Python.

## Research translated into interfaces

| Primary work | Interface adopted here | Still outside this repository |
|---|---|---|
| [OpenEQA (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html) | Episodic lookup at an explicit world and knowledge time, with source records | OpenEQA benchmark ingestion and answer scoring |
| [Embodied-RAG (2024)](https://arxiv.org/abs/2409.18313) | Bounded retrieval across entities, observations and relations | Learned hierarchical summaries and multimodal embeddings |
| [GraphEQA (CoRL 2025)](https://proceedings.mlr.press/v305/saxena25a.html) | Directed, evidence-backed scene relations; bounded subgraph traversal | Online metric-semantic graph construction and image reasoning |
| [Mind Palace (CoRL 2025)](https://proceedings.mlr.press/v305/ginting25a.html) | Explicit unresolved results for a planner to request further observation | Navigation and value-of-information exploration policy |
| [MemoryEQA (2025)](https://arxiv.org/abs/2505.13948) | Shared retrieval result for question answering, planning and VLA context | Learned module-specific retrieval policies |

These works guide interfaces; this implementation does not claim their model
accuracy or reproduce their benchmark results.

The robot-facing data path is:

```mermaid
flowchart LR
  P[RGB-D, audio, pose, sensors] --> A[Observation adapter / tracker]
  A --> E[(Append-only bitemporal ledger)]
  E --> O[Episodic observations]
  E --> G[Scene relation facts]
  E --> F[Semantic state facts]
  O --> R[Bounded evidence retrieval]
  G --> R
  F --> R
  R --> Q[Question answering and planner]
  R --> V[VLA context]
  F --> H[Immutable hot snapshot]
  H --> C[Control path]
```

The observation adapter is a contract for upstream perception: timestamp,
robot and episode identity, stable entity ID, source, optional frame and
position, and a URI into durable media. There is no RGB-D or audio decoder in
this repository. Relations are directed fact keys of the form
`relation:<name>:<target>` with a source entity as subject. Each assertion
references a stored observation. A changed relation must be retracted or have
an explicit validity end; otherwise both edges remain as distinct evidence.

`MemoryRetriever` pins one event cursor per request and retrieves a bounded
subgraph, state facts and observations. Its context includes status,
confidence, event IDs, evidence URIs and a `usable` flag. Only a verified,
unexpired assertion that passes the caller's optional freshness limit is
usable. Dynamic predicates require an explicit freshness limit. This retrieval
is off the control path.
The query API uses structured filters and full-text terms; it does not expose
arbitrary SQL or claim natural-language reasoning. A language model can
translate questions into these filters and cite the returned evidence, but
its answers must be evaluated separately.

## Runtime paths

### 1. Control path: bounded RAM read

The action/controller loop reads one immutable latest snapshot by
(robot_id, entity_id, predicate). It performs no SQL, filesystem operation,
network call, model call, or wait. Snapshot replacement happens after a
background commit. Missing, unknown, expired, or stale state is explicit and
must route to the controller's conservative fallback.

The snapshot is capped by max_hot_keys and history_per_key, with at most 4096
facts. Eviction returns missing, never a fabricated old value. A snapshot
carries a generation and event cursor. Its
read is O(1) in this Python process, but Python scheduling and garbage
collection are not hard real-time. A hard real-time C++ controller should read
a fixed-size shared-memory page or a latest-value ROS 2 message (KEEP_LAST,
depth 1), with a freshness limit.

### 2. Onboard memory service: bounded background work

OnboardMemory is scoped to one robot identity. Sensor/perception/planning
workers submit observations and assertions to a bounded queue. One writer
serializes durable writes; saturation rejects immediately with
MemoryQueueFull and increments a metric. Configured reserve slots are
available to facts, retractions, and explicitly critical observations;
best-effort observations cannot consume that reserve. Callers must handle
rejection and raise an operator fault for safety-critical evidence. They must
never wait for the queue in a control callback.

Accepted submissions detach caller-owned data and enforce a per-event byte
limit before queueing. Keep exactly one runtime writer per robot identity;
external writes require restart hydration because they do not update the hot
snapshot. Startup hydration loads bounded live histories without assuming the
wall clock is the simulator's valid-time clock.

Historical state, text, and spatial queries run on a separate bounded worker
pool. Saturation returns MemoryQueryBusy. Callers consume receipts and query
futures only from non-control threads. The service publishes a new immutable
hot snapshot only after facts/retractions commit. At startup it hydrates the
bounded snapshot from the local ledger before the controller declares memory
ready.

```mermaid
flowchart LR
  S[Sensors and tracker] -->|bounded submit| W[Onboard memory writer]
  W -->|commit| DB[(Local bitemporal WAL)]
  DB -->|hydrate or update| H[Immutable RAM snapshot]
  H -->|current state read| C[Controller and policy]
  C --> A[Actuation with live safety checks]
  P[Planner or operator] -->|async structured query| Q[Bounded query pool]
  Q --> DB
  Q -->|facts plus evidence IDs| P
```

### 3. Durable local record and evidence

The bitemporal append-only SQLite WAL ledger stores observations, verified
facts, hypotheses, unknown states, and retractions. observed_at is world
time; event IDs and recorded_at are knowledge time. Queries take both clocks,
so delayed upload and correction preserve earlier knowledge views.

High-rate sensor streams, images, and video live in a recorder such as
rosbag2/MCAP or a bounded local media store. The ledger stores evidence
identifiers and timestamps, not camera frames or 500 Hz joint samples. Every
position carries a coordinate-frame ID; transform it at the source timestamp
before writing. The upstream tracker owns persistent entity IDs and
uncertainty.

The current timestamp contract is one timezone-aware robot clock converted to
UTC. Do not mix wall-clock, simulation, or device-monotonic values. A ROS 2
integration needs an explicit clock-domain and boot identifier in the stored
event schema before it combines data across clock resets or replay domains.

## State and safety rules

1. Only upstream physical verification may label a fact verified; detector
   output alone stays a hypothesis.
2. A newer hypothesis, explicit unknown, expired value, or stale value never
   silently falls back to an older verified state.
3. Memory is advisory context. The closed-loop controller still checks current
   sensors and safety interlocks before acting.
4. The controller must have a tested memoryless fallback. A restart begins
   with an empty snapshot until local-ledger hydration succeeds.
5. Time travel is an off-path query. The control snapshot answers current
   state only; historical/counterfactual reads use the durable query API.

## Failure behavior

- Queue full: reject explicitly and count; do not block the caller.
- Disk full or database error: fail the receipt, increment failure metrics,
  keep the last committed snapshot, and surface memory health as degraded.
- Worker/process restart: reopen WAL, hydrate the hot set, then declare
  ready. Caller-supplied idempotency keys make retries safe.
- Power loss: WAL transactions preserve committed records; deployment
  still needs disk health monitoring, rotation/backup, and recovery drills.
- Network loss: local control and memory continue; cloud sync is an
  optional later replication path.

## Build sequence and gates

1. **Implemented:** define the memory/control boundary and single-robot API.
2. **Implemented:** immutable hot snapshots, bounded ingestion with reserved
   capacity, asynchronous bitemporal queries, startup hydration, and counters.
3. **Implemented:** retain the append-only bitemporal ledger as source of truth.
4. **Implemented:** bounded evidence-grounded retrieval over episodic records,
   state facts and directed scene relations, with one knowledge cursor.
5. **Next:** integrate a real ROS 2 recorder, timestamped TF transforms, stable
   object tracker, and hardware identity; this repository currently has none.
6. **Hardware gate:** measure snapshot read p50/p99/max; write acceptance and
   commit delay; queue high-water/rejection counts; reboot and full-disk
   recovery; stale-state errors; retrieval precision; task success.
7. **Evaluation gate:** compare memory enabled, disabled, and conflicting/donor
   memory using the same task seed and current observation.

No throughput or task-success claim is made until those robot-side measurements
exist.
