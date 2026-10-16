# Queryable memory for an embodied agent

## Decision

Build the off-loop episodic/world-state record described as Loop D in
`research.md`. The existing repository has no perception pipeline, trained
policy, object tracker, or memory API. A learned recurrent policy would be an
unverifiable claim here. This layer can ingest their outputs later and can be
tested now with explicit observations.

## Requirements and invariants

1. **Two clocks.** `observed_at` says when something happened in the world;
   `recorded_at` and a monotonic event ID say when the robot learned it. A
   query accepts both a world time and a knowledge cutoff. Late uploads and
   corrected perceptions must never rewrite what the robot knew earlier.
2. **Evidence before belief.** Every asserted state points to a stored
   observation from the same robot. The record includes its source, optional
   media URI, frame, pose, and raw metadata. An observation is not silently
   promoted into a verified fact.
3. **Append-only correction.** Facts and retractions are immutable events.
   Retraction removes a claim from subsequent knowledge views, while earlier
   knowledge views retain it. Newer state observations supersede older ones by
   world time. An explicit `unknown` fact blocks a stale earlier value.
4. **Conservative read.** At a requested world time, pick the latest applicable
   assertion known by the cutoff. If it is unverified, unknown, too old, or
   expired, report that state rather than falling back to an earlier value.
   The caller may set a maximum age for safety-sensitive predicates.
5. **Queryable episodes.** Search observation text with SQLite FTS5 and filter
   by robot, time interval, entity, episode, and optional 3D radius in a named
   coordinate frame. Return evidence IDs and timestamps, not uncited prose.
6. **Replayability.** Every response includes the knowledge cursor. Tests
   cover late arrival, correction, cross-robot isolation, spatial filtering,
   idempotent writes, and restart persistence.

## Storage and integration

SQLite WAL with transactions, foreign keys, indexes, FTS5, and per-call
connections provides a durable single-host implementation with no external
service dependency. Database work is **off the robot control loop**. Controllers
consume a cached, validated state snapshot at their own cadence; memory does
not issue motor commands. An upstream object tracker must assign stable entity
IDs and perform coordinate-frame transforms. This store preserves their output
and its provenance; it cannot solve identity or perception errors by itself.

The API exposes `record_observation`, `assert_fact`, `retract_fact`,
`query_state`, `search_observations`, and `history`. A CLI exercises the same
API. Inputs use timezone-aware UTC timestamps and JSON values. The event ID is
the exact knowledge cursor; wall-clock cutoffs are supported for convenience.

## Verification

Run standard-library unit tests against temporary on-disk databases, then a
CLI smoke test across process restarts. This proves storage/query semantics,
not robot task success or learned memory quality. A future robot integration
must measure retrieval precision, stale-state errors, identity swaps, end-to-end
task success, and no-memory/counterfactual-memory baselines.
