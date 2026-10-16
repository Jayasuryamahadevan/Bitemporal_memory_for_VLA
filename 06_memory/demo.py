"""Show the onboard split between a RAM read and bitemporal disk queries."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robomemory import OnboardMemory


def main():
    t = datetime.now(timezone.utc) - timedelta(minutes=2)
    query_time = t + timedelta(minutes=15)
    with tempfile.TemporaryDirectory() as directory:
        with OnboardMemory(Path(directory) / "memory.db", robot_id="arm-1") as memory:
            seen = memory.submit_observation(
                observed_at=t, kind="vision", text="red cup on table",
                source="front-camera", entity_id="cup-7", frame_id="map",
                position=(1.0, 2.0, 0.7), idempotency_key="frame-1",
            )
            first = memory.submit_fact(
                subject="cup-7", predicate="location", value="table",
                valid_from=t, evidence_event_id=seen.result(timeout=2),
                status="verified", confidence=0.99,
            )
            first_event_id = first.result(timeout=2)

            later_t = t + timedelta(minutes=10)
            later_seen = memory.submit_observation(
                observed_at=later_t, kind="vision", text="red cup in drawer",
                source="front-camera", entity_id="cup-7",
                idempotency_key="frame-2",
            )
            memory.submit_fact(
                subject="cup-7", predicate="location", value="drawer",
                valid_from=later_t, evidence_event_id=later_seen.result(timeout=2),
                status="verified", confidence=0.97,
            ).result(timeout=2)

            # The action path reads the published RAM snapshot only.
            current = memory.read_state(
                subject="cup-7", predicate="location", now=datetime.now(timezone.utc),
            )
            print("Current cached state:", current.status, current.fact.value)

            # Historical queries run on the bounded query pool, not in control.
            then = memory.query_state_async(
                subject="cup-7", predicate="location", valid_at=query_time,
                known_event_id=first_event_id,
            ).result(timeout=2)
            now_known = memory.query_state_async(
                subject="cup-7", predicate="location", valid_at=query_time,
            ).result(timeout=2)
            print(f"At {query_time.isoformat()}, earlier knowledge:", then.fact.value)
            print(f"At {query_time.isoformat()}, after new evidence:", now_known.fact.value)
            print("Runtime health counters:", memory.stats())


if __name__ == "__main__":
    main()
