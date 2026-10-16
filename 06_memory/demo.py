"""Small, deterministic demonstration of world-time and knowledge-time queries."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robomemory import MemoryStore


def main():
    t = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as directory:
        memory = MemoryStore(Path(directory) / "memory.db")
        seen = memory.record_observation(
            robot_id="arm-1", observed_at=t, kind="vision",
            text="red cup on table", source="front-camera", entity_id="cup-7",
            frame_id="map", position=(1.0, 2.0, 0.7),
        )
        first = memory.assert_fact(
            robot_id="arm-1", subject="cup-7", predicate="location",
            value="table", valid_from=t, evidence_event_id=seen,
        )
        later_t = t + timedelta(minutes=10)
        later_seen = memory.record_observation(
            robot_id="arm-1", observed_at=later_t, kind="vision",
            text="red cup in drawer", source="front-camera", entity_id="cup-7",
        )
        memory.assert_fact(
            robot_id="arm-1", subject="cup-7", predicate="location",
            value="drawer", valid_from=later_t, evidence_event_id=later_seen,
        )
        query_time = t + timedelta(minutes=15)
        before = memory.query_state(robot_id="arm-1", subject="cup-7",
                                    predicate="location", valid_at=query_time,
                                    known_event_id=first)
        after = memory.query_state(robot_id="arm-1", subject="cup-7",
                                   predicate="location", valid_at=query_time)
        print(f"At {query_time.isoformat()}, before new evidence: {before.fact.value}")
        print(f"At {query_time.isoformat()}, with new evidence: {after.fact.value}")
        print("Supporting observation:", memory.get_observation(
            robot_id="arm-1", event_id=after.fact.evidence_event_id).text)


if __name__ == "__main__":
    main()
