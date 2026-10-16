from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from robomemory import MemoryStore


T0 = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)


class MemoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "robot.db"
        self.memory = MemoryStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def observation(self, at=T0, robot="r1", text="red cup on table",
                    entity="cup-7", position=(1.0, 2.0, 0.7), **kwargs):
        return self.memory.record_observation(
            robot_id=robot, observed_at=at, kind="vision", text=text,
            source="front-camera", entity_id=entity, frame_id="map",
            position=position, metadata={"detector": "v1"}, **kwargs,
        )

    def state(self, at=T0, **kwargs):
        return self.memory.query_state(
            robot_id="r1", subject="cup-7", predicate="location",
            valid_at=at, **kwargs,
        )

    def test_two_clocks_late_arrival_and_retraction(self):
        first_obs = self.observation()
        first = self.memory.assert_fact(
            robot_id="r1", subject="cup-7", predicate="location",
            value="table", valid_from=T0, evidence_event_id=first_obs,
        )
        self.assertEqual(self.state(T0 + timedelta(minutes=30)).fact.value, "table")

        later_obs = self.observation(T0 + timedelta(minutes=20), text="cup in drawer")
        later = self.memory.assert_fact(
            robot_id="r1", subject="cup-7", predicate="location",
            value="drawer", valid_from=T0 + timedelta(minutes=20),
            evidence_event_id=later_obs,
        )
        self.assertEqual(self.state(T0 + timedelta(minutes=30), known_event_id=first).fact.value,
                         "table")
        first_recorded_at = self.memory.history(
            robot_id="r1", subject="cup-7", predicate="location",
            known_event_id=first)[0]["fact"].recorded_at
        self.assertEqual(self.state(T0 + timedelta(minutes=30),
                                    known_at=first_recorded_at).fact.value, "table")
        self.assertEqual(self.state(T0 + timedelta(minutes=30)).fact.value, "drawer")

        # Evidence arrived late: world-time order differs from event-ID order.
        middle_obs = self.observation(T0 + timedelta(minutes=10), text="cup on cart")
        middle = self.memory.assert_fact(
            robot_id="r1", subject="cup-7", predicate="location",
            value="cart", valid_from=T0 + timedelta(minutes=10),
            evidence_event_id=middle_obs,
        )
        self.assertEqual(self.state(T0 + timedelta(minutes=15), known_event_id=later).fact.value,
                         "table")
        self.assertEqual(self.state(T0 + timedelta(minutes=15)).fact.value, "cart")
        self.assertEqual(self.state(T0 + timedelta(minutes=30)).fact.value, "drawer")

        correction_obs = self.observation(T0 + timedelta(minutes=40),
                                          text="drawer detection was false")
        retraction = self.memory.retract_fact(
            robot_id="r1", fact_event_id=later, evidence_event_id=correction_obs,
            observed_at=T0 + timedelta(minutes=40), reason="tracker corrected",
        )
        self.assertEqual(self.state(T0 + timedelta(minutes=30), known_event_id=later).fact.value,
                         "drawer")
        self.assertEqual(self.state(T0 + timedelta(minutes=30)).fact.value, "cart")
        history = self.memory.history(robot_id="r1", subject="cup-7", predicate="location")
        self.assertEqual({item["fact"].event_id for item in history}, {first, later, middle})
        self.assertEqual(next(item for item in history if item["fact"].event_id == later)
                         ["retracted_by"], retraction)

    def test_unknown_unverified_stale_and_expiry_block_old_state(self):
        obs = self.observation()
        self.memory.assert_fact(robot_id="r1", subject="cup-7", predicate="location",
                                value="table", valid_from=T0, evidence_event_id=obs)
        maybe = self.observation(T0 + timedelta(minutes=2), text="cup maybe moved")
        self.memory.assert_fact(robot_id="r1", subject="cup-7", predicate="location",
                                value="cart", valid_from=T0 + timedelta(minutes=2),
                                evidence_event_id=maybe, status="hypothesis",
                                confidence=0.4)
        self.assertEqual(self.state(T0 + timedelta(minutes=3)).status, "hypothesis")
        confirmed = self.observation(T0 + timedelta(minutes=4), text="cup no longer visible")
        self.memory.assert_fact(robot_id="r1", subject="cup-7", predicate="location",
                                value=None, valid_from=T0 + timedelta(minutes=4),
                                evidence_event_id=confirmed, status="unknown")
        self.assertEqual(self.state(T0 + timedelta(minutes=5)).status, "unknown")

        new_obs = self.observation(T0 + timedelta(minutes=6), text="cup on shelf")
        self.memory.assert_fact(robot_id="r1", subject="cup-7", predicate="location",
                                value="shelf", valid_from=T0 + timedelta(minutes=6),
                                valid_until=T0 + timedelta(minutes=10),
                                evidence_event_id=new_obs)
        self.assertEqual(self.state(T0 + timedelta(minutes=7)).fact.value, "shelf")
        self.assertEqual(self.state(T0 + timedelta(minutes=9),
                                    max_age=timedelta(minutes=1)).status, "stale")
        self.assertEqual(self.state(T0 + timedelta(minutes=11)).status, "expired")

    def test_search_provenance_time_and_coordinate_frame(self):
        first = self.observation(evidence_uri="file:///frames/001.jpg", episode_id="run-1")
        self.observation(T0 + timedelta(minutes=1), text="blue cup on table",
                         position=(8.0, 2.0, 0.7), episode_id="run-1")
        self.observation(T0, robot="r2", text="red cup on table", episode_id="run-1")
        results = self.memory.search_observations(
            robot_id="r1", text="red cup", entity_id="cup-7", episode_id="run-1",
            observed_from=T0, observed_until=T0 + timedelta(minutes=1),
            near=(1.1, 2.0, 0.7), radius=0.2, frame_id="map",
        )
        self.assertEqual([item.event_id for item in results], [first])
        self.assertEqual(results[0].evidence_uri, "file:///frames/001.jpg")
        self.assertEqual(results[0].source, "front-camera")
        self.assertEqual(self.memory.search_observations(
            robot_id="r1", text="red cup", known_event_id=0), [])
        self.assertEqual(self.memory.search_observations(
            robot_id="r1", near=(1, 2, .7), radius=.2, frame_id="other"), [])
        self.assertIsNone(self.memory.get_observation(robot_id="r2", event_id=first))

    def test_idempotency_persistence_and_evidence_isolation(self):
        first = self.observation(idempotency_key="camera-frame-1")
        self.assertEqual(self.observation(idempotency_key="camera-frame-1"), first)
        with self.assertRaisesRegex(ValueError, "different content"):
            self.observation(text="changed", idempotency_key="camera-frame-1")
        with self.assertRaisesRegex(ValueError, "same robot"):
            self.memory.assert_fact(robot_id="r2", subject="cup-7", predicate="location",
                                    value="table", valid_from=T0, evidence_event_id=first)
        fact = self.memory.assert_fact(robot_id="r1", subject="cup-7",
                                       predicate="location", value="table",
                                       valid_from=T0, evidence_event_id=first,
                                       idempotency_key="state-1")
        self.assertEqual(self.memory.assert_fact(robot_id="r1", subject="cup-7",
                         predicate="location", value="table", valid_from=T0,
                         evidence_event_id=first, idempotency_key="state-1"), fact)
        reopened = MemoryStore(self.path)
        self.assertEqual(reopened.query_state(robot_id="r1", subject="cup-7",
                         predicate="location", valid_at=T0).fact.value, "table")
        with sqlite3.connect(self.path) as db:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                db.execute("UPDATE facts SET value_json='\"tampered\"' WHERE event_id=?", (fact,))
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.observation(datetime(2026, 10, 16, 10, 0))

    def test_concurrent_writers_have_unique_cursors(self):
        def write(i):
            return self.memory.record_observation(
                robot_id="r1", observed_at=T0 + timedelta(seconds=i),
                kind="proprio", text=f"joint sample {i}", source="encoders",
                idempotency_key=f"joint-{i}",
            )
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(write, range(20)))
        self.assertEqual(len(set(ids)), 20)
        self.assertEqual(len(self.memory.search_observations(robot_id="r1", limit=20)), 20)


if __name__ == "__main__":
    unittest.main()
