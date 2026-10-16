from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import time
import unittest

from robomemory import (
    MemoryQueueFull, MemoryQueryBusy, MemoryRuntimeClosed, MemoryStore,
    OnboardMemory,
)


UTC = timezone.utc
T0 = datetime(2026, 10, 16, 10, 0, tzinfo=UTC)


class OnboardMemoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "robot.db"
        self.memory = OnboardMemory(self.path, robot_id="r1")

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def observation(self, memory=None, *, observed_at=T0, text="cup on table",
                    idempotency_key=None):
        runtime = memory or self.memory
        return runtime.submit_observation(
            observed_at=observed_at, kind="vision", text=text,
            source="front-camera", entity_id="cup-7", idempotency_key=idempotency_key,
        )

    def fact(self, evidence_id, *, value="table", valid_from=T0,
             idempotency_key=None):
        return self.memory.submit_fact(
            subject="cup-7", predicate="location", value=value,
            valid_from=valid_from, evidence_event_id=evidence_id,
            idempotency_key=idempotency_key,
        )

    def test_snapshot_publishes_after_commit_and_async_query_matches(self):
        evidence = self.observation().result(timeout=3)
        self.assertEqual(self.memory.read_state(
            subject="cup-7", predicate="location", now=T0,
        ).status, "missing")

        fact_id = self.fact(evidence).result(timeout=3)
        answer = self.memory.read_state(
            subject="cup-7", predicate="location", now=T0 + timedelta(seconds=1),
        )
        self.assertEqual(answer.status, "verified")
        self.assertEqual(answer.fact.event_id, fact_id)
        async_answer = self.memory.query_state_async(
            subject="cup-7", predicate="location", valid_at=T0 + timedelta(seconds=1),
        ).result(timeout=3)
        self.assertEqual(async_answer.fact.value, "table")
        self.assertGreaterEqual(self.memory.stats().writes_completed, 2)

    def test_restart_hydrates_only_this_robot_and_retraction_updates_snapshot(self):
        evidence = self.observation().result(timeout=3)
        fact_id = self.fact(evidence).result(timeout=3)
        self.assertTrue(self.memory.close(timeout=3))

        other = MemoryStore(self.path)
        other_obs = other.record_observation(
            robot_id="r2", observed_at=T0, kind="vision", text="other robot",
            source="camera",
        )
        other.assert_fact(robot_id="r2", subject="cup-7", predicate="location",
                          value="shelf", valid_from=T0, evidence_event_id=other_obs)

        restarted = OnboardMemory(self.path, robot_id="r1")
        try:
            answer = restarted.read_state(
                subject="cup-7", predicate="location", now=T0 + timedelta(seconds=1),
            )
            self.assertEqual(answer.fact.event_id, fact_id)
            self.assertEqual(answer.fact.value, "table")
            with self.assertRaises(ValueError):
                restarted.read_state(robot_id="r2", subject="cup-7",
                                     predicate="location", now=T0)

            correction = restarted.submit_observation(
                observed_at=T0 + timedelta(minutes=1), kind="vision",
                text="cup moved", source="front-camera",
            ).result(timeout=3)
            restarted.submit_retraction(
                fact_event_id=fact_id, evidence_event_id=correction,
                observed_at=T0 + timedelta(minutes=1), reason="tracker correction",
            ).result(timeout=3)
            self.assertEqual(restarted.read_state(
                subject="cup-7", predicate="location", now=T0 + timedelta(seconds=1),
            ).status, "missing")
        finally:
            self.assertTrue(restarted.close(timeout=3))

    def test_idempotent_old_fact_replay_does_not_replace_newer_hot_state(self):
        old_evidence = self.observation(idempotency_key="frame-1").result(timeout=3)
        old_fact = self.fact(old_evidence, value="table", idempotency_key="fact-1").result(timeout=3)
        new_evidence = self.observation(
            observed_at=T0 + timedelta(minutes=1), text="cup in drawer",
            idempotency_key="frame-2",
        ).result(timeout=3)
        new_fact = self.fact(
            new_evidence, value="drawer", valid_from=T0 + timedelta(minutes=1),
            idempotency_key="fact-2",
        ).result(timeout=3)
        self.assertGreater(new_fact, old_fact)

        replay = self.fact(old_evidence, value="table", idempotency_key="fact-1")
        self.assertEqual(replay.result(timeout=3), old_fact)
        answer = self.memory.read_state(
            subject="cup-7", predicate="location", now=T0 + timedelta(minutes=2),
        )
        self.assertEqual(answer.fact.event_id, new_fact)
        self.assertEqual(answer.fact.value, "drawer")

    def test_bounded_queue_rejects_best_effort_but_preserves_reserved_slot(self):
        self.memory.close(timeout=3)
        memory = OnboardMemory(self.path, robot_id="r1", write_queue_capacity=2,
                               reserved_critical_slots=1)
        evidence = memory._store.record_observation(
            robot_id="r1", observed_at=T0, kind="vision", text="known cup",
            source="camera",
        )
        entered = threading.Event()
        release = threading.Event()
        original = memory._store.record_observation

        def blocked_observation(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test writer release timed out")
            return original(**kwargs)

        memory._store.record_observation = blocked_observation
        first = memory.submit_observation(
            observed_at=T0, kind="vision", text="first", source="camera",
        )
        self.assertTrue(entered.wait(2))
        second = memory.submit_observation(
            observed_at=T0, kind="vision", text="second", source="camera",
        )
        with self.assertRaises(MemoryQueueFull):
            memory.submit_observation(
                observed_at=T0, kind="vision", text="third", source="camera",
            )
        critical = memory.submit_fact(
            subject="cup-7", predicate="location", value="table",
            valid_from=T0, evidence_event_id=evidence,
        )
        release.set()
        first.result(timeout=3)
        second.result(timeout=3)
        critical.result(timeout=3)
        stats = memory.stats()
        self.assertEqual(stats.writes_rejected, 1)
        self.assertTrue(memory.close(timeout=3))

    def test_query_limit_is_nonblocking_and_close_rejects_new_work(self):
        self.memory.close(timeout=3)
        memory = OnboardMemory(self.path, robot_id="r1", query_workers=1,
                               max_inflight_queries=1)
        entered = threading.Event()
        release = threading.Event()
        original = memory._store.query_state

        def blocked_query(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test query release timed out")
            return original(**kwargs)

        memory._store.query_state = blocked_query
        first = memory.query_state_async(
            subject="cup-7", predicate="location", valid_at=T0,
        )
        self.assertTrue(entered.wait(2))
        with self.assertRaises(MemoryQueryBusy):
            memory.query_state_async(
                subject="cup-7", predicate="location", valid_at=T0,
            )
        release.set()
        self.assertEqual(first.result(timeout=3).status, "missing")
        self.assertTrue(memory.close(timeout=3))
        with self.assertRaises(MemoryRuntimeClosed):
            memory.submit_observation(
                observed_at=T0, kind="vision", text="closed", source="camera",
            )


if __name__ == "__main__":
    unittest.main()
