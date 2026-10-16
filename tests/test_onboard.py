from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import unittest

from robomemory import (
    MemoryPayloadTooLarge, MemoryQueueFull, MemoryQueryBusy,
    MemoryRuntimeClosed, MemoryStore,
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

    def test_future_fact_and_prior_state_survive_restart(self):
        evidence = self.observation().result(timeout=3)
        earlier = self.fact(evidence, value="table").result(timeout=3)
        future_time = datetime.now(UTC) + timedelta(days=1)
        future_evidence = self.observation(
            observed_at=future_time, text="cup on future shelf",
        ).result(timeout=3)
        later = self.fact(
            future_evidence, value="shelf", valid_from=future_time,
        ).result(timeout=3)
        self.assertTrue(self.memory.close(timeout=3))

        restarted = OnboardMemory(self.path, robot_id="r1", history_per_key=2)
        try:
            before = restarted.read_state(
                subject="cup-7", predicate="location", now=future_time - timedelta(seconds=1),
            )
            after = restarted.read_state(
                subject="cup-7", predicate="location", now=future_time,
            )
            self.assertEqual(before.fact.event_id, earlier)
            self.assertEqual(after.fact.event_id, later)
        finally:
            self.assertTrue(restarted.close(timeout=3))

    def test_retractions_beyond_cache_history_reveal_older_live_fact(self):
        self.assertTrue(self.memory.close(timeout=3))
        memory = OnboardMemory(self.path, robot_id="r1", history_per_key=2)
        try:
            first_obs = self.observation(memory).result(timeout=3)
            oldest = memory.submit_fact(
                subject="cup-7", predicate="location", value="table",
                valid_from=T0, evidence_event_id=first_obs,
            ).result(timeout=3)
            recent = []
            for minute in range(1, 5):
                at = T0 + timedelta(minutes=minute)
                observation = self.observation(
                    memory, observed_at=at, text=f"cup at position {minute}",
                ).result(timeout=3)
                fact = memory.submit_fact(
                    subject="cup-7", predicate="location", value=f"spot-{minute}",
                    valid_from=at, evidence_event_id=observation,
                ).result(timeout=3)
                recent.append(fact)
            correction = self.observation(
                memory, observed_at=T0 + timedelta(minutes=5), text="tracker corrected",
            ).result(timeout=3)
            for fact in reversed(recent):
                memory.submit_retraction(
                    fact_event_id=fact, evidence_event_id=correction,
                    observed_at=T0 + timedelta(minutes=5), reason="false detection",
                ).result(timeout=3)
            answer = memory.read_state(
                subject="cup-7", predicate="location", now=T0 + timedelta(minutes=6),
            )
            self.assertEqual(answer.fact.event_id, oldest)
            self.assertEqual(answer.fact.value, "table")
        finally:
            self.assertTrue(memory.close(timeout=3))

    def test_enqueued_value_is_detached_and_payload_size_is_bounded(self):
        evidence = self.observation().result(timeout=3)
        entered = threading.Event()
        release = threading.Event()
        original = self.memory._store.assert_fact

        def blocked_fact(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test writer release timed out")
            return original(**kwargs)

        self.memory._store.assert_fact = blocked_fact
        value = {"pose": [1, 2, 3]}
        receipt = self.fact(evidence, value=value)
        self.assertTrue(entered.wait(2))
        value["pose"][0] = 999
        release.set()
        receipt.result(timeout=3)
        durable = self.memory.query_state_async(
            subject="cup-7", predicate="location", valid_at=T0,
        ).result(timeout=3)
        self.assertEqual(durable.fact.value, {"pose": [1, 2, 3]})

        with self.assertRaises(MemoryPayloadTooLarge):
            self.memory.submit_observation(
                observed_at=T0, kind="vision", text="x" * 70000, source="camera",
            )
        self.assertEqual(self.memory.stats().writes_accepted, 2)

    def test_invalid_memory_bounds_fail_at_startup(self):
        for kwargs in ({"history_per_key": 1001}, {"max_hot_keys": 1000},
                       {"max_write_bytes": 0}, {"write_queue_capacity": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OnboardMemory(self.path, robot_id="r1", **kwargs)

    def test_boot_cache_evicts_oldest_key_after_new_state(self):
        self.assertTrue(self.memory.close(timeout=3))
        store = MemoryStore(self.path)
        for subject in ("old", "new"):
            observation = store.record_observation(
                robot_id="r1", observed_at=T0, kind="vision",
                text=f"{subject} object", source="camera",
            )
            store.assert_fact(
                robot_id="r1", subject=subject, predicate="location",
                value="table", valid_from=T0, evidence_event_id=observation,
            )
        memory = OnboardMemory(self.path, robot_id="r1", max_hot_keys=2)
        try:
            observation = memory.submit_observation(
                observed_at=T0, kind="vision", text="latest object", source="camera",
            ).result(timeout=3)
            memory.submit_fact(
                subject="latest", predicate="location", value="shelf",
                valid_from=T0, evidence_event_id=observation,
            ).result(timeout=3)
            self.assertEqual(memory.read_state(
                subject="old", predicate="location", now=T0,
            ).status, "missing")
            self.assertEqual(memory.read_state(
                subject="new", predicate="location", now=T0,
            ).fact.value, "table")
            self.assertEqual(memory.read_state(
                subject="latest", predicate="location", now=T0,
            ).fact.value, "shelf")
        finally:
            self.assertTrue(memory.close(timeout=3))

    def test_failed_assertion_does_not_publish_and_writer_recovers(self):
        invalid = self.memory.submit_fact(
            subject="cup-7", predicate="location", value="table",
            valid_from=T0, evidence_event_id=999999,
        )
        with self.assertRaisesRegex(ValueError, "evidence must be"):
            invalid.result(timeout=3)
        self.assertEqual(self.memory.read_state(
            subject="cup-7", predicate="location", now=T0,
        ).status, "missing")
        self.assertEqual(self.memory.stats().writes_failed, 1)
        self.assertTrue(self.memory.stats().degraded)

        observation = self.observation().result(timeout=3)
        valid = self.fact(observation).result(timeout=3)
        self.assertEqual(self.memory.read_state(
            subject="cup-7", predicate="location", now=T0,
        ).fact.event_id, valid)

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
