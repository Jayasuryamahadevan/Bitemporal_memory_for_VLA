"""Compare bounded RAM answers with the durable bitemporal query oracle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest

from robomemory import MemoryStore, OnboardMemory


class DifferentialMemoryTest(unittest.TestCase):
    def test_out_of_order_facts_and_retractions_match_durable_state(self):
        base = datetime(2026, 10, 16, 10, tzinfo=timezone.utc)
        rng = Random(17)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "differential.db"
            memory = OnboardMemory(path, robot_id="sim", max_hot_keys=8,
                                   history_per_key=64)
            store = MemoryStore(path)
            written: list[int] = []
            try:
                for index in range(30):
                    at = base + timedelta(seconds=rng.randrange(21))
                    observation = memory.submit_observation(
                        observed_at=at, kind="sensor", text=f"sample {index}",
                        source="sim",
                    ).result(timeout=3)
                    fact = memory.submit_fact(
                        subject="object", predicate="location", value=f"spot-{index}",
                        valid_from=at, evidence_event_id=observation,
                    ).result(timeout=3)
                    written.append(fact)
                    if index % 4 == 3:
                        target = written[rng.randrange(len(written))]
                        correction = memory.submit_observation(
                            observed_at=at, kind="correction",
                            text=f"correction {index}", source="sim",
                        ).result(timeout=3)
                        memory.submit_retraction(
                            fact_event_id=target, evidence_event_id=correction,
                            observed_at=at, reason="corrected",
                        ).result(timeout=3)

                    for second in range(21):
                        when = base + timedelta(seconds=second)
                        cached = memory.read_state(
                            subject="object", predicate="location", now=when,
                        )
                        durable = store.query_state(
                            robot_id="sim", subject="object", predicate="location",
                            valid_at=when,
                        )
                        self.assertEqual(
                            (cached.status, cached.fact.event_id if cached.fact else None),
                            (durable.status, durable.fact.event_id if durable.fact else None),
                            (index, second),
                        )
            finally:
                self.assertTrue(memory.close(timeout=3))


if __name__ == "__main__":
    unittest.main()
