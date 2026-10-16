from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from robomemory import MemoryRetriever, MemoryStore, OnboardMemory


T0 = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)


class RetrievalTest(unittest.TestCase):
    def test_graph_evidence_knowledge_time_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryStore(Path(temp) / "memory.db")
            retriever = MemoryRetriever(store, robot_id="r1")
            cup_seen = store.record_observation(
                robot_id="r1", observed_at=T0, kind="rgbd",
                text="red cup on kitchen table", source="camera/front",
                entity_id="cup", frame_id="map", position=(1, 2, 0.8),
                evidence_uri="mcap://run-1/frame-10",
            )
            table_seen = store.record_observation(
                robot_id="r1", observed_at=T0, kind="rgbd",
                text="table in kitchen", source="camera/front",
                entity_id="table", frame_id="map", position=(1, 2, 0),
            )
            store.assert_fact(robot_id="r1", subject="table", predicate="color",
                              value="brown", valid_from=T0,
                              evidence_event_id=table_seen)
            edge_id = retriever.assert_relation(
                source_entity_id="cup", relation="on", target_entity_id="table",
                valid_from=T0, evidence_event_id=cup_seen,
                status="verified", confidence=0.95,
            )
            old = retriever.retrieve(
                valid_at=T0, entity_id="cup", predicates=("color",),
                graph_hops=1, text="red cup", known_event_id=edge_id,
            )
            self.assertEqual([(h.source, h.relation, h.target) for h in old.relations],
                             [("cup", "on", "table")])
            self.assertEqual(old.states[1][2].fact.value, "brown")
            self.assertEqual(old.observations[0].event_id, cup_seen)
            self.assertEqual(old.evidence[0].evidence_uri, "mcap://run-1/frame-10")

            correction = store.record_observation(
                robot_id="r1", observed_at=T0 + timedelta(minutes=1),
                kind="rgbd", text="cup no longer on table",
                source="camera/front", entity_id="cup",
            )
            store.retract_fact(robot_id="r1", fact_event_id=edge_id,
                               evidence_event_id=correction,
                               observed_at=T0 + timedelta(minutes=1),
                               reason="tracker correction")
            new = retriever.retrieve(valid_at=T0, entity_id="cup",
                                     predicates=("color",), graph_hops=1)
            self.assertEqual(new.relations, ())
            self.assertEqual(len(new.states), 1)
            self.assertEqual(old.relations[0].state.fact.event_id, edge_id)

            hypothesis = retriever.assert_relation(
                source_entity_id="cup", relation="near", target_entity_id="sink",
                valid_from=T0 + timedelta(minutes=2),
                evidence_event_id=correction,
            )
            current = retriever.retrieve(
                valid_at=T0 + timedelta(minutes=3), entity_id="cup",
                predicates=("color",), graph_hops=1,
            )
            self.assertEqual(current.relations[0].state.fact.event_id, hypothesis)
            self.assertEqual(current.relations[0].state.status, "hypothesis")
            self.assertEqual(len(current.states), 1)  # uncertain edge is not traversed
            context = current.context(max_age=timedelta(minutes=1))
            self.assertEqual(context["relations"][0]["value"], None)
            self.assertEqual(context["states"][0]["status"], "missing")
            self.assertEqual(context["relations"][0]["evidence_event_id"], correction)

            foreign = store.record_observation(
                robot_id="r2", observed_at=T0, kind="rgbd",
                text="secret cup", source="other-camera", entity_id="cup",
            )
            text_only = retriever.retrieve(valid_at=T0, text="cup")
            self.assertNotIn(foreign, [obs.event_id for obs in text_only.observations])
            self.assertEqual(text_only.context()["observations"][0]["text"],
                             "red cup on kitchen table")

    def test_onboard_async_retrieval_and_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            with OnboardMemory(Path(temp) / "memory.db", robot_id="r1") as runtime:
                seen = runtime.submit_observation(
                    observed_at=T0, kind="rgbd", text="cup on table",
                    source="camera", entity_id="cup",
                ).result(timeout=2)
                runtime.submit_relation(
                    source_entity_id="cup", relation="on",
                    target_entity_id="table", valid_from=T0,
                    evidence_event_id=seen, status="verified",
                    confidence=1.0,
                ).result(timeout=2)
                result = runtime.retrieve_async(
                    valid_at=T0, entity_id="cup", graph_hops=1,
                    max_entities=1,
                ).result(timeout=2)
                self.assertTrue(result.truncated)
                self.assertEqual(result.relations[0].target, "table")
                self.assertEqual(result.known_event_id,
                                 runtime.snapshot.known_event_id)

    def test_question_text_discovers_entity_and_preserves_exact_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryStore(Path(temp) / "memory.db")
            retriever = MemoryRetriever(store, robot_id="r1")
            cup = store.record_observation(
                robot_id="r1", observed_at=T0, kind="rgbd",
                text="red cup beside sink", source="camera", entity_id="cup",
            )
            store.record_observation(
                robot_id="r1", observed_at=T0, kind="rgbd",
                text="blue cup on floor", source="camera", entity_id="blue-cup",
            )
            retriever.assert_relation(
                source_entity_id="cup", relation="beside", target_entity_id="sink",
                valid_from=T0, evidence_event_id=cup,
                status="verified", confidence=1,
            )
            result = retriever.retrieve(
                valid_at=T0, text="Where is the red cup?", graph_hops=1,
            )
            self.assertEqual([obs.event_id for obs in result.observations], [cup])
            self.assertEqual(result.relations[0].target, "sink")
            self.assertFalse(store.search_observations(
                robot_id="r1", text="Where is the red cup?"))


if __name__ == "__main__":
    unittest.main()
