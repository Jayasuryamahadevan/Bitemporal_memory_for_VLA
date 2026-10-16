"""Deterministic embodied-memory replay benchmark.

This is a local, synthetic benchmark of memory retrieval and temporal safety.
It is not an OpenEQA answer-accuracy score or a robot task-success score.
Run from the repository root: python -m benchmarks.warehouse_replay
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import statistics
import sys
import tempfile
import time

from robomemory import MemoryRetriever, MemoryStore


T0 = datetime(2026, 10, 16, 10, tzinfo=timezone.utc)


def run(*, distractors: int = 250, optimize: bool = True) -> dict:
    if not 0 <= distractors <= 50_000:
        raise ValueError("distractors must be between 0 and 50,000")
    with tempfile.TemporaryDirectory() as temp:
        store = MemoryStore(Path(temp) / "patrol.db")
        retriever = MemoryRetriever(store, robot_id="warehouse-1")

        def observe(entity: str, text: str, minute: int, *, robot="warehouse-1") -> int:
            return store.record_observation(
                robot_id=robot, observed_at=T0 + timedelta(minutes=minute),
                kind="simulated-rgbd", text=text, source="warehouse-replay",
                episode_id="patrol-1", entity_id=entity,
                frame_id="map", evidence_uri=f"mcap://patrol-1/{robot}/{entity}/{minute}",
            )

        for index in range(distractors):
            observe(f"crate-{index}", f"blue crate {index} in loading bay", -1)
        observe("blue-cup", "blue cup near the loading bay", 0)
        observe("red-cup", "red cup on wooden table", 0)
        table_seen = observe("wooden-table", "wooden table in kitchen", 0)
        shelf_seen = observe("storage-shelf", "shelf inside storage room", 0)
        observe("red-cup", "red cup on distant table", 0, robot="warehouse-2")
        cup_seen = store.search_observations(
            robot_id="warehouse-1", entity_id="red-cup", limit=1,
        )[0].event_id
        color = store.assert_fact(
            robot_id="warehouse-1", subject="red-cup", predicate="color",
            value="red", valid_from=T0, evidence_event_id=cup_seen,
        )
        old_edge = retriever.assert_relation(
            source_entity_id="red-cup", relation="on",
            target_entity_id="wooden-table", valid_from=T0,
            evidence_event_id=cup_seen, status="verified", confidence=1,
        )
        retriever.assert_relation(
            source_entity_id="wooden-table", relation="inside",
            target_entity_id="kitchen", valid_from=T0,
            evidence_event_id=table_seen, status="verified", confidence=1,
        )
        initial_cursor = retriever.assert_relation(
            source_entity_id="storage-shelf", relation="inside",
            target_entity_id="storage-room", valid_from=T0,
            evidence_event_id=shelf_seen, status="verified", confidence=1,
        )
        moved_seen = observe("red-cup", "red cup moved to storage shelf", 10)
        corrected = observe("red-cup", "old table detection corrected", 11)
        store.retract_fact(
            robot_id="warehouse-1", fact_event_id=old_edge,
            evidence_event_id=corrected, observed_at=T0 + timedelta(minutes=11),
            reason="object tracker corrected old position",
        )
        new_edge = retriever.assert_relation(
            source_entity_id="red-cup", relation="on",
            target_entity_id="storage-shelf",
            valid_from=T0 + timedelta(minutes=10),
            evidence_event_id=moved_seen, status="verified", confidence=1,
        )
        battery_seen = observe("red-cup", "battery state not observed", 13)
        store.assert_fact(
            robot_id="warehouse-1", subject="red-cup", predicate="battery",
            value=None, status="unknown", valid_from=T0 + timedelta(minutes=13),
            evidence_event_id=battery_seen,
        )

        if optimize:
            store.optimize_search()  # Offline import maintenance, off control path.

        cases: list[dict] = []

        def check(name: str, result, passed: bool, evidence_id: int | None = None) -> None:
            cases.append({"name": name, "passed": bool(passed),
                          "known_event_id": result.known_event_id,
                          "evidence_hit": evidence_id is None or evidence_id in
                          {obs.event_id for obs in result.observations + result.evidence}})
            cases[-1]["passed"] &= cases[-1]["evidence_hit"]

        timings: list[tuple[str, float]] = []

        def retrieve(label: str, **kwargs):
            start = time.perf_counter()
            result = retriever.retrieve(**kwargs)
            timings.append((label, (time.perf_counter() - start) * 1000))
            return result

        old = retrieve("old_question", valid_at=T0 + timedelta(minutes=1),
                       text="Where was the red cup?", predicates=("color",),
                       graph_hops=2, known_event_id=initial_cursor)
        check("question_to_episode", old,
              any(obs.event_id == cup_seen for obs in old.observations), cup_seen)
        check("lexical_precision", old,
              all(obs.entity_id == "red-cup" for obs in old.observations))
        check("two_hop_room", old,
              any(edge.target == "kitchen" for edge in old.relations), table_seen)
        before = retrieve("prior_knowledge", valid_at=T0 + timedelta(minutes=12),
                          entity_id="red-cup", graph_hops=1,
                          known_event_id=old_edge)
        check("prior_knowledge", before,
              {edge.target for edge in before.relations} == {"wooden-table"}, cup_seen)
        after = retrieve("corrected_question", valid_at=T0 + timedelta(minutes=12),
                         text="Where is the red cup now?", graph_hops=2)
        check("corrected_graph", after,
              {edge.target for edge in after.relations} ==
              {"storage-shelf", "storage-room"} and
              after.known_event_id >= new_edge, moved_seen)
        unknown = retrieve("unknown", valid_at=T0 + timedelta(minutes=14),
                           entity_id="red-cup", predicates=("battery",))
        check("unknown_abstention", unknown,
              unknown.context()["states"][0]["usable"] is False and
              unknown.states[0][2].status == "unknown")
        stale = retrieve("stale", valid_at=T0 + timedelta(minutes=14),
                         entity_id="red-cup", predicates=("color",))
        check("freshness_abstention", stale,
              stale.states[0][2].fact.event_id == color and
              stale.context(max_age=timedelta(minutes=2))["states"][0]
              ["status"] == "stale")
        missing = retrieve("missing", valid_at=T0 + timedelta(minutes=14),
                           text="Where is the green bottle?")
        check("missing_abstention", missing,
              not missing.observations and not missing.relations)
        check("robot_isolation", after,
              all(obs.robot_id == "warehouse-1" for obs in
                  after.observations + after.evidence))

        baseline = store.search_observations(
            robot_id="warehouse-1", text="Where was the red cup?",
            observed_until=T0 + timedelta(minutes=1),
        )
        report = {
            "benchmark": "warehouse_replay_v1",
            "kind": "synthetic deterministic memory replay",
            "fts_optimized_after_import": optimize,
            "observations": distractors + 8,
            "cases_passed": sum(case["passed"] for case in cases),
            "cases_total": len(cases),
            "cases": cases,
            "exact_all_word_baseline_hit": bool(baseline),
            "retrieval_latency_ms_median": round(statistics.median(ms for _, ms in timings), 3),
            "retrieval_latency_ms_max": round(max(ms for _, ms in timings), 3),
            "retrieval_latency_ms_by_query": {name: round(ms, 3) for name, ms in timings},
        }
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distractors", type=int, default=250)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(distractors=args.distractors, optimize=not args.no_optimize)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["cases_passed"] == result["cases_total"] else 1)
