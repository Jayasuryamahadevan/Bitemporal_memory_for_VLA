"""Evidence-grounded retrieval for planners, EQA tools, and VLA context.

This module does not infer geometry or turn model output into verified facts.
Upstream perception supplies stable entity IDs, frame transforms and evidence.
All methods run off the motor-control path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import re
from typing import Any

from .store import Fact, MemoryStore, Observation, StateAnswer


RELATION_PREFIX = "relation:"
_RELATION_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def relation_predicate(relation: str, target_entity_id: str) -> str:
    """Encode one directed edge as a distinct bitemporal fact key."""
    if not isinstance(relation, str) or not _RELATION_NAME.fullmatch(relation):
        raise ValueError("relation must be an ASCII name without punctuation")
    if not isinstance(target_entity_id, str) or not target_entity_id.strip():
        raise ValueError("target_entity_id must be nonempty")
    return f"{RELATION_PREFIX}{relation}:{target_entity_id.strip()}"


@dataclass(frozen=True)
class RelationHit:
    source: str
    relation: str
    target: str
    state: StateAnswer


@dataclass(frozen=True)
class RetrievalResult:
    robot_id: str
    valid_at: datetime
    known_event_id: int
    states: tuple[tuple[str, str, StateAnswer], ...]
    relations: tuple[RelationHit, ...]
    observations: tuple[Observation, ...]
    evidence: tuple[Observation, ...]
    truncated: bool

    def context(self, *, max_age: timedelta | None = None,
                min_confidence: float = 0.0) -> dict[str, Any]:
        """Build bounded, machine-readable context with provenance.

        Only verified, unexpired, sufficiently fresh assertions are marked
        usable. Every other assertion is reported as unresolved so an
        agent can seek new evidence instead of acting on an old value.
        """
        if max_age is not None and max_age.total_seconds() < 0:
            raise ValueError("max_age must be nonnegative")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")

        def entry(state: StateAnswer) -> dict[str, Any]:
            fact = state.fact
            assert fact is not None
            status = state.status
            if status == "verified" and fact.confidence < min_confidence:
                status = "hypothesis"
            if status == "verified" and max_age is not None and (
                self.valid_at - fact.valid_from > max_age
            ):
                status = "stale"
            if status == "verified" and fact.valid_until is not None and (
                self.valid_at >= fact.valid_until
            ):
                status = "expired"
            return {
                "status": status, "usable": status == "verified",
                "value": fact.value if status == "verified" else None,
                "fact_event_id": fact.event_id,
                "evidence_event_id": fact.evidence_event_id,
                "valid_from": fact.valid_from.isoformat(),
                "confidence": fact.confidence,
            }

        states = [{"entity_id": entity, "predicate": predicate,
                   **(entry(state) if state.fact is not None else
                      {"status": "missing", "usable": False, "value": None,
                       "fact_event_id": None, "evidence_event_id": None})}
                  for entity, predicate, state in self.states]
        edges = [{"source": hit.source, "relation": hit.relation,
                  "target": hit.target, **entry(hit.state)}
                 for hit in self.relations]
        evidence = [{"event_id": obs.event_id, "entity_id": obs.entity_id,
                     "source": obs.source, "evidence_uri": obs.evidence_uri,
                     "observed_at": obs.observed_at.isoformat(),
                     "frame_id": obs.frame_id} for obs in self.evidence]
        observations = [{"event_id": obs.event_id, "entity_id": obs.entity_id,
                         "kind": obs.kind, "text": obs.text,
                         "observed_at": obs.observed_at.isoformat(),
                         "source": obs.source, "frame_id": obs.frame_id,
                         "evidence_uri": obs.evidence_uri}
                        for obs in self.observations]
        return {"robot_id": self.robot_id, "valid_at": self.valid_at.isoformat(),
                "known_event_id": self.known_event_id, "states": states,
                "relations": edges, "observations": observations,
                "evidence": evidence,
                "truncated": self.truncated}


class MemoryRetriever:
    """Bounded semantic, episodic and graph retrieval from one local ledger."""

    def __init__(self, store: MemoryStore, *, robot_id: str):
        if not isinstance(robot_id, str) or not robot_id.strip():
            raise ValueError("robot_id must be nonempty")
        self.store = store
        self.robot_id = robot_id.strip()

    def assert_relation(
        self, *, source_entity_id: str, relation: str,
        target_entity_id: str, valid_from: datetime, evidence_event_id: int,
        status: str = "hypothesis", confidence: float = 0.5,
        valid_until: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        """Record a directed scene-graph edge with the normal fact evidence."""
        if status == "unknown":
            raise ValueError("retract a relation or assert an explicit state instead")
        return self.store.assert_fact(
            robot_id=self.robot_id, subject=source_entity_id,
            predicate=relation_predicate(relation, target_entity_id),
            value=target_entity_id.strip(), valid_from=valid_from,
            evidence_event_id=evidence_event_id, status=status,
            confidence=confidence, valid_until=valid_until,
            idempotency_key=idempotency_key,
        )

    def retrieve(
        self, *, valid_at: datetime, entity_id: str | None = None,
        predicates: tuple[str, ...] = (), text: str | None = None,
        graph_hops: int = 0, max_entities: int = 16, max_relations: int = 32,
        max_observations: int = 16, known_at: datetime | None = None,
        known_event_id: int | None = None,
    ) -> RetrievalResult:
        """Pin a knowledge cursor and retrieve a small, attributable subgraph."""
        if entity_id is None and text is None:
            raise ValueError("provide entity_id or text")
        if entity_id is not None and (not isinstance(entity_id, str)
                                      or not entity_id.strip()):
            raise ValueError("entity_id must be a nonempty string")
        if type(graph_hops) is not int or not 0 <= graph_hops <= 2:
            raise ValueError("graph_hops must be 0, 1 or 2")
        if any(type(v) is not int or not 1 <= v <= 100 for v in
               (max_entities, max_relations, max_observations)):
            raise ValueError("retrieval budgets must be integers from 1 to 100")
        if len(predicates) > 16 or any(not isinstance(p, str) or not p.strip()
                                       for p in predicates):
            raise ValueError("predicates must contain at most 16 nonempty names")
        cursor = self.store.knowledge_cursor(known_at=known_at,
                                             known_event_id=known_event_id)
        frontier = [entity_id] if entity_id else []
        visited: set[str] = set()
        states: list[tuple[str, str, StateAnswer]] = []
        relations: list[RelationHit] = []
        truncated = False
        for depth in range(graph_hops + 1):
            next_frontier: list[str] = []
            for entity in frontier:
                if entity in visited:
                    continue
                if len(visited) >= max_entities:
                    truncated = True
                    break
                visited.add(entity)
                for predicate in predicates:
                    states.append((entity, predicate, self.store.query_state(
                        robot_id=self.robot_id, subject=entity,
                        predicate=predicate, valid_at=valid_at,
                        known_event_id=cursor,
                    )))
                if depth == graph_hops:
                    continue
                remaining = max_relations - len(relations)
                if remaining <= 0:
                    truncated = True
                    break
                edge_facts = self.store.query_entity_facts(
                    robot_id=self.robot_id, subject=entity, valid_at=valid_at,
                    known_event_id=cursor, predicate_prefix=RELATION_PREFIX,
                    limit=remaining + 1,
                )
                if len(edge_facts) > remaining:
                    truncated = True
                for fact in edge_facts[:remaining]:
                    parsed = fact.predicate[len(RELATION_PREFIX):].split(":", 1)
                    if (len(parsed) != 2 or not _RELATION_NAME.fullmatch(parsed[0])
                            or not parsed[1] or fact.value != parsed[1]):
                        continue
                    relation, target = parsed
                    status = "expired" if fact.valid_until is not None and (
                        valid_at >= fact.valid_until
                    ) else fact.status
                    relations.append(RelationHit(entity, relation, target,
                                                 StateAnswer(status, valid_at,
                                                             cursor, fact)))
                    if status == "verified" and target not in visited:
                        next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break
        found: dict[int, Observation] = {}
        for terms, subject in ((text, None), (None, entity_id)):
            if terms is None and subject is None:
                continue
            for obs in self.store.search_observations(
                robot_id=self.robot_id, text=terms, entity_id=subject,
                known_event_id=cursor, observed_until=valid_at + timedelta(microseconds=1),
                limit=max_observations + 1,
            ):
                found[obs.event_id] = obs
            if len(found) > max_observations:
                truncated = True
        observations = tuple(sorted(found.values(),
                                    key=lambda obs: (obs.observed_at, obs.event_id),
                                    reverse=True)[:max_observations])
        evidence_ids = {state.fact.evidence_event_id for _, _, state in states
                        if state.fact is not None}
        evidence_ids.update(hit.state.fact.evidence_event_id for hit in relations)
        evidence = tuple(obs for event_id in sorted(evidence_ids)
                         if (obs := self.store.get_observation(
                             robot_id=self.robot_id, event_id=event_id)) is not None)
        return RetrievalResult(self.robot_id, valid_at, cursor, tuple(states),
                               tuple(relations), observations, evidence, truncated)
