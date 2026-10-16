"""Command line interface for the durable embodied memory store."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import sys

from .store import MemoryStore


def _time(text: str) -> datetime:
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO 8601 timestamp: {text}") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone, e.g. Z")
    return value


def _output(value):
    if is_dataclass(value):
        return _output(asdict(value))
    if isinstance(value, dict):
        return {key: _output(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_output(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="SQLite database path")
    commands = parser.add_subparsers(dest="command", required=True)

    observe = commands.add_parser("observe", help="append a sourced observation")
    observe.add_argument("--robot", required=True)
    observe.add_argument("--at", type=_time, required=True)
    observe.add_argument("--kind", required=True)
    observe.add_argument("--text", required=True)
    observe.add_argument("--source", required=True)
    observe.add_argument("--episode")
    observe.add_argument("--entity")
    observe.add_argument("--evidence-uri")
    observe.add_argument("--frame")
    observe.add_argument("--position", type=float, nargs=3)
    observe.add_argument("--metadata", type=json.loads, default=None)
    observe.add_argument("--key")

    assertion = commands.add_parser("assert", help="assert an evidence-backed state")
    assertion.add_argument("--robot", required=True)
    assertion.add_argument("--subject", required=True)
    assertion.add_argument("--predicate", required=True)
    assertion.add_argument("--value", type=json.loads, required=True,
                           help="JSON value, e.g. '\"kitchen\"'")
    assertion.add_argument("--at", type=_time, required=True)
    assertion.add_argument("--until", type=_time)
    assertion.add_argument("--evidence", type=int, required=True)
    assertion.add_argument("--status", choices=("verified", "hypothesis", "unknown"),
                           default="verified")
    assertion.add_argument("--confidence", type=float, default=1.0)
    assertion.add_argument("--key")

    retract = commands.add_parser("retract", help="retract a fact with new evidence")
    retract.add_argument("--robot", required=True)
    retract.add_argument("--fact", type=int, required=True)
    retract.add_argument("--evidence", type=int, required=True)
    retract.add_argument("--at", type=_time, required=True)
    retract.add_argument("--reason", required=True)
    retract.add_argument("--key")

    state = commands.add_parser("state", help="query state at world and knowledge times")
    state.add_argument("--robot", required=True)
    state.add_argument("--subject", required=True)
    state.add_argument("--predicate", required=True)
    state.add_argument("--at", type=_time, required=True)
    state.add_argument("--known-at", type=_time)
    state.add_argument("--known-event-id", type=int)
    state.add_argument("--max-age-seconds", type=float)
    state.add_argument("--min-confidence", type=float, default=0.0)

    history = commands.add_parser("history", help="show fact assertions and retractions")
    history.add_argument("--robot", required=True)
    history.add_argument("--subject", required=True)
    history.add_argument("--predicate", required=True)
    history.add_argument("--known-at", type=_time)
    history.add_argument("--known-event-id", type=int)
    history.add_argument("--limit", type=int, default=100)

    search = commands.add_parser("search", help="search sourced observations")
    search.add_argument("--robot", required=True)
    search.add_argument("--text")
    search.add_argument("--from", dest="observed_from", type=_time)
    search.add_argument("--until", dest="observed_until", type=_time)
    search.add_argument("--known-at", type=_time)
    search.add_argument("--known-event-id", type=int)
    search.add_argument("--entity")
    search.add_argument("--episode")
    search.add_argument("--near", type=float, nargs=3)
    search.add_argument("--radius", type=float)
    search.add_argument("--frame")
    search.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    try:
        memory = MemoryStore(args.db)
        if args.command == "observe":
            result = {"event_id": memory.record_observation(
                robot_id=args.robot, observed_at=args.at, kind=args.kind,
                text=args.text, source=args.source, episode_id=args.episode,
                entity_id=args.entity, evidence_uri=args.evidence_uri,
                frame_id=args.frame, position=tuple(args.position) if args.position else None,
                metadata=args.metadata, idempotency_key=args.key,
            )}
        elif args.command == "assert":
            result = {"event_id": memory.assert_fact(
                robot_id=args.robot, subject=args.subject, predicate=args.predicate,
                value=args.value, valid_from=args.at, valid_until=args.until,
                evidence_event_id=args.evidence, status=args.status,
                confidence=args.confidence, idempotency_key=args.key,
            )}
        elif args.command == "retract":
            result = {"event_id": memory.retract_fact(
                robot_id=args.robot, fact_event_id=args.fact,
                evidence_event_id=args.evidence, observed_at=args.at,
                reason=args.reason, idempotency_key=args.key,
            )}
        elif args.command == "state":
            result = memory.query_state(
                robot_id=args.robot, subject=args.subject, predicate=args.predicate,
                valid_at=args.at, known_at=args.known_at,
                known_event_id=args.known_event_id,
                max_age=timedelta(seconds=args.max_age_seconds)
                if args.max_age_seconds is not None else None,
                min_confidence=args.min_confidence,
            )
        elif args.command == "history":
            result = memory.history(
                robot_id=args.robot, subject=args.subject, predicate=args.predicate,
                known_at=args.known_at, known_event_id=args.known_event_id,
                limit=args.limit,
            )
        else:
            result = memory.search_observations(
                robot_id=args.robot, text=args.text,
                observed_from=args.observed_from, observed_until=args.observed_until,
                known_at=args.known_at, known_event_id=args.known_event_id,
                entity_id=args.entity, episode_id=args.episode,
                near=tuple(args.near) if args.near else None,
                radius=args.radius, frame_id=args.frame, limit=args.limit,
            )
        print(json.dumps(_output(result), indent=2, ensure_ascii=False))
        return 0
    except (ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"memory: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
