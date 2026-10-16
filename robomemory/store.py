"""Off-loop observation ledger and evidence-backed world-state queries.

An event ID is the exact transaction-time cursor. `observed_at`/`valid_at`
are world time. Facts are immutable; later assertions or retractions revise
the view without erasing an earlier knowledge snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any


UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
SCHEMA_VERSION = 1


def _micros(value: datetime) -> int:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware datetimes")
    delta = value.astimezone(UTC) - EPOCH
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds)


def _datetime(value: int) -> datetime:
    return EPOCH + timedelta(microseconds=value)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _positive_limit(value: int) -> int:
    if not isinstance(value, int) or not 1 <= value <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")
    return value


@dataclass(frozen=True)
class Observation:
    event_id: int
    robot_id: str
    episode_id: str | None
    observed_at: datetime
    recorded_at: datetime
    kind: str
    text: str
    entity_id: str | None
    source: str
    evidence_uri: str | None
    frame_id: str | None
    position: tuple[float, float, float] | None
    metadata: Any


@dataclass(frozen=True)
class Fact:
    event_id: int
    robot_id: str
    subject: str
    predicate: str
    value: Any
    status: str
    valid_from: datetime
    valid_until: datetime | None
    recorded_at: datetime
    confidence: float
    evidence_event_id: int


@dataclass(frozen=True)
class StateAnswer:
    status: str  # verified, hypothesis, unknown, stale, expired, missing
    valid_at: datetime
    known_event_id: int
    fact: Fact | None


class MemoryStore:
    """Single-host SQLite store. Use outside hard real-time control threads."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("use a file path; memory is durable across connections")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            # Journal mode is persistent for the database. Set it once during
            # lifecycle startup rather than on every read connection.
            db.execute("PRAGMA journal_mode=WAL")
            existing = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('schema_meta','events')"
            )}
            if "schema_meta" in existing:
                row = db.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
                if row is None or row[0] != str(SCHEMA_VERSION):
                    raise RuntimeError("unsupported memory schema version")
            elif "events" in existing:
                raise RuntimeError("existing memory database has no schema version")
            db.executescript(f"""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    robot_id TEXT NOT NULL,
                    episode_id TEXT,
                    kind TEXT NOT NULL CHECK (kind IN ('observation','fact','retraction')),
                    observed_at_us INTEGER NOT NULL,
                    recorded_at_us INTEGER NOT NULL,
                    entity_id TEXT,
                    text TEXT,
                    source TEXT,
                    evidence_uri TEXT,
                    frame_id TEXT,
                    x REAL, y REAL, z REAL,
                    payload_json TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    idempotency_key TEXT,
                    UNIQUE (robot_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS events_time
                    ON events(robot_id, kind, observed_at_us DESC, id DESC);
                CREATE INDEX IF NOT EXISTS events_entity
                    ON events(robot_id, entity_id, observed_at_us DESC);
                CREATE INDEX IF NOT EXISTS events_recorded
                    ON events(recorded_at_us, id);
                CREATE INDEX IF NOT EXISTS events_space
                    ON events(robot_id, frame_id, x, y, z);
                CREATE VIRTUAL TABLE IF NOT EXISTS observation_fts USING fts5(
                    text, entity_id, content='', tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS facts (
                    event_id INTEGER PRIMARY KEY REFERENCES events(id),
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('verified','hypothesis','unknown')),
                    valid_from_us INTEGER NOT NULL,
                    valid_until_us INTEGER,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    evidence_event_id INTEGER NOT NULL REFERENCES events(id),
                    CHECK (valid_until_us IS NULL OR valid_until_us > valid_from_us)
                );
                CREATE INDEX IF NOT EXISTS facts_lookup
                    ON facts(subject, predicate, valid_from_us DESC, event_id DESC);
                CREATE TABLE IF NOT EXISTS retractions (
                    event_id INTEGER PRIMARY KEY REFERENCES events(id),
                    target_fact_event_id INTEGER NOT NULL REFERENCES facts(event_id),
                    evidence_event_id INTEGER NOT NULL REFERENCES events(id)
                );
                CREATE INDEX IF NOT EXISTS retractions_target
                    ON retractions(target_fact_event_id, event_id);
                CREATE TRIGGER IF NOT EXISTS events_no_update
                    BEFORE UPDATE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                    END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete
                    BEFORE DELETE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                    END;
                CREATE TRIGGER IF NOT EXISTS facts_no_update
                    BEFORE UPDATE ON facts BEGIN
                    SELECT RAISE(ABORT, 'facts are append-only');
                    END;
                CREATE TRIGGER IF NOT EXISTS facts_no_delete
                    BEFORE DELETE ON facts BEGIN
                    SELECT RAISE(ABORT, 'facts are append-only');
                    END;
                CREATE TRIGGER IF NOT EXISTS retractions_no_update
                    BEFORE UPDATE ON retractions BEGIN
                    SELECT RAISE(ABORT, 'retractions are append-only');
                    END;
                CREATE TRIGGER IF NOT EXISTS retractions_no_delete
                    BEFORE DELETE ON retractions BEGIN
                    SELECT RAISE(ABORT, 'retractions are append-only');
                    END;
                INSERT OR IGNORE INTO schema_meta(key, value)
                    VALUES ('version', '{SCHEMA_VERSION}');
                COMMIT;
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    @staticmethod
    def _append(
        db: sqlite3.Connection, *, robot_id: str, episode_id: str | None,
        kind: str, observed_at_us: int, entity_id: str | None, text: str | None,
        source: str | None, evidence_uri: str | None, frame_id: str | None,
        position: tuple[float, float, float] | None, payload: dict[str, Any],
        idempotency_key: str | None,
    ) -> tuple[int, bool]:
        request_hash = hashlib.sha256(_json(payload).encode()).hexdigest()
        if idempotency_key is not None:
            prior = db.execute(
                "SELECT id, request_hash FROM events WHERE robot_id=? AND idempotency_key=?",
                (robot_id, idempotency_key),
            ).fetchone()
            if prior is not None:
                if prior["request_hash"] != request_hash:
                    raise ValueError("idempotency key reused with different content")
                return prior["id"], False
        last = db.execute("SELECT recorded_at_us FROM events ORDER BY id DESC LIMIT 1").fetchone()
        recorded_at_us = max(_micros(datetime.now(UTC)), last[0] + 1 if last else 0)
        xyz = position if position is not None else (None, None, None)
        cursor = db.execute("""
            INSERT INTO events (
                robot_id, episode_id, kind, observed_at_us, recorded_at_us,
                entity_id, text, source, evidence_uri, frame_id, x, y, z,
                payload_json, request_hash, idempotency_key
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (robot_id, episode_id, kind, observed_at_us, recorded_at_us,
              entity_id, text, source, evidence_uri, frame_id, *xyz,
              _json(payload), request_hash, idempotency_key))
        return cursor.lastrowid, True

    @staticmethod
    def _evidence(db: sqlite3.Connection, robot_id: str, event_id: int) -> sqlite3.Row:
        row = db.execute(
            "SELECT id, episode_id FROM events WHERE id=? AND robot_id=? AND kind='observation'",
            (event_id, robot_id),
        ).fetchone()
        if row is None:
            raise ValueError("evidence must be an observation from the same robot")
        return row

    def record_observation(
        self, *, robot_id: str, observed_at: datetime, kind: str, text: str,
        source: str, episode_id: str | None = None, entity_id: str | None = None,
        evidence_uri: str | None = None, frame_id: str | None = None,
        position: tuple[float, float, float] | None = None,
        metadata: Any = None, idempotency_key: str | None = None,
    ) -> int:
        robot_id = _nonempty(robot_id, "robot_id")
        kind = _nonempty(kind, "kind")
        text = _nonempty(text, "text")
        source = _nonempty(source, "source")
        observed_at_us = _micros(observed_at)
        if position is not None:
            if frame_id is None or len(position) != 3 or not all(math.isfinite(v) for v in position):
                raise ValueError("position needs a frame_id and three finite coordinates")
            frame_id = _nonempty(frame_id, "frame_id")
            position = tuple(float(v) for v in position)
        elif frame_id is not None:
            frame_id = _nonempty(frame_id, "frame_id")
        payload = dict(robot_id=robot_id, observed_at_us=observed_at_us, kind=kind,
                       text=text, source=source, episode_id=episode_id,
                       entity_id=entity_id, evidence_uri=evidence_uri,
                       frame_id=frame_id, position=position, metadata=metadata)
        _json(payload)  # reject non-JSON / NaN data before opening a transaction
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event_id, inserted = self._append(
                db, robot_id=robot_id, episode_id=episode_id, kind="observation",
                observed_at_us=observed_at_us, entity_id=entity_id, text=text,
                source=source, evidence_uri=evidence_uri, frame_id=frame_id,
                position=position, payload=payload, idempotency_key=idempotency_key,
            )
            if inserted:
                db.execute("INSERT INTO observation_fts(rowid,text,entity_id) VALUES (?,?,?)",
                           (event_id, text, entity_id or ""))
            db.commit()
            return event_id

    def assert_fact(
        self, *, robot_id: str, subject: str, predicate: str, value: Any,
        valid_from: datetime, evidence_event_id: int, status: str = "verified",
        confidence: float = 1.0, valid_until: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        robot_id = _nonempty(robot_id, "robot_id")
        subject = _nonempty(subject, "subject")
        predicate = _nonempty(predicate, "predicate")
        if status not in ("verified", "hypothesis", "unknown"):
            raise ValueError("status must be verified, hypothesis, or unknown")
        if status == "unknown" and value is not None:
            raise ValueError("unknown state must have a null value")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        from_us = _micros(valid_from)
        until_us = _micros(valid_until) if valid_until is not None else None
        if until_us is not None and until_us <= from_us:
            raise ValueError("valid_until must be later than valid_from")
        payload = dict(robot_id=robot_id, subject=subject, predicate=predicate,
                       value=value, valid_from_us=from_us, valid_until_us=until_us,
                       evidence_event_id=evidence_event_id, status=status,
                       confidence=confidence)
        value_json = _json(value)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            evidence = self._evidence(db, robot_id, evidence_event_id)
            event_id, inserted = self._append(
                db, robot_id=robot_id, episode_id=evidence["episode_id"], kind="fact",
                observed_at_us=from_us, entity_id=subject, text=None,
                source=None, evidence_uri=None, frame_id=None, position=None,
                payload=payload, idempotency_key=idempotency_key,
            )
            if inserted:
                db.execute("""
                    INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)
                """, (event_id, subject, predicate, value_json, status,
                      from_us, until_us, confidence, evidence_event_id))
            db.commit()
            return event_id

    def retract_fact(
        self, *, robot_id: str, fact_event_id: int, evidence_event_id: int,
        observed_at: datetime, reason: str, idempotency_key: str | None = None,
    ) -> int:
        robot_id = _nonempty(robot_id, "robot_id")
        reason = _nonempty(reason, "reason")
        observed_at_us = _micros(observed_at)
        payload = dict(robot_id=robot_id, fact_event_id=fact_event_id,
                       evidence_event_id=evidence_event_id,
                       observed_at_us=observed_at_us, reason=reason)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            evidence = self._evidence(db, robot_id, evidence_event_id)
            target = db.execute("""
                SELECT f.event_id FROM facts f JOIN events e ON e.id=f.event_id
                WHERE f.event_id=? AND e.robot_id=?
            """, (fact_event_id, robot_id)).fetchone()
            if target is None:
                raise ValueError("fact does not belong to this robot")
            event_id, inserted = self._append(
                db, robot_id=robot_id, episode_id=evidence["episode_id"],
                kind="retraction", observed_at_us=observed_at_us,
                entity_id=None, text=reason, source=None, evidence_uri=None,
                frame_id=None, position=None, payload=payload,
                idempotency_key=idempotency_key,
            )
            if inserted:
                db.execute("INSERT INTO retractions VALUES (?,?,?)",
                           (event_id, fact_event_id, evidence_event_id))
            db.commit()
            return event_id

    @staticmethod
    def _cursor(db: sqlite3.Connection, known_at: datetime | None,
                known_event_id: int | None) -> int:
        if known_at is not None and known_event_id is not None:
            raise ValueError("specify known_at or known_event_id, not both")
        if known_event_id is not None:
            if not isinstance(known_event_id, int) or known_event_id < 0:
                raise ValueError("known_event_id must be a nonnegative integer")
            if known_event_id and db.execute(
                "SELECT 1 FROM events WHERE id=?", (known_event_id,)
            ).fetchone() is None:
                raise ValueError("known_event_id does not exist")
            return known_event_id
        if known_at is not None:
            row = db.execute("SELECT COALESCE(MAX(id),0) FROM events WHERE recorded_at_us<=?",
                             (_micros(known_at),)).fetchone()
        else:
            row = db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()
        return row[0]

    @staticmethod
    def _fact(row: sqlite3.Row) -> Fact:
        return Fact(
            event_id=row["event_id"], robot_id=row["robot_id"],
            subject=row["subject"], predicate=row["predicate"],
            value=json.loads(row["value_json"]), status=row["status"],
            valid_from=_datetime(row["valid_from_us"]),
            valid_until=_datetime(row["valid_until_us"]) if row["valid_until_us"] is not None else None,
            recorded_at=_datetime(row["recorded_at_us"]),
            confidence=row["confidence"], evidence_event_id=row["evidence_event_id"],
        )

    def query_state(
        self, *, robot_id: str, subject: str, predicate: str, valid_at: datetime,
        known_at: datetime | None = None, known_event_id: int | None = None,
        max_age: timedelta | None = None, min_confidence: float = 0.0,
    ) -> StateAnswer:
        robot_id = _nonempty(robot_id, "robot_id")
        subject = _nonempty(subject, "subject")
        predicate = _nonempty(predicate, "predicate")
        valid_us = _micros(valid_at)
        if max_age is not None and max_age.total_seconds() < 0:
            raise ValueError("max_age must be nonnegative")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        with self._connect() as db:
            cursor = self._cursor(db, known_at, known_event_id)
            row = db.execute("""
                SELECT f.*, e.robot_id, e.recorded_at_us FROM facts f
                JOIN events e ON e.id=f.event_id
                WHERE e.robot_id=? AND f.subject=? AND f.predicate=?
                  AND f.valid_from_us<=? AND f.event_id<=?
                  AND NOT EXISTS (
                    SELECT 1 FROM retractions r
                    WHERE r.target_fact_event_id=f.event_id AND r.event_id<=?
                  )
                ORDER BY f.valid_from_us DESC, f.event_id DESC LIMIT 1
            """, (robot_id, subject, predicate, valid_us, cursor, cursor)).fetchone()
        if row is None:
            return StateAnswer("missing", valid_at, cursor, None)
        fact = self._fact(row)
        if fact.valid_until is not None and valid_us >= _micros(fact.valid_until):
            status = "expired"
        elif max_age is not None and valid_at - fact.valid_from > max_age:
            status = "stale"
        elif fact.status == "verified" and fact.confidence < min_confidence:
            status = "hypothesis"
        else:
            status = fact.status
        return StateAnswer(status, valid_at, cursor, fact)

    def history(
        self, *, robot_id: str, subject: str, predicate: str,
        known_at: datetime | None = None, known_event_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.history_snapshot(
            robot_id=robot_id, subject=subject, predicate=predicate,
            known_at=known_at, known_event_id=known_event_id, limit=limit,
        )[1]

    def history_snapshot(
        self, *, robot_id: str, subject: str, predicate: str,
        known_at: datetime | None = None, known_event_id: int | None = None,
        limit: int = 100,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Read history and its transaction cursor from one SQLite snapshot."""
        robot_id = _nonempty(robot_id, "robot_id")
        subject = _nonempty(subject, "subject")
        predicate = _nonempty(predicate, "predicate")
        _positive_limit(limit)
        with self._connect() as db:
            db.execute("BEGIN")
            cursor = self._cursor(db, known_at, known_event_id)
            rows = db.execute("""
                SELECT f.*, e.robot_id, e.recorded_at_us,
                       (SELECT MIN(r.event_id) FROM retractions r
                        WHERE r.target_fact_event_id=f.event_id AND r.event_id<=?)
                       AS retracted_by
                FROM facts f JOIN events e ON e.id=f.event_id
                WHERE e.robot_id=? AND f.subject=? AND f.predicate=? AND f.event_id<=?
                ORDER BY f.valid_from_us DESC, f.event_id DESC LIMIT ?
            """, (cursor, robot_id, subject, predicate, cursor, limit)).fetchall()
            result = [{"fact": self._fact(row), "retracted_by": row["retracted_by"]}
                      for row in rows]
            db.commit()
        return cursor, result

    def get_fact(self, *, robot_id: str, event_id: int) -> Fact | None:
        """Fetch one immutable assertion by its event ID."""
        with self._connect() as db:
            row = db.execute("""
                SELECT f.*, e.robot_id, e.recorded_at_us FROM facts f
                JOIN events e ON e.id=f.event_id
                WHERE f.event_id=? AND e.robot_id=?
            """, (event_id, robot_id)).fetchone()
        return self._fact(row) if row else None

    def knowledge_cursor(
        self, *, known_at: datetime | None = None,
        known_event_id: int | None = None,
    ) -> int:
        """Pin one knowledge horizon for a multi-part retrieval request."""
        with self._connect() as db:
            return self._cursor(db, known_at, known_event_id)

    def query_entity_facts(
        self, *, robot_id: str, subject: str, valid_at: datetime,
        known_event_id: int, predicate_prefix: str = "", limit: int = 100,
    ) -> list[Fact]:
        """Latest live assertion per predicate for one entity and horizon.

        Callers must inspect status, confidence, age and expiry before using
        these assertions for planning. A newer uncertain assertion still wins.
        """
        robot_id = _nonempty(robot_id, "robot_id")
        subject = _nonempty(subject, "subject")
        _positive_limit(limit)
        if not isinstance(predicate_prefix, str):
            raise ValueError("predicate_prefix must be a string")
        valid_us = _micros(valid_at)
        with self._connect() as db:
            cursor = self._cursor(db, None, known_event_id)
            rows = db.execute("""
                WITH ranked AS (
                    SELECT f.*, e.robot_id, e.recorded_at_us,
                           ROW_NUMBER() OVER (
                               PARTITION BY f.predicate
                               ORDER BY f.valid_from_us DESC, f.event_id DESC
                           ) AS state_rank
                    FROM facts f JOIN events e ON e.id=f.event_id
                    WHERE e.robot_id=? AND f.subject=? AND f.valid_from_us<=?
                      AND f.event_id<=?
                      AND substr(f.predicate, 1, length(?))=?
                      AND NOT EXISTS (
                          SELECT 1 FROM retractions r
                          WHERE r.target_fact_event_id=f.event_id
                            AND r.event_id<=?
                      )
                )
                SELECT * FROM ranked WHERE state_rank=1
                ORDER BY predicate ASC LIMIT ?
            """, (robot_id, subject, valid_us, cursor, predicate_prefix,
                  predicate_prefix, cursor, limit)).fetchall()
        return [self._fact(row) for row in rows]

    def active_facts_snapshot(
        self, *, robot_id: str, subject: str, predicate: str, limit: int = 8,
    ) -> tuple[int, list[Fact]]:
        """Return live assertions and a matching knowledge cursor for one key."""
        robot_id = _nonempty(robot_id, "robot_id")
        subject = _nonempty(subject, "subject")
        predicate = _nonempty(predicate, "predicate")
        _positive_limit(limit)
        with self._connect() as db:
            db.execute("BEGIN")
            cursor = self._cursor(db, None, None)
            rows = db.execute("""
                SELECT f.*, e.robot_id, e.recorded_at_us FROM facts f
                JOIN events e ON e.id=f.event_id
                WHERE e.robot_id=? AND f.subject=? AND f.predicate=?
                  AND f.event_id<=?
                  AND NOT EXISTS (
                    SELECT 1 FROM retractions r
                    WHERE r.target_fact_event_id=f.event_id AND r.event_id<=?
                  )
                ORDER BY f.valid_from_us DESC, f.event_id DESC LIMIT ?
            """, (robot_id, subject, predicate, cursor, cursor, limit)).fetchall()
            result = [self._fact(row) for row in rows]
            db.commit()
        return cursor, result

    def hot_facts(
        self, *, robot_id: str, max_keys: int, history_per_key: int,
    ) -> tuple[int, list[Fact]]:
        """Hydrate bounded live histories, including facts valid in the future.

        Keys are ordered by oldest to newest live assertion so insertion-order
        cache eviction removes the oldest of the selected keys first.
        """
        robot_id = _nonempty(robot_id, "robot_id")
        _positive_limit(max_keys)
        _positive_limit(history_per_key)
        with self._connect() as db:
            db.execute("BEGIN")
            cursor = self._cursor(db, None, None)
            rows = db.execute("""
                WITH active AS (
                    SELECT f.*, e.robot_id, e.recorded_at_us,
                           ROW_NUMBER() OVER (
                               PARTITION BY f.subject, f.predicate
                               ORDER BY f.valid_from_us DESC, f.event_id DESC
                           ) AS rank_in_key,
                           MAX(f.event_id) OVER (
                               PARTITION BY f.subject, f.predicate
                           ) AS key_latest_event_id
                    FROM facts f JOIN events e ON e.id=f.event_id
                    WHERE e.robot_id=? AND f.event_id<=?
                      AND NOT EXISTS (
                        SELECT 1 FROM retractions r
                        WHERE r.target_fact_event_id=f.event_id AND r.event_id<=?
                      )
                ), selected AS (
                    SELECT subject, predicate, key_latest_event_id
                    FROM active WHERE rank_in_key=1
                    ORDER BY key_latest_event_id DESC LIMIT ?
                )
                SELECT active.* FROM active
                JOIN selected USING (subject, predicate)
                WHERE active.rank_in_key<=?
                ORDER BY selected.key_latest_event_id ASC, active.rank_in_key ASC
            """, (robot_id, cursor, cursor, max_keys, history_per_key)).fetchall()
            result = [self._fact(row) for row in rows]
            db.commit()
        return cursor, result

    def latest_facts(
        self, *, robot_id: str, valid_at: datetime, limit: int = 1000,
    ) -> tuple[int, list[Fact]]:
        """Load the latest non-retracted state per subject/predicate for boot."""
        robot_id = _nonempty(robot_id, "robot_id")
        _positive_limit(limit)
        valid_us = _micros(valid_at)
        with self._connect() as db:
            db.execute("BEGIN")
            cursor = db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
            rows = db.execute("""
                WITH candidates AS (
                    SELECT f.*, e.robot_id, e.recorded_at_us,
                           ROW_NUMBER() OVER (
                               PARTITION BY f.subject, f.predicate
                               ORDER BY f.valid_from_us DESC, f.event_id DESC
                           ) AS state_rank
                    FROM facts f JOIN events e ON e.id=f.event_id
                    WHERE e.robot_id=? AND f.valid_from_us<=? AND f.event_id<=?
                      AND NOT EXISTS (
                        SELECT 1 FROM retractions r
                        WHERE r.target_fact_event_id=f.event_id AND r.event_id<=?
                      )
                )
                SELECT * FROM candidates WHERE state_rank=1
                ORDER BY valid_from_us DESC, event_id DESC LIMIT ?
            """, (robot_id, valid_us, cursor, cursor, limit)).fetchall()
            result = [self._fact(row) for row in rows]
            db.commit()
        return cursor, result

    @staticmethod
    def _observation(row: sqlite3.Row) -> Observation:
        position = (row["x"], row["y"], row["z"]) if row["x"] is not None else None
        payload = json.loads(row["payload_json"])
        return Observation(
            event_id=row["id"], robot_id=row["robot_id"],
            episode_id=row["episode_id"], observed_at=_datetime(row["observed_at_us"]),
            recorded_at=_datetime(row["recorded_at_us"]), kind=payload["kind"],
            text=row["text"], entity_id=row["entity_id"], source=row["source"],
            evidence_uri=row["evidence_uri"], frame_id=row["frame_id"],
            position=position, metadata=payload["metadata"],
        )

    def search_observations(
        self, *, robot_id: str, text: str | None = None,
        observed_from: datetime | None = None, observed_until: datetime | None = None,
        known_at: datetime | None = None, known_event_id: int | None = None,
        entity_id: str | None = None, episode_id: str | None = None,
        near: tuple[float, float, float] | None = None,
        radius: float | None = None, frame_id: str | None = None,
        limit: int = 20, match_mode: str = "all",
    ) -> list[Observation]:
        robot_id = _nonempty(robot_id, "robot_id")
        _positive_limit(limit)
        if match_mode not in ("all", "any"):
            raise ValueError("match_mode must be 'all' or 'any'")
        if (near is None) != (radius is None):
            raise ValueError("near and radius must be supplied together")
        if near is not None:
            if frame_id is None or len(near) != 3 or not all(math.isfinite(v) for v in near):
                raise ValueError("spatial query needs a frame and three finite coordinates")
            if not math.isfinite(radius) or radius < 0:
                raise ValueError("radius must be finite and nonnegative")
            frame_id = _nonempty(frame_id, "frame_id")
        tokens = re.findall(r"\w+", text, re.UNICODE) if text is not None else []
        if text is not None and not tokens:
            raise ValueError("text query needs at least one searchable word")
        conditions = ["e.robot_id=?", "e.kind='observation'", "e.id<=?"]
        with self._connect() as db:
            cursor = self._cursor(db, known_at, known_event_id)
            params: list[Any] = [robot_id, cursor]
            if observed_from is not None:
                conditions.append("e.observed_at_us>=?")
                params.append(_micros(observed_from))
            if observed_until is not None:
                conditions.append("e.observed_at_us<?")
                params.append(_micros(observed_until))
            if entity_id is not None:
                conditions.append("e.entity_id=?")
                params.append(entity_id)
            if episode_id is not None:
                conditions.append("e.episode_id=?")
                params.append(episode_id)
            if near is not None:
                conditions.extend(("e.frame_id=?", "e.x BETWEEN ? AND ?",
                                   "e.y BETWEEN ? AND ?", "e.z BETWEEN ? AND ?",
                                   "((e.x-?)*(e.x-?)+(e.y-?)*(e.y-?)+(e.z-?)*(e.z-?))<=?"))
                params.extend((frame_id, near[0]-radius, near[0]+radius,
                               near[1]-radius, near[1]+radius,
                               near[2]-radius, near[2]+radius,
                               near[0], near[0], near[1], near[1], near[2], near[2], radius*radius))
            if tokens:
                conditions.append("observation_fts MATCH ?")
                operator = " AND " if match_mode == "all" else " OR "
                params.append(operator.join('"' + token + '"' for token in tokens))
                join = "JOIN observation_fts ON observation_fts.rowid=e.id"
                # FTS5 can use its optimized rank path with LIMIT. It maps to
                # BM25 by default and avoids explicit bm25() sorting work.
                order = "observation_fts.rank, e.observed_at_us DESC, e.id DESC"
            else:
                join = ""
                order = "e.observed_at_us DESC, e.id DESC"
            sql = ("SELECT e.* FROM events e " + join + " WHERE " +
                   " AND ".join(conditions) + " ORDER BY " + order + " LIMIT ?")
            rows = db.execute(sql, (*params, limit)).fetchall()
        return [self._observation(row) for row in rows]

    def optimize_search(self) -> None:
        """Compact the FTS index after bulk import, with the runtime stopped.

        This can take a write lock and should run only in an offline/idle
        maintenance window, never from an onboard query or control callback.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO observation_fts(observation_fts) VALUES ('optimize')")
            db.commit()

    def get_observation(self, *, robot_id: str, event_id: int) -> Observation | None:
        with self._connect() as db:
            row = db.execute("""
                SELECT * FROM events WHERE id=? AND robot_id=? AND kind='observation'
            """, (event_id, robot_id)).fetchone()
        return self._observation(row) if row else None
