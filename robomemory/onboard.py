"""Onboard runtime: RAM reads on the action path, durable work off-path.

The control loop must only call ``read_state``. Observation writes and queries
are submitted to bounded background workers and may be rejected immediately
when their queues are full. This Python runtime is suitable for an onboard
non-real-time process; hard real-time controllers should consume a fixed-size
snapshot over shared memory or a latest-value ROS 2 topic.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import copy
import json
import queue
import threading
import time

from .store import Fact, MemoryStore, StateAnswer


UTC = timezone.utc
Key = tuple[str, str, str]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_fact(fact: Fact) -> Fact:
    return replace(fact, value=_freeze(fact.value))


class MemoryQueueFull(RuntimeError):
    """A write was rejected because bounded queue capacity was exhausted."""


class MemoryQueryBusy(RuntimeError):
    """The query worker pool is at its configured in-flight limit."""


class MemoryRuntimeClosed(RuntimeError):
    pass


class MemoryPayloadTooLarge(ValueError):
    """A write exceeds the configured per-event payload limit."""


class WriteReceipt:
    """Completion handle. Waiting is only appropriate outside a control loop."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._value: Any = None
        self._error: BaseException | None = None

    def _resolve(self, value: Any = None, error: BaseException | None = None) -> None:
        self._value, self._error = value, error
        self._done.set()

    def result(self, timeout: float | None = None) -> Any:
        if not self._done.wait(timeout):
            raise TimeoutError("memory write has not completed")
        if self._error is not None:
            raise self._error
        return self._value

    @property
    def done(self) -> bool:
        return self._done.is_set()


@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable copy-on-write hot state, read without a database call."""

    generation: int
    published_at: datetime
    known_event_id: int
    facts: Mapping[Key, tuple[Fact, ...]]

    def read_state(
        self, *, robot_id: str, subject: str, predicate: str,
        max_age: timedelta | None = None, min_confidence: float = 0.0,
        now: datetime,
    ) -> StateAnswer:
        """Return current cached state in O(1); this method performs no I/O."""
        at = now
        if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        if max_age is not None and max_age.total_seconds() < 0:
            raise ValueError("max_age must be nonnegative")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        key = (robot_id, subject, predicate)
        candidates = self.facts.get(key, ())
        fact = next((item for item in candidates if item.valid_from <= at), None)
        if fact is None:
            return StateAnswer("missing", at, self.known_event_id, None)
        if fact.valid_until is not None and at >= fact.valid_until:
            status = "expired"
        elif max_age is not None and at - fact.valid_from > max_age:
            status = "stale"
        elif fact.status == "verified" and fact.confidence < min_confidence:
            status = "hypothesis"
        else:
            status = fact.status
        return StateAnswer(status, at, self.known_event_id, fact)


@dataclass(frozen=True)
class RuntimeStats:
    queued_writes: int
    queue_capacity: int
    queue_high_water: int
    writes_accepted: int
    writes_completed: int
    writes_failed: int
    writes_rejected: int
    degraded: bool
    snapshot_generation: int
    snapshot_keys: int


@dataclass
class _WriteJob:
    operation: str
    kwargs: dict[str, Any]
    receipt: WriteReceipt


class OnboardMemory:
    """Bounded asynchronous writer plus immutable RAM snapshot for one robot.

    Construct this in a lifecycle/non-control thread. The database and query
    workers use local storage only. The control loop should retain the latest
    ``snapshot`` reference and call ``snapshot.read_state``; it should never
    wait on a ``WriteReceipt`` or query ``Future``.
    """

    def __init__(
        self, path: str | Path, *, robot_id: str, write_queue_capacity: int = 256,
        reserved_critical_slots: int = 16, max_hot_keys: int = 512,
        query_workers: int = 2,
        max_inflight_queries: int = 16, history_per_key: int = 8,
        max_write_bytes: int = 65536,
    ) -> None:
        capacities = (write_queue_capacity, max_hot_keys, query_workers,
                      max_inflight_queries, history_per_key, max_write_bytes)
        if any(type(value) is not int or value < 1 for value in capacities):
            raise ValueError("runtime capacities must be positive")
        if max_hot_keys > 1000 or history_per_key > 1000:
            raise ValueError("max_hot_keys and history_per_key cannot exceed 1000")
        if max_hot_keys * history_per_key > 4096:
            raise ValueError("hot snapshot capacity cannot exceed 4096 facts")
        if type(reserved_critical_slots) is not int or not 0 <= reserved_critical_slots < write_queue_capacity:
            raise ValueError("reserved_critical_slots must be >= 0 and below queue capacity")
        if not isinstance(robot_id, str) or not robot_id.strip():
            raise ValueError("robot_id must be a nonempty string")
        self.robot_id = robot_id.strip()
        self._store = MemoryStore(path)
        self._write_queue: queue.Queue[_WriteJob] = queue.Queue(write_queue_capacity)
        self._reserved_critical_slots = reserved_critical_slots
        self._max_hot_keys = max_hot_keys
        self._history_per_key = history_per_key
        self._max_write_bytes = max_write_bytes
        now = datetime.now(UTC)
        cursor, boot_facts = self._store.hot_facts(
            robot_id=self.robot_id, max_keys=max_hot_keys,
            history_per_key=history_per_key,
        )
        initial_lists: dict[Key, list[Fact]] = {}
        for fact in boot_facts:
            key = (fact.robot_id, fact.subject, fact.predicate)
            initial_lists.setdefault(key, []).append(_freeze_fact(fact))
        initial = {key: tuple(facts) for key, facts in initial_lists.items()}
        self._snapshot = MemorySnapshot(1, now, cursor, MappingProxyType(initial))
        self._submit_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._query_futures: set[Future[Any]] = set()
        self._closed = False
        self._closing = threading.Event()
        self._accepted = self._completed = self._failed = self._rejected = 0
        self._queue_high_water = 0
        self._query_slots = threading.BoundedSemaphore(max_inflight_queries)
        self._query_pool = ThreadPoolExecutor(max_workers=query_workers,
                                              thread_name_prefix="robot-memory-query")
        self._writer = threading.Thread(target=self._writer_main,
                                        name="robot-memory-writer", daemon=False)
        self._writer.start()

    @property
    def snapshot(self) -> MemorySnapshot:
        """Latest immutable state pointer. No locks, database, or RPC on read."""
        return self._snapshot

    def read_state(
        self, *, subject: str, predicate: str, now: datetime,
        max_age: timedelta | None = None, min_confidence: float = 0.0,
        robot_id: str | None = None,
    ) -> StateAnswer:
        if robot_id is not None and robot_id != self.robot_id:
            raise ValueError("runtime can only read its configured robot")
        return self._snapshot.read_state(
            robot_id=self.robot_id, subject=subject, predicate=predicate, now=now,
            max_age=max_age, min_confidence=min_confidence,
        )

    def _robot_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs.setdefault("robot_id", self.robot_id)
        if kwargs["robot_id"] != self.robot_id:
            raise ValueError("runtime can only access its configured robot")
        return kwargs

    def _submit_write(self, operation: str, kwargs: dict[str, Any], *,
                      critical: bool) -> WriteReceipt:
        kwargs = self._prepare_write_kwargs(kwargs)
        with self._submit_lock:
            if self._closed:
                raise MemoryRuntimeClosed("memory runtime is closed")
            best_effort_capacity = (
                self._write_queue.maxsize - self._reserved_critical_slots
            )
            if operation == "observation" and not critical and (
                    self._write_queue.qsize() >= best_effort_capacity):
                with self._metrics_lock:
                    self._rejected += 1
                raise MemoryQueueFull(
                    "best-effort queue is full; reserved write capacity remains"
                )
            receipt = WriteReceipt()
            with self._metrics_lock:
                try:
                    self._write_queue.put_nowait(_WriteJob(operation, kwargs, receipt))
                except queue.Full as exc:
                    self._rejected += 1
                    raise MemoryQueueFull("memory write queue is full") from exc
                self._accepted += 1
                self._queue_high_water = max(
                    self._queue_high_water, self._write_queue.qsize()
                )
            return receipt

    def _prepare_write_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Detach caller-owned values and bound each queued job by bytes."""
        detached = copy.deepcopy(kwargs)
        size_view = {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in detached.items()
        }
        try:
            encoded = json.dumps(size_view, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("memory write fields must be JSON-compatible") from exc
        if len(encoded) > self._max_write_bytes:
            raise MemoryPayloadTooLarge(
                f"memory write exceeds {self._max_write_bytes} bytes"
            )
        return detached

    def submit_observation(self, *, critical: bool = False,
                           **kwargs: Any) -> WriteReceipt:
        """Queue a sourced observation. Resolve receipt off the control path."""
        return self._submit_write("observation", self._robot_kwargs(kwargs),
                                  critical=critical)

    def submit_fact(self, **kwargs: Any) -> WriteReceipt:
        """Queue an evidence-backed assertion; receipt yields its event ID."""
        return self._submit_write("fact", self._robot_kwargs(kwargs), critical=True)

    def submit_retraction(self, **kwargs: Any) -> WriteReceipt:
        return self._submit_write("retraction", self._robot_kwargs(kwargs), critical=True)

    def query_state_async(self, **kwargs: Any) -> Future[StateAnswer]:
        return self._submit_query(self._store.query_state, self._robot_kwargs(kwargs))

    def search_observations_async(self, **kwargs: Any) -> Future[list[Any]]:
        return self._submit_query(self._store.search_observations, self._robot_kwargs(kwargs))

    def history_async(self, **kwargs: Any) -> Future[list[dict[str, Any]]]:
        return self._submit_query(self._store.history, self._robot_kwargs(kwargs))

    def _submit_query(self, operation, kwargs):
        with self._submit_lock:
            if self._closed:
                raise MemoryRuntimeClosed("memory runtime is closed")
            if not self._query_slots.acquire(blocking=False):
                raise MemoryQueryBusy("query limit reached; caller was not blocked")
            try:
                future = self._query_pool.submit(operation, **kwargs)
            except BaseException:
                self._query_slots.release()
                raise
            with self._query_lock:
                self._query_futures.add(future)
            future.add_done_callback(self._query_finished)
            return future

    def _query_finished(self, future: Future[Any]) -> None:
        with self._query_lock:
            self._query_futures.discard(future)
        self._query_slots.release()

    def _writer_main(self) -> None:
        while not self._closing.is_set() or not self._write_queue.empty():
            try:
                job = self._write_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if job.operation == "observation":
                    event_id = self._store.record_observation(**job.kwargs)
                    self._advance_cursor(event_id)
                    result: Any = event_id
                elif job.operation == "fact":
                    event_id = self._store.assert_fact(**job.kwargs)
                    self._refresh_key(job.kwargs["robot_id"], job.kwargs["subject"],
                                      job.kwargs["predicate"], event_id)
                    result = event_id
                else:
                    event_id = self._store.retract_fact(**job.kwargs)
                    target = self._store.get_fact(
                        robot_id=job.kwargs["robot_id"],
                        event_id=job.kwargs["fact_event_id"],
                    )
                    if target is not None:
                        self._refresh_key(target.robot_id, target.subject,
                                          target.predicate, event_id)
                    else:
                        self._advance_cursor(event_id)
                    result = event_id
                with self._metrics_lock:
                    self._completed += 1
                job.receipt._resolve(result)
            except BaseException as exc:
                with self._metrics_lock:
                    self._failed += 1
                job.receipt._resolve(error=exc)
            finally:
                self._write_queue.task_done()

    def _advance_cursor(self, event_id: int) -> None:
        old = self._snapshot
        self._snapshot = MemorySnapshot(
            old.generation + 1, datetime.now(UTC), max(old.known_event_id, event_id), old.facts
        )

    def _refresh_key(self, robot_id: str, subject: str, predicate: str,
                     known_event_id: int) -> None:
        cursor, facts = self._store.active_facts_snapshot(
            robot_id=robot_id, subject=subject, predicate=predicate,
            limit=self._history_per_key,
        )
        self._publish_history(robot_id, subject, predicate, facts,
                              max(cursor, known_event_id))

    def _publish_history(self, robot_id: str, subject: str, predicate: str,
                         active_facts: list[Fact], known_event_id: int) -> None:
        key = (robot_id, subject, predicate)
        facts = tuple(_freeze_fact(fact) for fact in active_facts)
        old = self._snapshot
        updated = dict(old.facts)
        if facts:
            updated.pop(key, None)
            updated[key] = facts
            if len(updated) > self._max_hot_keys:
                # Eviction produces an explicit cache miss; it never returns an
                # older value for a key that the cache no longer owns.
                updated.pop(next(iter(updated)))
        else:
            updated.pop(key, None)
        self._snapshot = MemorySnapshot(
            old.generation + 1, datetime.now(UTC),
            max(old.known_event_id, known_event_id), MappingProxyType(updated),
        )

    def stats(self) -> RuntimeStats:
        with self._metrics_lock:
            accepted, completed, failed, rejected, high_water = (
                self._accepted, self._completed, self._failed, self._rejected,
                self._queue_high_water,
            )
        snapshot = self._snapshot
        return RuntimeStats(self._write_queue.qsize(), self._write_queue.maxsize,
                            high_water, accepted, completed, failed, rejected,
                            bool(failed or rejected), snapshot.generation,
                            len(snapshot.facts))

    def close(self, timeout: float | None = 5.0) -> bool:
        """Drain accepted writes and stop workers. Call only during lifecycle shutdown."""
        with self._submit_lock:
            self._closed = True
            self._closing.set()
        deadline = None if timeout is None else time.monotonic() + timeout
        self._writer.join(None if deadline is None else max(0.0, deadline-time.monotonic()))
        writer_stopped = not self._writer.is_alive()
        with self._query_lock:
            pending = set(self._query_futures)
        if deadline is None:
            wait(pending)
        elif pending:
            wait(pending, timeout=max(0.0, deadline-time.monotonic()))
        with self._query_lock:
            queries_stopped = not self._query_futures
        self._query_pool.shutdown(wait=False, cancel_futures=False)
        return writer_stopped and queries_stopped

    def __enter__(self) -> "OnboardMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
