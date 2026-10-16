"""MuJoCo Ant pose-memory replay, separate from control performance.

Requires the optional sim extra. Run from the repository root:
  python -m benchmarks.mujoco_ant_replay
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

from robomemory import OnboardMemory


def run(*, steps: int = 200, capture_every: int = 10) -> dict:
    if steps < 1 or capture_every < 1:
        raise ValueError("steps and capture_every must be positive")
    try:
        import gymnasium as gym
        import mujoco  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("install the project with the sim extra") from exc

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "ant.db"
        env = gym.make("Ant-v5")
        memory = OnboardMemory(path, robot_id="mujoco-ant-benchmark",
                               clock_domain="sim:ant-benchmark-7")
        samples: list[tuple[datetime, int, dict[str, float]]] = []
        query_ms: list[float] = []
        try:
            env.reset(seed=7)
            env.action_space.seed(11)
            sim_epoch = datetime.now(timezone.utc) + timedelta(minutes=1)

            def capture(step: int) -> None:
                data = env.unwrapped.data
                at = sim_epoch + timedelta(seconds=float(data.time))
                pose = dict(zip(("x", "y", "z"),
                                (float(v) for v in data.qpos[:3])))
                receipt = memory.submit_bundle(
                    bundle_key=f"ant-benchmark-7/{step}",
                    observation=dict(
                        observed_at=at, kind="simulated-proprioception",
                        text=f"Ant base pose step {step}", source="mujoco:Ant-v5/qpos",
                        episode_id="ant-benchmark-7", entity_id="ant/base",
                        frame_id="mujoco/world",
                        position=(pose["x"], pose["y"], pose["z"]),
                    ),
                    facts=[dict(subject="ant/base", predicate="pose_xyz_m",
                                value=pose, status="verified", confidence=1.0)],
                ).result(timeout=5)
                fact = receipt.fact_event_ids[0]
                samples.append((at, fact, pose))

            capture(0)
            zero = env.action_space.low * 0
            for step in range(1, steps + 1):
                env.step(zero)
                if step % capture_every == 0:
                    capture(step)

            matched = 0
            for at, fact, pose in samples:
                start = time.perf_counter()
                result = memory.retrieve_async(
                    valid_at=at, entity_id="ant/base",
                    predicates=("pose_xyz_m",), max_observations=1,
                ).result(timeout=5)
                query_ms.append((time.perf_counter() - start) * 1000)
                state = result.states[0][2]
                if (state.fact is not None and state.fact.event_id == fact
                        and state.fact.value == pose and result.evidence):
                    matched += 1
            last_at, last_id, last_pose = samples[-1]
            earlier = memory.query_state_async(
                subject="ant/base", predicate="pose_xyz_m",
                valid_at=last_at, known_event_id=samples[0][1],
            ).result(timeout=5)
            memory.close(timeout=5)
            restarted = OnboardMemory(path, robot_id="mujoco-ant-benchmark",
                                       clock_domain="sim:ant-benchmark-7")
            try:
                recovered = restarted.read_state(
                    subject="ant/base", predicate="pose_xyz_m", now=last_at,
                )
                restart_ok = (recovered.fact is not None and
                              recovered.fact.event_id == last_id and
                              recovered.fact.value == last_pose)
            finally:
                restarted.close(timeout=5)
            return {
                "benchmark": "mujoco_ant_replay_v1", "robot": "Gymnasium Ant-v5",
                "steps": steps, "samples": len(samples),
                "pose_retrieval_matches": matched,
                "prior_knowledge_matches": earlier.fact is not None and
                earlier.fact.event_id == samples[0][1],
                "restart_matches": restart_ok,
                "query_latency_ms_median": round(statistics.median(query_ms), 3),
                "query_latency_ms_max": round(max(query_ms), 3),
                "control_latency_evaluated": False,
            }
        finally:
            memory.close(timeout=5)
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--capture-every", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(steps=args.steps, capture_every=args.capture_every)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["pose_retrieval_matches"] == result["samples"]
             and result["prior_knowledge_matches"] and result["restart_matches"] else 1)
