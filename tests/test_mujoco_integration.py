"""Optional end-to-end smoke test against Gymnasium's MuJoCo Ant robot."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

try:
    import gymnasium as gym
    import mujoco  # noqa: F401 - require the Gymnasium MuJoCo extra as well.
except ImportError:  # Keep the core memory test suite usable without sim extras.
    gym = None

from robomemory import OnboardMemory


UTC = timezone.utc


@unittest.skipUnless(gym is not None, "install 06_memory/requirements-sim.txt")
class MuJoCoMemoryIntegrationTest(unittest.TestCase):
    def test_ant_sensor_states_are_queryable_across_simulated_time(self):
        with tempfile.TemporaryDirectory() as temp:
            robot_id = "mujoco-ant-smoke-test"
            env = gym.make("Ant-v5")
            memory = OnboardMemory(Path(temp) / "ant-memory.db", robot_id=robot_id)
            try:
                env.reset(seed=7)
                env.action_space.seed(11)
                sim_epoch = datetime.now(UTC)
                samples: list[tuple[datetime, int, tuple[float, float, float]]] = []

                def capture(step: int) -> None:
                    data = env.unwrapped.data
                    pose = tuple(float(value) for value in data.qpos[:3])
                    world_time = sim_epoch + timedelta(seconds=float(data.time))
                    evidence_id = memory.submit_observation(
                        observed_at=world_time,
                        kind="simulated-proprioception",
                        text=f"Ant base pose at simulation step {step}",
                        source="mujoco:Ant-v5/qpos",
                        episode_id="ant-smoke-7",
                        entity_id="ant/base",
                        frame_id="mujoco/world",
                        position=pose,
                        metadata={"simulation_time_s": float(data.time), "step": step},
                        idempotency_key=f"ant-smoke-7/step-{step}/observation",
                    ).result(timeout=3)
                    fact_id = memory.submit_fact(
                        subject="ant/base",
                        predicate="pose_xyz_m",
                        value={"x": pose[0], "y": pose[1], "z": pose[2]},
                        valid_from=world_time,
                        evidence_event_id=evidence_id,
                        status="verified",  # Ground truth from the simulator.
                        confidence=1.0,
                        idempotency_key=f"ant-smoke-7/step-{step}/pose",
                    ).result(timeout=3)
                    samples.append((world_time, fact_id, pose))

                capture(step=0)
                zero_action = env.action_space.low * 0
                for step in range(1, 21):
                    env.step(zero_action)
                    if step % 5 == 0:
                        capture(step)

                first_time, first_event, first_pose = samples[0]
                last_time, last_event, last_pose = samples[-1]
                self.assertNotEqual(first_pose, last_pose, "MuJoCo produced no pose change")

                # Replay the final world time using the first and last knowledge
                # cursors. This checks both clocks against real simulator events.
                earlier_knowledge = memory.query_state_async(
                    subject="ant/base", predicate="pose_xyz_m",
                    valid_at=last_time, known_event_id=first_event,
                ).result(timeout=3)
                latest_knowledge = memory.query_state_async(
                    subject="ant/base", predicate="pose_xyz_m",
                    valid_at=last_time, known_event_id=last_event,
                ).result(timeout=3)
                self.assertEqual(earlier_knowledge.fact.event_id, first_event)
                self.assertEqual(latest_knowledge.fact.event_id, last_event)
                self.assertEqual(latest_knowledge.fact.value,
                                 {"x": last_pose[0], "y": last_pose[1], "z": last_pose[2]})

                cached = memory.read_state(
                    subject="ant/base", predicate="pose_xyz_m", now=last_time,
                )
                self.assertEqual(cached.fact.event_id, last_event)
            finally:
                self.assertTrue(memory.close(timeout=3))
                env.close()


if __name__ == "__main__":
    unittest.main()
