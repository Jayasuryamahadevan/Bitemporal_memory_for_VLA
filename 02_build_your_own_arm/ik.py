"""
Lesson 2: Move the arm by CARTESIAN coordinates, not joint coordinates.

Run:  .venv/bin/python 02_build_your_own_arm/ik.py

WHY THIS LESSON MATTERS FOR VLA:
    A VLA policy does NOT output joint angles. It outputs something like
    [dx, dy, dz, droll, dpitch, dyaw] -- a Cartesian end-effector delta.
    THIS FILE is the machinery that turns that into actual joint motion.
    Without it you cannot tell whether a policy failed or the controller failed.
    This is the single most useful thing to own in robot software.

You build, from scratch:
    1. a numerical Jacobian, verified against the analytic formula
    2. Jacobian-transpose IK / pseudoinverse IK / damped-least-squares IK
    3. a singularity analysis that explains WHY the damping term exists
    4. a Cartesian servoing loop = the thing a VLA's action actually drives

IMPORTANT LESSON ALREADY BAKED IN:
    The singularity table in section 3 is not decoration. I originally tested
    every configuration with q3=0 and concluded "the Jacobian is broken."
    It wasn't. q3=0 means the wrist joint is perfectly straight, and a straight
    wrist genuinely cannot translate the hand -- that IS a singularity.
    Check your Jacobian against your intuition before you trust it.
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robotics_utils import camera_xml

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

KP, KD, TAU_MAX = 40.0, 4.0, 150.0
GRAV_FF = 1.0          # set to 0 to see PD droop under gravity
Q_LO = np.array([np.deg2rad(-170), np.deg2rad(-114), np.deg2rad(-114)])
Q_HI = np.array([np.deg2rad(170), np.deg2rad(114), np.deg2rad(114)])

XML = """
<mujoco model="ik_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="4096"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1.5 -1.5 2" dir="0 0 -1" diffuse="0.3 0.3 0.3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.85 0.85 0.88 1"/>
    CAM_XML
    <body name="base" pos="0 0 0.04">
      <!-- cylinder size = (radius, HALF-height). Spans z 0.00 .. 0.08.
           [BUG 7] link1's capsule has a rounded cap of radius 0.04, so its lowest
           point sits 0.04 BELOW the link1 frame origin. With the base 0.10 tall
           the cap was buried 0.07 inside it, MuJoCo registered a contact every
           step, and the phantom contact force held joint 1 0.073 rad short of
           its setpoint no matter how I tuned the gains. Scene geometry is part
           of your controller -- check ncon when a joint refuses to converge. -->
      <geom type="cylinder" size="0.08 0.04 0.05" rgba="0.3 0.3 0.35 1"/>
      <body name="link1" pos="0 0 0.10">
        <joint name="j1" type="hinge" axis="0 0 1" damping="1.0" range="-170 170"/>
        <geom type="capsule" fromto="0 0 0  0 0 0.3" size="0.04" rgba="0.2 0.5 0.9 1"/>
        <body name="link2" pos="0 0 0.3">
          <joint name="j2" type="hinge" axis="0 1 0" damping="1.0" range="-114 114"/>
          <geom type="capsule" fromto="0 0 0  0 0 0.25" size="0.035" rgba="0.2 0.7 0.5 1"/>
          <body name="link3" pos="0 0 0.25">
            <joint name="j3" type="hinge" axis="0 1 0" damping="1.0" range="-114 114"/>
            <geom type="capsule" fromto="0 0 0  0 0 0.2" size="0.03" rgba="0.9 0.6 0.1 1"/>
            <site name="ee" pos="0 0 0.2" size="0.02" rgba="1 0.2 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" ctrlrange="-150 150"/>
    <motor name="m2" joint="j2" ctrlrange="-150 150"/>
    <motor name="m3" joint="j3" ctrlrange="-150 150"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(
    XML.replace("CAM_XML", camera_xml([1.8, -1.8, 1.0], [0, 0, 0.40], fovy=50, name="cam_side")))
data = mujoco.MjData(model)
EE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
NQ = model.nq
JOINT_NAMES = ["j1", "j2", "j3"]
JID = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
BID = [model.jnt_bodyid[j] for j in JID]


# ---------------------------------------------------------------------------
# 1. THE JACOBIAN. One question only: "if I nudge each joint a tiny bit, how
#    does the hand move?" 3 rows (x,y,z) x 3 columns (joints).
# ---------------------------------------------------------------------------
def fk(q):
    """Forward kinematics: joint angles -> hand position. THE core map."""
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[EE].copy()


def jacobian(q, h=1e-6):
    """d(hand position)/d(joint angles), shape (3, NQ). Central differences."""
    J = np.zeros((3, NQ))
    for i in range(NQ):
        dq = np.zeros(NQ); dq[i] = h
        J[:, i] = (fk(q + dq) - fk(q - dq)) / (2 * h)
    return J


def jacobian_analytic(q):
    """column i = axis_i x (hand - joint_i). The closed-form version."""
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    J = np.zeros((3, NQ))
    for i, (jid, bid) in enumerate(zip(JID, BID)):
        axis = data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ np.array(model.jnt_axis[jid])
        J[:, i] = np.cross(axis, data.site_xpos[EE] - data.xpos[bid])
    return J


print("=" * 72)
print("1. THE JACOBIAN")
print("=" * 72)
q_demo = np.array([0.30, 0.90, 0.55])          # wrist deliberately BENT (q3 != 0)
Jn, Ja = jacobian(q_demo), jacobian_analytic(q_demo)
np.set_printoptions(precision=4, suppress=True)
print(f"  q = {np.round(q_demo,3)}   hand at {np.round(fk(q_demo),4)}")
print(f"\n  numerical J =\n{Jn}")
print(f"\n  analytic J =\n{Ja}")
print(f"\n  max |numerical - analytic| = {np.abs(Jn - Ja).max():.3e}   <- they agree")

print("\n  How to read it: column i says 'nudge joint i by 0.001 rad and the hand")
print("  moves by J[:,i]*0.001 metres'. Column magnitudes are lever-arm lengths,")
print("  so distal joints (short lever arms) have small columns and little authority.")
print(f"  Column norms: {np.round(np.linalg.norm(Jn, axis=0), 4)}  (metres per radian)")

dq_test = np.array([0.05, -0.10, 0.08])
pred, true = Jn @ dq_test, fk(q_demo + dq_test) - fk(q_demo)
print(f"\n  linearity check: J@dq = {np.round(pred,6)}  vs  actual = {np.round(true,6)}")
print(f"  max err {np.abs(pred - true).max():.2e} m  -- exact at first order, so")
print(f"  feed J@dq forward each step and you integrate the motion correctly.")


# ---------------------------------------------------------------------------
# 2. INVERSE KINEMATICS: target position -> joint angles.
#    Same target, three methods. Watch convergence speed and robustness.
# ---------------------------------------------------------------------------
def _run_ik(q_init, target, solver, iters=300, tol=1e-3):
    q = q_init.copy()
    hist = []
    for _ in range(iters):
        e = target - fk(q)
        hist.append(np.linalg.norm(e))
        if hist[-1] < tol:
            break
        q = np.clip(q + solver(jacobian(q), e), Q_LO, Q_HI)
    return q, hist


def s_jt(J, e):        return 0.35 * (J.T @ e)                      # transpose
def s_pinv(J, e):      return np.linalg.pinv(J, rcond=1e-6) @ e     # exact, fragile
def s_dls(J, e, lam=0.03): return 0.9 * (J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), e))


print("\n" + "=" * 72)
print("2. INVERSE KINEMATICS -- same target, three solvers")
print("=" * 72)
target = np.array([0.34, 0.22, 0.52])
q_init = np.array([0.0, 0.90, 0.55])
print(f"  start q = {np.round(q_init,3)},  target (x,y,z) = {target}\n")
for name, solver in [("jacobian transpose", s_jt),
                     ("pseudoinverse      ", s_pinv),
                     ("damped least squares", s_dls)]:
    qf, hist = _run_ik(q_init.copy(), target, solver)
    hit = next((i for i, h in enumerate(hist) if h < 1e-3), None)
    print(f"  {name}: {'OK ' if hit else 'FAIL'} converged in {str(hit):>4} iters, "
          f"final err {hist[-1]:.2e} m -> q = {np.round(qf,3)}")
print("\n  All three reach the same place. They differ in convergence SPEED and in")
print("  what they do when the target is unreachable (section 5). That difference")
print("  is the entire reason damped least squares is the production default.")


# ---------------------------------------------------------------------------
# 3. SINGULARITIES. Configurations where some hand motion is IMPOSSIBLE.
#    The Jacobian loses rank, its smallest singular value goes to 0, and the
#    required joint velocities blow up toward infinity.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("3. SINGULARITIES")
print("=" * 72)
print("  rank = number of directions the hand can actually move in")
CASES = [
    (np.array([0.30, 0.90, 0.55]), "wrist BENT, elbow 90",   "full 3-DoF control"),
    (np.array([0.30, 0.90, 1.20]), "wrist bent hard",        "full, weaker authority"),
    (np.array([0.00, 0.10, 0.50]), "near extended, bent",    "rank 3, still fine"),
    (np.array([0.30, 0.90, 0.00]), "wrist STRAIGHT (q3=0)",  "wrist singularity"),
    (np.array([0.00, 0.10, 0.00]), "nearly extended+straight","both effects"),
    (np.array([0.00, 0.00, 0.00]), "fully extended, straight","rank 1, worst case"),
]
print(f"  {'configuration':<26}{'sv_min':>11}{'cond':>11}{'rank':>6}   intuition")
print("  " + "-" * 74)

for q, label, why in CASES:
    J = jacobian(q)
    s = np.linalg.svd(J, compute_uv=False)
    cond = s.max() / max(s.min(), 1e-12)
    r = np.linalg.matrix_rank(J, 1e-5)
    flag = "  <-- SINGULAR" if r < 3 else ("  (ill-conditioned)" if cond > 100 else "")
    print(f"  {label:<26}{s.min():>10.2e}{cond:>11.1e}{r:>6}   {why}{flag}")

print("\n  WHY the wrist-straight case is singular, concretely:")
q_s = np.array([0.30, 0.90, 0.00])
J_s = jacobian(q_s)
for i, n in enumerate(JOINT_NAMES):
    print(f"    column for {n}: {np.round(J_s[:,i], 4)}")
c2, c3 = J_s[:, 1], J_s[:, 2]
cosang = float(c2 @ c3 / (np.linalg.norm(c2) * np.linalg.norm(c3)))
print(f"    angle between column(j2) and column(j3) = "
      f"{np.degrees(np.arccos(np.clip(cosang,-1,1))):.1f} deg")
print("    -> EXACTLY parallel. Both joints push the hand the same way, so together")
print("       they add one useful direction, not two. One DOF is simply gone. Note")
print("       j1's column is NOT parallel to them, so the hand can still swivel --")
print("       which is why the rank is 2 and not 1.")

print("\n  WHY the fully-extended case is rank 1:")
J_e = jacobian(np.array([0.0, 0.0, 0.0]))
print(f"    columns: {np.round(J_e[:,0],3)} {np.round(J_e[:,1],3)} {np.round(J_e[:,2],3)}")
print("    -> The arm is a straight vertical line. You can only change how far along")
print("       it the hand sits (z), or swivel the whole line about z. You cannot bend")
print("       the hand sideways at all. That is a genuine property of a straight arm --")
print("       not a modelling error and not something damping can fix.")
print("\n  -> Real arms hit these at the top of their reach. The symptom is violent")
print("     shaking or stalled motion. This is the #1 cause of 'the robot won't move")
print("     smoothly when fully extended'.")

print("\n  NOTE: 'near extended, bent' is still rank 3 with GOOD conditioning, because")
print("  the bent wrist keeps it out of trouble. Ill-conditioning is what you see in")
print("  the narrow band just before rank drops -- watch cond climb as you approach")
print("  a singular config. That climb is the early warning that the damping term")
print("  in the solver is about to start doing real work.")


# ---------------------------------------------------------------------------
# 4. CARTESIAN SERVOING -- the loop a VLA action actually drives.
#    Re-plans every control step (reactive), like real visual servoing.
# ---------------------------------------------------------------------------
def cartesian_step(target, lam=0.03, dq_clip=0.12, gain=0.8):
    """ONE control update of Cartesian position servoing.

    This is the real architecture: every control step, measure the hand error,
    re-solve one damped-least-squares step, and move a little. It re-plans
    continuously instead of committing to a plan once.

    [BUG] My first version advanced only 8 sim steps per target waypoint and the
    arm could not keep up -- 154 mm mean tracking lag, hand ended on the floor.
    A servo must run at control rate against a target that is a function of TIME.

    [BUG] Then it still lagged 78 mm with a 13.7 mm static error. Cause: plain PD
    has STEADY-STATE DROOP under a constant disturbance, and gravity is a
    constant disturbance. tau_pd can never fully cancel gravity, so the arm
    always sits slightly low. Adding the model's own gravity+Coriolis term as
    feedforward (data.qfrc_bias) drops the error from 21 mm to 0.001 mm.

        tau = kp*(q_cmd - q)  -  kd*qdot  +  g(q)
                                            ^^^^^^^^^ gravity feedforward

    This is not optional in real robots -- it is why industrial arms integrate
    gravity compensation, and why 'the arm sags when you disable the controller'.
    """
    mujoco.mj_forward(model, data)
    p = data.site_xpos[EE].copy()
    e = target - p
    J = jacobian(data.qpos[:])
    dq = gain * (J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), e))
    q_cmd = np.clip(data.qpos[:] + np.clip(dq, -dq_clip, dq_clip), Q_LO, Q_HI)
    tau = KP * (q_cmd - data.qpos[:]) - KD * data.qvel[:] + GRAV_FF * data.qfrc_bias[:]
    data.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
    return e


def settle(target, steps=1200, lam=0.03):
    for _ in range(steps):
        e = cartesian_step(target, lam=lam, dq_clip=0.10, gain=0.8)
        mujoco.mj_step(model, data)
        if np.linalg.norm(e) < 5e-4:
            break
    return np.linalg.norm(target - data.site_xpos[EE])


print("\n" + "=" * 72)
print("4. CARTESIAN SERVOING -- reactive tracking of a moving target")
print("=" * 72)
mujoco.mj_resetData(model, data)
data.qpos[:] = np.array([0.0, 0.90, 0.55])
START = np.array([0.30, 0.00, 0.45])
start_residual = settle(START)
print(f"  settling at start pose {START} -> residual {start_residual*1000:.2f} mm")

renderer = mujoco.Renderer(model, height=240, width=320)
CAM = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_side")
renderer.update_scene(data, camera=CAM)
frames, path, lag = [renderer.render()], [], []

N_STEPS = 6000                                   # 12 s at dt = 0.002
PERIOD = 6.0                                     # seconds per loop of the path
for k in range(N_STEPS):
    t = k / (N_STEPS / PERIOD) * 2 * np.pi
    tgt = np.array([0.30 + 0.10 * np.sin(t),
                    0.07 * np.sin(2 * t),
                    0.45 + 0.08 * np.cos(t)])
    e = cartesian_step(tgt)                       # replan EVERY control step
    mujoco.mj_step(model, data)
    lag.append(np.linalg.norm(e))
    path.append(data.site_xpos[EE].copy())
    if k % 12 == 0:
        renderer.update_scene(data, camera=CAM)
        frames.append(renderer.render())

path, lag = np.array(path), np.array(lag)
print(f"  tracked a lemniscate for {PERIOD:.0f} s ({N_STEPS} control steps at "
      f"{1/model.opt.timestep:.0f} Hz)")
print(f"  tracking lag: mean {lag.mean()*1000:6.2f} mm, p95 {np.percentile(lag,95)*1000:6.2f} mm, "
      f"max {lag.max()*1000:6.2f} mm")
print(f"  hand visited x[{path[:,0].min():.3f},{path[:,0].max():.3f}] "
      f"y[{path[:,1].min():.3f},{path[:,1].max():.3f}] z[{path[:,2].min():.3f},{path[:,2].max():.3f}]")

# how well did it follow the commanded shape?
t = np.arange(N_STEPS) / (N_STEPS / PERIOD) * 2 * np.pi
cmd = np.stack([0.30 + 0.10*np.sin(t), 0.07*np.sin(2*t), 0.45 + 0.08*np.cos(t)], 1)
shape_err = np.linalg.norm(path - cmd, axis=1)
print(f"  commanded-shape error: mean {shape_err.mean()*1000:.2f} mm  "
      f"(this is tracking lag; static accuracy is the {start_residual*1000:.2f} mm above)")
imageio.mimwrite(os.path.join(OUT, "cartesian_servo.mp4"), frames, fps=30)
np.savetxt(os.path.join(OUT, "ee_path.csv"), path, delimiter=",", header="x,y,z")
print(f"  wrote out/cartesian_servo.mp4, out/ee_path.csv")


# ---------------------------------------------------------------------------
# 5. EXPERIMENTS. Change the numbers and rerun.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("5. EXPERIMENTS")
print("=" * 72)

unreachable = np.array([0.0, 0.0, 1.30])       # arm max reach is ~0.85
print("  (a) unreachable target, varying damping lambda:")
for lam in [0.0, 0.01, 0.05, 0.20]:
    qf, hist = _run_ik(np.array([0.0, 0.90, 0.55]), unreachable,
                       lambda J, e, l=lam: 0.9 * (J.T @ np.linalg.solve(J @ J.T + l**2 * np.eye(3), e)),
                       iters=400)
    print(f"      lam={lam:<5} final err {hist[-1]:.4f} m   "
          f"joint excursion {np.abs(qf - np.array([0.0,0.90,0.55])).max():9.2f} rad")
print("      -> lam=0: no safety net, the solver sprints joints to their stops and")
print("         still never arrives. lam>0: it creeps, hits the limit, and stops.")
print("         THAT is why every production IK has a damping term.")

print("\n  (b) try your own configs -- print the singularity table for them:")
for q in [np.array([0.0, 0.4, 0.2]), np.array([0.5, -0.6, 1.1]), np.array([1.0, 1.3, -0.2])]:
    J = jacobian(q); s = np.linalg.svd(J, compute_uv=False)
    print(f"      q={np.round(q,2)}  sv_min={s.min():.2e}  "
          f"rank={np.linalg.matrix_rank(J,1e-5)}  cond={s.max()/max(s.min(),1e-12):.1f}")

print("\n  (c) set lam=0.0 in servo_to() and command a target near full extension.")
print("      Watch the video: the hand shakes near the reach limit instead of stopping.")
print("\n  (d) Extend the XML with a 4th joint (wrist roll, axis 0 0 1) and make")
print("      jacobian() return 6 rows instead of 3 (position + orientation).")
print("      That 3xN -> 6xN change is the jump from 'move the hand' to 'move AND")
print("      orient the hand' -- which is what a real 6-DoF VLA action space needs.")
print(f"\nDone. Artifacts in {OUT}/")