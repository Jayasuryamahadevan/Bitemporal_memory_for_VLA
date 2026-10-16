"""Shared MuJoCo helpers: a correct camera lookat, and a state snapshotter.

The lookat function is here because getting a camera pointed the right way in
MuJoCo is a classic time sink. The convention (verified empirically, not guessed):

    camera views along  -z  of its own frame
    cam_pos            = eye position in world coords
    xyaxes             = the world-frame x axis, then y axis, of the camera
    z_axis             = x_axis x y_axis        (right-handed)

so to aim at a target:

    z = -normalize(target - eye)
    x = normalize(cross(world_up, z))
    y = cross(z, x)
"""
import numpy as np
import mujoco


def lookat(eye, target, world_up=(0, 0, 1)):
    """Return (pos, xyaxes, quat_wxyz) for a camera at `eye` pointing at `target`."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)
    z = -z                                  # camera looks down its own -z

    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-8:                            # looking straight up/down: pick any perp
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)

    xyaxes = np.concatenate([x, y])         # this is the MJCF "xyaxes" attribute
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(np.column_stack([x, y, z])).ravel())
    return eye, xyaxes, quat


def camera_xml(eye, target, fovy=50, name="cam"):
    """An <camera> tag ready to paste into MJCF."""
    pos, xyaxes, _ = lookat(eye, target)
    return (f'<camera name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'xyaxes="{xyaxes[0]:.4f} {xyaxes[1]:.4f} {xyaxes[2]:.4f} '
            f'{xyaxes[3]:.4f} {xyaxes[4]:.4f} {xyaxes[5]:.4f}" fovy="{fovy}"/>')


def set_camera(model, cam_id, eye, target):
    """Repoint an existing camera at runtime (no XML reload, no new GL context)."""
    pos, xyaxes, quat = lookat(eye, target)
    model.cam_pos[cam_id] = pos
    model.cam_quat[cam_id] = quat