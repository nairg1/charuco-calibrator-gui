"""Synthetic ChArUco views rendered with known intrinsics and poses."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from charuco_calibrator.core import BoardSettings, matrix_to_rt, render_board_view, write_json

ROOT = Path(__file__).resolve().parents[1]
BOARD = BoardSettings(8, 6, 0.0948, 0.0711, "DICT_4X4_50", legacy_pattern=True)
IMAGE_SIZE = (1280, 720)
K_TRUE = np.array([[1000.0, 0.0, 652.0], [0.0, 1006.0, 353.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.12, 0.06, 0.0, 0.0, 0.0])
BOARD_CENTRE = np.array([BOARD.squares_x * BOARD.square_length / 2, BOARD.squares_y * BOARD.square_length / 2, 0.0])


def board_pose(tilt_x_deg, tilt_y_deg, roll_deg, centre):
    rotation = cv2.Rodrigues(np.radians([tilt_x_deg, tilt_y_deg, roll_deg]))[0]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(centre, dtype=float) - rotation @ BOARD_CENTRE
    return transform


def random_board_poses(count, max_tilt_deg, seed):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(count):
        depth = rng.uniform(1.3, 2.0)
        centre = [rng.uniform(-0.25, 0.25) * depth, rng.uniform(-0.15, 0.15) * depth, depth]
        poses.append(
            board_pose(
                rng.uniform(-max_tilt_deg, max_tilt_deg),
                rng.uniform(-max_tilt_deg, max_tilt_deg),
                rng.uniform(-15, 15),
                centre,
            )
        )
    return poses


def render(transform, camera_matrix=K_TRUE, dist=DIST_TRUE, size=IMAGE_SIZE):
    rvec, tvec = matrix_to_rt(cv2, np, transform)
    return render_board_view(BOARD, camera_matrix, rvec, tvec, size, dist)


def write_views(folder: Path, transforms, camera_matrix=K_TRUE, dist=DIST_TRUE, prefix="f"):
    folder.mkdir(parents=True, exist_ok=True)
    for index, transform in enumerate(transforms):
        cv2.imwrite(str(folder / f"{prefix}{index:03d}.png"), render(transform, camera_matrix, dist))
    return folder


def write_video(path: Path, transforms, camera_matrix=K_TRUE, dist=DIST_TRUE, repeat=1):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, IMAGE_SIZE)
    if not writer.isOpened():
        return None
    for transform in transforms:
        frame = cv2.cvtColor(render(transform, camera_matrix, dist), cv2.COLOR_GRAY2BGR)
        for _ in range(repeat):
            writer.write(frame)
    writer.release()
    return path


def look_at(centre, target):
    centre = np.asarray(centre, dtype=float)
    forward = np.asarray(target, dtype=float) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([right, down, forward])
    transform[:3, 3] = -transform[:3, :3] @ centre
    return transform


RIG_TARGET = [0.3, 0.0, 2.0]
RIG_WORLD_TO_CAMERA = [
    look_at([0.0, 0.0, 0.0], RIG_TARGET),
    look_at([1.0, 0.05, 0.2], RIG_TARGET),
    look_at([0.2, -0.8, 0.3], RIG_TARGET),
]
RIG_K = [
    np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]),
    np.array([[900.0, 0.0, 630.0], [0.0, 905.0, 365.0], [0.0, 0.0, 1.0]]),
    np.array([[1100.0, 0.0, 645.0], [0.0, 1100.0, 350.0], [0.0, 0.0, 1.0]]),
]
RIG_DIST = [np.zeros(5), np.array([-0.1, 0.05, 0.0, 0.0, 0.0]), np.zeros(5)]


def rig_reference_to_camera():
    reference_inverse = np.linalg.inv(RIG_WORLD_TO_CAMERA[0])
    return [transform @ reference_inverse for transform in RIG_WORLD_TO_CAMERA]


def rig_board_poses(count, seed=1):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(count):
        centre = np.array(RIG_TARGET) + [rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15), rng.uniform(-0.3, 0.3)]
        poses.append(board_pose(rng.uniform(-25, 25), rng.uniform(-10, 30), rng.uniform(-15, 15), centre))
    return poses


def write_rig(root: Path, board_world_poses, as_video=False):
    """Render every board pose in every rig camera; return camera entries for multi-camera extrinsics."""
    entries = []
    for index, world_to_camera in enumerate(RIG_WORLD_TO_CAMERA):
        name = f"cam{index + 1}"
        intrinsics_path = root / f"{name}_intrinsics.json"
        write_json(
            intrinsics_path,
            {
                "camera_name": name,
                "camera_matrix": RIG_K[index].tolist(),
                "distortion_coefficients": RIG_DIST[index].tolist(),
            },
        )
        views = [world_to_camera @ pose for pose in board_world_poses]
        if as_video:
            source = write_video(root / f"{name}.avi", views, RIG_K[index], RIG_DIST[index])
            if source is None:
                return None
        else:
            source = write_views(root / name, views, RIG_K[index], RIG_DIST[index])
        entries.append({"camera_name": name, "intrinsics_path": str(intrinsics_path), "image_path": str(source)})
    return entries
