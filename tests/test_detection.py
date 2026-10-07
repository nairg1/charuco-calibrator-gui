import os
import subprocess
import sys

import numpy as np
import pytest

from charuco_calibrator.core import (
    BoardSettings,
    detect_charuco_in_array,
    estimate_pose_from_detection,
    rotation_angle_deg,
    run_self_check,
)
from synthetic import BOARD, DIST_TRUE, K_TRUE, ROOT, board_pose, render

ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def test_self_check_in_process():
    result = run_self_check(verbose=False)
    assert result["charuco_corners"] == result["expected_corners"]
    assert result["rotation_error_deg"] < 0.1


def test_self_check_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "charuco_calibrator", "--self-check"], env=ENV, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "Self check OK" in completed.stdout


def test_launcher_self_check():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "charuco_calibrator_gui.py"), "--self-check"], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("legacy", [True, False])
def test_detect_and_pose_on_synthetic_board(legacy):
    import cv2

    board = BoardSettings(BOARD.squares_x, BOARD.squares_y, BOARD.square_length, BOARD.marker_length, "DICT_4X4_100", legacy)
    transform = board_pose(20, -15, 5, [0.05, 0.02, 1.5])
    rvec = cv2.Rodrigues(transform[:3, :3])[0]
    from charuco_calibrator.core import render_board_view

    image = render_board_view(board, K_TRUE, rvec, transform[:3, 3], (1280, 720), DIST_TRUE)
    detection = detect_charuco_in_array(image, board, 12, "synthetic")
    assert detection.accepted
    assert detection.charuco_corner_count == (board.squares_x - 1) * (board.squares_y - 1)
    pose = estimate_pose_from_detection(
        detection, {"camera_matrix": K_TRUE.tolist(), "distortion_coefficients": DIST_TRUE.tolist()}, board
    )
    estimate = np.array(pose["T_board_to_camera"])
    assert rotation_angle_deg(np, estimate[:3, :3], transform[:3, :3]) < 0.1
    assert np.linalg.norm(estimate[:3, 3] - transform[:3, 3]) < 0.002
    assert pose["reprojection_error"] < 0.5


def test_blank_image_is_rejected():
    detection = detect_charuco_in_array(np.full((480, 640), 128, np.uint8), BOARD, 12, "blank")
    assert not detection.accepted
    assert detection.reason == "No ArUco markers detected"


def test_unreadable_image_is_rejected(tmp_path):
    from charuco_calibrator.core import detect_charuco_in_image

    detection = detect_charuco_in_image(tmp_path / "missing.png", BOARD, 12)
    assert not detection.accepted
    assert detection.reason == "Could not read image"


def test_core_and_cli_do_not_import_tkinter(tmp_path):
    code = (
        "import sys; sys.modules['tkinter'] = None\n"
        "from charuco_calibrator.cli import main\n"
        "import charuco_calibrator, charuco_calibrator.core\n"
        "raise SystemExit(main(['--self-check']))\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], env=ENV, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_render_matches_projection():
    import cv2

    transform = board_pose(10, 25, 0, [0.0, 0.0, 1.6])
    image = render(transform)
    detection = detect_charuco_in_array(image, BOARD, 12, "synthetic")
    from charuco_calibrator.core import board_correspondences

    obj, img = board_correspondences(cv2, np, BOARD, detection.charuco_corners, detection.charuco_ids)
    projected, _ = cv2.projectPoints(obj, cv2.Rodrigues(transform[:3, :3])[0], transform[:3, 3], K_TRUE, DIST_TRUE)
    error = np.linalg.norm(projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    assert np.median(error) < 0.75
