import numpy as np
import pytest

from charuco_calibrator.core import (
    IntrinsicsOptions,
    calibrate_intrinsics,
    intrinsics_quality_warnings,
)
from synthetic import BOARD, DIST_TRUE, IMAGE_SIZE, K_TRUE, random_board_poses, write_views


@pytest.fixture(scope="module")
def diverse_folder(tmp_path_factory):
    return write_views(tmp_path_factory.mktemp("diverse"), random_board_poses(15, 40, seed=0))


@pytest.fixture(scope="module")
def frontal_folder(tmp_path_factory):
    return write_views(tmp_path_factory.mktemp("frontal"), random_board_poses(12, 2, seed=3))


@pytest.fixture(scope="module")
def mixed_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mixed")
    write_views(folder, random_board_poses(40, 2, seed=5), prefix="a_frontal")
    write_views(folder, random_board_poses(8, 40, seed=6), prefix="b_tilted")
    return folder


def relative_errors(payload):
    matrix = np.array(payload["camera_matrix"])
    values = [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]
    truth = [K_TRUE[0, 0], K_TRUE[1, 1], K_TRUE[0, 2], K_TRUE[1, 2]]
    return np.abs(np.array(values) - truth) / np.array(truth)


def test_diverse_views_recover_intrinsics(diverse_folder):
    payload = calibrate_intrinsics(diverse_folder, "cam", BOARD, 12, False)
    assert payload["accepted_frame_count"] == 15
    assert np.all(relative_errors(payload) < 0.01), relative_errors(payload)
    assert np.allclose(payload["distortion_coefficients"][:2], DIST_TRUE[:2], atol=0.02)
    assert payload["reprojection_error"] < 0.5
    assert payload["quality_warnings"] == []
    assert payload["auto_constrained"] is False
    assert payload["orientation_diversity"]["distinct_orientations"] >= 3
    used = [item for item in payload["accepted_frames"] if item["used_for_calibration"]]
    assert len(used) == 15 and all(item["reprojection_error"] < 1.0 for item in used)
    assert all("reprojection_error" in item for item in payload["view_poses"])
    assert payload["coverage"]["grid"] == [8, 6]
    assert sum(map(sum, payload["coverage"]["counts"])) == sum(item["charuco_corner_count"] for item in used)


def test_low_diversity_views_raise_warnings(frontal_folder):
    payload = calibrate_intrinsics(frontal_folder, "cam", BOARD, 12, False, options=IntrinsicsOptions(auto_constrain=False))
    warnings = payload["quality_warnings"]
    assert payload["orientation_diversity"]["distinct_orientations"] < 3
    assert any("distinct board orientation" in message for message in warnings)
    assert any("poorly constrained" in message for message in warnings)


def test_low_diversity_views_are_auto_constrained(frontal_folder):
    payload = calibrate_intrinsics(frontal_folder, "cam", BOARD, 12, False)
    matrix = payload["camera_matrix"]
    assert payload["auto_constrained"] is True
    assert payload["effective_calibration_options"]["fix_aspect_ratio"] is True
    assert matrix[0][0] == pytest.approx(matrix[1][1])
    assert matrix[0][2] == pytest.approx((IMAGE_SIZE[0] - 1) / 2)
    assert payload["quality_warnings"][0].startswith("Degenerate view set")
    assert "unconstrained_camera_matrix" in payload


def test_pose_diverse_frame_selection(mixed_folder):
    payload = calibrate_intrinsics(mixed_folder, "cam", BOARD, 12, False, options=IntrinsicsOptions(max_frames=12))
    used = [item["image_path"] for item in payload["accepted_frames"] if item["used_for_calibration"]]
    assert payload["frame_selection"] == {"method": "pose_diversity", "max_frames": 12, "candidates": 48, "selected": 12}
    assert sum("b_tilted" in path for path in used) >= 7
    assert np.all(relative_errors(payload) < 0.01), relative_errors(payload)
    assert payload["quality_warnings"] == []


def test_calibration_flags(diverse_folder):
    options = IntrinsicsOptions(fix_aspect_ratio=True, fix_principal_point=True, zero_tangential=True, fix_k3=True)
    payload = calibrate_intrinsics(diverse_folder, "cam", BOARD, 12, False, options=options)
    matrix = payload["camera_matrix"]
    dist = payload["distortion_coefficients"]
    assert matrix[0][0] == pytest.approx(matrix[1][1])
    assert matrix[0][2] == pytest.approx((IMAGE_SIZE[0] - 1) / 2)
    assert matrix[1][2] == pytest.approx((IMAGE_SIZE[1] - 1) / 2)
    assert dist[2] == 0.0 and dist[3] == 0.0 and dist[4] == 0.0
    assert payload["calibration_options"]["fix_principal_point"] is True

    free = calibrate_intrinsics(diverse_folder, "cam", BOARD, 12, False, options=IntrinsicsOptions(fix_k3=False))
    assert free["distortion_coefficients"][4] != 0.0


def test_quality_warnings_for_degenerate_matrix():
    orientation = {
        "distinct_orientations": 1,
        "orientation_spread_deg": 0.2,
        "tilt_about_x_range_deg": 0.1,
        "tilt_about_y_range_deg": 0.2,
        "max_tilt_deg": 75.0,
    }
    coverage = {"fraction": 0.08}
    matrix = [[905.9, 0.0, 926.1], [0.0, 188.1, 447.8], [0.0, 0.0, 1.0]]
    warnings = intrinsics_quality_warnings(
        matrix, (1920, 1080), orientation, coverage, {"fx": 30.0, "fy": 40.0}, IntrinsicsOptions()
    )
    text = "\n".join(warnings)
    assert "fx/fy ratio" in text
    assert "fy=188.1 px is implausible" in text
    assert "distinct board orientation" in text
    assert "cover only 8%" in text
    assert "standard deviation" in text

    matrix = [[1000.0, 0.0, 959.5], [0.0, 1000.0, 700.0], [0.0, 0.0, 1.0]]
    warnings = intrinsics_quality_warnings(
        matrix, (1920, 1080), {**orientation, "distinct_orientations": 5, "tilt_about_x_range_deg": 50,
                               "tilt_about_y_range_deg": 50}, {"fraction": 0.9}, None, IntrinsicsOptions()
    )
    assert len(warnings) == 1 and "Principal point" in warnings[0]


def test_too_few_frames(tmp_path):
    write_views(tmp_path, random_board_poses(2, 30, seed=9))
    with pytest.raises(RuntimeError, match="at least 3"):
        calibrate_intrinsics(tmp_path, "cam", BOARD, 12, False)
