import json

import numpy as np
import pytest

from charuco_calibrator.core import (
    calibrate_intrinsics,
    estimate_extrinsics_for_path,
    estimate_multi_camera_extrinsics,
    read_json,
    write_json,
)
from synthetic import BOARD, DIST_TRUE, K_TRUE, random_board_poses, rig_board_poses, write_rig, write_views

BOARD_KEYS = {"squares_x", "squares_y", "square_length", "marker_length", "dictionary_name", "legacy_pattern"}
POSE_KEYS = {"rvec", "tvec", "T_board_to_camera", "T_camera_to_board", "image_path", "marker_count", "charuco_corner_count"}
LEGACY_INTRINSICS_KEYS = {
    "camera_name", "mode", "board", "image_size", "reprojection_error", "camera_matrix", "distortion_coefficients",
    "accepted_frame_count", "rejected_frame_count", "accepted_frames", "rejected_frames", "view_poses",
}
LEGACY_MULTI_CAMERA_KEYS = POSE_KEYS | {
    "camera_name", "intrinsics_path", "T_camera_to_reference", "T_reference_to_camera", "reference_camera",
}

OLD_INTRINSICS_JSON = {
    "camera_name": "cam1",
    "mode": "intrinsics",
    "board": {"squares_x": 8, "squares_y": 6, "square_length": 0.0948, "marker_length": 0.0711,
              "dictionary_name": "DICT_4X4_50", "legacy_pattern": True},
    "image_size": {"width": 1280, "height": 720},
    "reprojection_error": 0.3,
    "camera_matrix": K_TRUE.tolist(),
    "distortion_coefficients": DIST_TRUE.tolist(),
    "accepted_frame_count": 3,
    "rejected_frame_count": 0,
    "accepted_frames": [],
    "rejected_frames": [],
    "view_poses": [],
}


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    folder = write_views(tmp_path_factory.mktemp("views"), random_board_poses(8, 35, seed=11))
    (folder / "blank.png").write_bytes((folder / "f000.png").read_bytes()[:10])
    return folder


def test_intrinsics_json_round_trip_and_legacy_keys(views, tmp_path):
    payload = calibrate_intrinsics(views, "cam1", BOARD, 12, False)
    path = tmp_path / "cam1_intrinsics.json"
    write_json(path, payload)
    loaded = read_json(path)
    assert loaded == json.loads(json.dumps(payload))
    assert LEGACY_INTRINSICS_KEYS <= set(loaded)
    assert set(loaded["board"]) == BOARD_KEYS
    assert set(loaded["image_size"]) == {"width", "height"}
    assert np.array(loaded["camera_matrix"]).shape == (3, 3)
    assert len(loaded["distortion_coefficients"]) == 5
    assert loaded["rejected_frame_count"] == 1
    assert {"image_path", "reason", "marker_count", "charuco_corner_count"} <= set(loaded["rejected_frames"][0])
    assert {"image_path", "marker_count", "charuco_corner_count"} <= set(loaded["accepted_frames"][0])
    assert {"image_path", "rvec", "tvec"} <= set(loaded["view_poses"][0])
    assert isinstance(loaded["quality_warnings"], list)


def test_old_intrinsics_json_drives_extrinsics(views, tmp_path):
    single = estimate_extrinsics_for_path(views / "f000.png", "cam1", OLD_INTRINSICS_JSON, BOARD, 8, False)
    assert single["mode"] == "extrinsics_single"
    assert {"camera_name", "mode", "intrinsics_source", "pose"} <= set(single)
    assert POSE_KEYS <= set(single["pose"])

    minimal = {"camera_matrix": K_TRUE.tolist(), "distortion_coefficients": DIST_TRUE.tolist()}
    batch = estimate_extrinsics_for_path(views, "cam1", minimal, BOARD, 8, False)
    assert batch["mode"] == "extrinsics_batch"
    assert {"camera_name", "mode", "intrinsics_source", "estimated_pose_count", "failed_pose_count", "poses",
            "failed_images"} <= set(batch)
    assert batch["estimated_pose_count"] == 8 and batch["failed_pose_count"] == 1
    assert POSE_KEYS <= set(batch["poses"][0])
    write_json(tmp_path / "batch.json", batch)
    assert read_json(tmp_path / "batch.json")["poses"][0]["rvec"] == batch["poses"][0]["rvec"]


def test_multi_json_legacy_keys(tmp_path):
    entries = write_rig(tmp_path, rig_board_poses(3, seed=7))
    payload = estimate_multi_camera_extrinsics(entries, BOARD, 8, "cam1")
    write_json(tmp_path / "multi.json", payload)
    loaded = read_json(tmp_path / "multi.json")
    assert {"mode", "reference_camera", "board", "cameras"} <= set(loaded)
    assert loaded["mode"] == "multi_camera_extrinsics"
    for item in loaded["cameras"]:
        assert LEGACY_MULTI_CAMERA_KEYS <= set(item)
        assert item["reference_camera"] == "cam1"
        to_ref = np.array(item["T_camera_to_reference"])
        assert np.allclose(to_ref @ np.array(item["T_reference_to_camera"]), np.eye(4), atol=1e-9)
        assert np.allclose(np.linalg.inv(np.array(item["T_board_to_camera"])), item["T_camera_to_board"], atol=1e-9)


def test_visualizer_accepts_old_and_new_json(tmp_path):
    pytest.importorskip("tkinter")
    from charuco_calibrator.gui import CharucoCalibratorApp

    app = object.__new__(CharucoCalibratorApp)
    entries = write_rig(tmp_path, rig_board_poses(2, seed=8))
    new_payload = estimate_multi_camera_extrinsics(entries, BOARD, 8, "cam1")
    old_payload = {
        key: new_payload[key] for key in ("mode", "reference_camera", "board")
    } | {"cameras": [{key: item[key] for key in LEGACY_MULTI_CAMERA_KEYS} for item in new_payload["cameras"]]}
    for payload in (new_payload, old_payload):
        mode, cameras, shapes, _axes, _summary = app._prepare_visual_scene(payload)
        assert mode == "multi_camera_extrinsics" and len(cameras) == 3 and len(shapes) == 1
