import numpy as np
import pytest

from charuco_calibrator.core import estimate_multi_camera_extrinsics, rotation_angle_deg, scipy_available
from synthetic import BOARD, rig_board_poses, rig_reference_to_camera, write_rig


@pytest.fixture(scope="module")
def rig_folders(tmp_path_factory):
    return write_rig(tmp_path_factory.mktemp("rig"), rig_board_poses(12, seed=2))


def pose_errors(payload):
    truth = rig_reference_to_camera()
    errors = []
    for index, item in enumerate(payload["cameras"]):
        estimate = np.array(item["T_reference_to_camera"])
        centre_estimate = np.linalg.inv(estimate)[:3, 3]
        centre_truth = np.linalg.inv(truth[index])[:3, 3]
        errors.append(
            (rotation_angle_deg(np, estimate[:3, :3], truth[index][:3, :3]), np.linalg.norm(centre_estimate - centre_truth))
        )
    return errors


@pytest.mark.parametrize("bundle_adjust", [True, False])
def test_multi_frame_relative_pose(rig_folders, bundle_adjust):
    if bundle_adjust and not scipy_available():
        pytest.skip("scipy not installed")
    payload = estimate_multi_camera_extrinsics(rig_folders, BOARD, 12, "cam1", bundle_adjust=bundle_adjust)
    assert payload["frame_set_count"] == 12
    assert payload["refinement"] == ("bundle_adjustment" if bundle_adjust else "robust_average")
    limit_mm = 1.0 if bundle_adjust else 3.0
    for rotation_error, centre_error in pose_errors(payload):
        assert rotation_error < 0.1
        assert centre_error * 1000 < limit_mm
    for item in payload["cameras"]:
        assert item["frame_count"] == 12
        assert item["reprojection_error_px"] < 0.5
        assert item["reprojection_error_px_initial"] is not None


def test_single_synchronized_image_per_camera(rig_folders):
    entries = [dict(entry, image_path=f"{entry['image_path']}/f000.png") for entry in rig_folders]
    payload = estimate_multi_camera_extrinsics(entries, BOARD, 12, "cam2")
    assert payload["refinement"] == "single_frame"
    assert payload["reference_camera"] == "cam2"
    assert np.allclose(payload["cameras"][1]["T_camera_to_reference"], np.eye(4), atol=1e-9)
    for rotation_error, _centre_error in pose_errors_relative_to(payload, 1):
        assert rotation_error < 0.2


def pose_errors_relative_to(payload, reference_index):
    truth = rig_reference_to_camera()
    reference_inverse = np.linalg.inv(truth[reference_index])
    out = []
    for index, item in enumerate(payload["cameras"]):
        expected = truth[index] @ reference_inverse
        estimate = np.array(item["T_reference_to_camera"])
        out.append((rotation_angle_deg(np, estimate[:3, :3], expected[:3, :3]), np.linalg.norm(estimate[:3, 3] - expected[:3, 3])))
    return out


def test_frame_set_limit_and_every(rig_folders):
    payload = estimate_multi_camera_extrinsics(rig_folders, BOARD, 12, None, max_frame_sets=3)
    assert payload["frame_set_count"] == 3
    from charuco_calibrator.core import FrameSampling

    payload = estimate_multi_camera_extrinsics(rig_folders, BOARD, 12, None, sampling=FrameSampling(every=2))
    assert payload["frame_sets"] == ["f000", "f002", "f004", "f006", "f008", "f010"]


def test_multi_from_videos(tmp_path):
    entries = write_rig(tmp_path, rig_board_poses(6, seed=4), as_video=True)
    if entries is None:
        pytest.skip("No MJPG video writer in this OpenCV build")
    payload = estimate_multi_camera_extrinsics(entries, BOARD, 12, "cam1")
    assert payload["input_type"] == "video"
    assert payload["frame_sets"] == ["0", "1", "2", "3", "4", "5"]
    for rotation_error, centre_error in pose_errors(payload):
        assert rotation_error < 0.2
        assert centre_error < 0.003


def test_missing_reference_camera(rig_folders):
    with pytest.raises(RuntimeError, match="Reference camera not found"):
        estimate_multi_camera_extrinsics(rig_folders, BOARD, 12, "nope")


def test_mixed_input_types_rejected(rig_folders):
    entries = [rig_folders[0], dict(rig_folders[1], image_path=f"{rig_folders[1]['image_path']}/f000.png")]
    with pytest.raises(RuntimeError, match="same input type"):
        estimate_multi_camera_extrinsics(entries, BOARD, 12, None)


def test_partially_overlapping_folders(rig_folders, tmp_path):
    import shutil
    from pathlib import Path

    entries = []
    for index, entry in enumerate(rig_folders):
        folder = tmp_path / entry["camera_name"]
        shutil.copytree(entry["image_path"], folder)
        for frame in range(index * 4, index * 4 + 4):
            (folder / f"f{frame:03d}.png").unlink(missing_ok=True)
        entries.append(dict(entry, image_path=str(folder)))
    assert not set.intersection(*[{path.stem for path in Path(e["image_path"]).iterdir()} for e in entries])
    payload = estimate_multi_camera_extrinsics(entries, BOARD, 12, "cam1", max_frame_sets=0, bundle_adjust=scipy_available())
    assert payload["frame_set_count"] == 12
    for item in payload["cameras"]:
        assert item["frame_count"] == 8
    for rotation_error, centre_error in pose_errors(payload):
        assert rotation_error < 0.1
        assert centre_error * 1000 < 3.0
