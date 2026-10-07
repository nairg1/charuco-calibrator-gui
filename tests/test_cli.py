import json
import os
import subprocess
import sys

import numpy as np
import pytest

from charuco_calibrator.cli import main
from synthetic import BOARD, K_TRUE, ROOT, random_board_poses, rig_board_poses, write_rig, write_video, write_views

BOARD_ARGS = [
    "--squares-x", str(BOARD.squares_x), "--squares-y", str(BOARD.squares_y),
    "--square-length", str(BOARD.square_length), "--marker-length", str(BOARD.marker_length),
    "--dictionary", BOARD.dictionary_name, "--legacy-pattern",
]


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    return write_views(tmp_path_factory.mktemp("cli_views"), random_board_poses(12, 40, seed=21))


def test_cli_intrinsics_from_images(views, tmp_path, capsys):
    output = tmp_path / "cam1.json"
    code = main(["intrinsics", "--images", str(views), "--output", str(output), "--quiet", *BOARD_ARGS])
    assert code == 0
    payload = json.loads(output.read_text())
    assert np.allclose(payload["camera_matrix"][0][0], K_TRUE[0, 0], rtol=0.01)
    out = capsys.readouterr().out
    assert "Quality warnings: none" in out and "Corner coverage" in out


def test_cli_intrinsics_from_video_headless(tmp_path):
    video = write_video(tmp_path / "board.avi", random_board_poses(12, 40, seed=22), repeat=2)
    if video is None:
        pytest.skip("No MJPG video writer in this OpenCV build")
    output = tmp_path / "video_intrinsics.json"
    code = (
        "import sys; sys.modules['tkinter'] = None\n"
        "from charuco_calibrator.cli import main\n"
        f"raise SystemExit(main({['intrinsics', '--video', str(video), '--every', '2', '--output', str(output), '--quiet', *BOARD_ARGS]!r}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["accepted_frame_count"] == 12
    assert payload["accepted_frames"][1]["image_path"].endswith("#frame=2")
    assert np.allclose(payload["camera_matrix"][0][0], K_TRUE[0, 0], rtol=0.01)


def test_cli_fail_on_warnings(tmp_path):
    folder = write_views(tmp_path / "frontal", random_board_poses(6, 2, seed=23))
    code = main(["intrinsics", "--images", str(folder), "--output", str(tmp_path / "o.json"), "--quiet",
                 "--fail-on-warnings", *BOARD_ARGS])
    assert code == 2


def test_cli_extrinsics(views, tmp_path):
    intrinsics = tmp_path / "intr.json"
    assert main(["intrinsics", "--images", str(views), "--output", str(intrinsics), "--quiet", *BOARD_ARGS]) == 0
    single = tmp_path / "single.json"
    assert main(["extrinsics", "--intrinsics", str(intrinsics), "--image", str(views / "f000.png"),
                 "--output", str(single), *BOARD_ARGS]) == 0
    assert json.loads(single.read_text())["mode"] == "extrinsics_single"
    batch = tmp_path / "batch.json"
    assert main(["extrinsics", "--intrinsics", str(intrinsics), "--images", str(views), "--every", "3",
                 "--output", str(batch), *BOARD_ARGS]) == 0
    assert json.loads(batch.read_text())["estimated_pose_count"] == 4


def test_cli_multi(tmp_path, capsys):
    entries = write_rig(tmp_path, rig_board_poses(5, seed=24))
    args = ["multi", "--reference", "cam1", "--output", str(tmp_path / "multi.json"), "--quiet", *BOARD_ARGS]
    for entry in entries:
        args += ["--camera", entry["camera_name"], "--intrinsics", entry["intrinsics_path"], "--source", entry["image_path"]]
    assert main(args) == 0
    payload = json.loads((tmp_path / "multi.json").read_text())
    assert payload["frame_set_count"] == 5
    assert all(item["reprojection_error_px"] < 0.5 for item in payload["cameras"])
    assert "rms px" in capsys.readouterr().out


def test_cli_multi_argument_mismatch(tmp_path, capsys):
    assert main(["multi", "--camera", "a", "--camera", "b", "--intrinsics", "x.json", "--source", "y"]) == 1
    assert "one --intrinsics" in capsys.readouterr().err


def test_cli_missing_input(tmp_path, capsys):
    assert main(["intrinsics", "--images", str(tmp_path / "nope"), "--quiet"]) == 1
    assert "does not exist" in capsys.readouterr().err
