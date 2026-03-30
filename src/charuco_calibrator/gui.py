#!/usr/bin/env python3
"""Standalone ChArUco calibration GUI for intrinsics and extrinsics."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


APP_TITLE = "ChArUco Intrinsics / Extrinsics Calibrator"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
APP_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_ROOT = APP_DIR / "charuco_calibration_output"
DEFAULT_INTRINSICS_ROOT = DEFAULT_CALIBRATION_ROOT / "intrinsics"
DEFAULT_EXTRINSICS_ROOT = DEFAULT_CALIBRATION_ROOT / "extrinsics"
DEFAULT_MULTI_CAMERA_COUNT = 6
ARUCO_DICTIONARIES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_ARUCO_ORIGINAL",
]
DEPENDENCY_HINT = (
    "This calibrator needs numpy and opencv-contrib-python.\n\n"
    "Install them with:\n"
    "python3 -m pip install -r requirements.txt"
)


@dataclass
class BoardSettings:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary_name: str
    legacy_pattern: bool = False


@dataclass
class DetectionSummary:
    image_path: str
    accepted: bool
    marker_count: int
    charuco_corner_count: int
    reason: str
    image_size: tuple[int, int] | None = None
    charuco_corners: Any | None = None
    charuco_ids: Any | None = None


def load_backend():
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"{DEPENDENCY_HINT}\n\nImport error: {exc}") from exc

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build does not include the aruco module.\n\n"
            "Install opencv-contrib-python, not plain opencv-python."
        )

    return cv2, np


def get_dictionary(cv2, dictionary_name: str):
    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary_name, None)
    if dict_id is None:
        raise ValueError(f"Unsupported dictionary: {dictionary_name}")

    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def create_board(cv2, settings: BoardSettings):
    aruco = cv2.aruco
    dictionary = get_dictionary(cv2, settings.dictionary_name)
    if hasattr(aruco, "CharucoBoard") and callable(aruco.CharucoBoard):
        board = aruco.CharucoBoard(
            (settings.squares_x, settings.squares_y),
            settings.square_length,
            settings.marker_length,
            dictionary,
        )
        if settings.legacy_pattern and hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)
        return board
    if hasattr(aruco, "CharucoBoard_create"):
        board = aruco.CharucoBoard_create(
            settings.squares_x,
            settings.squares_y,
            settings.square_length,
            settings.marker_length,
            dictionary,
        )
        if settings.legacy_pattern and hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)
        return board
    raise RuntimeError("This OpenCV ArUco build does not support CharucoBoard creation.")


def create_detector_parameters(cv2):
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters"):
        return aruco.DetectorParameters()
    return aruco.DetectorParameters_create()


def iter_image_paths(folder: Path, recursive: bool) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sanitize_name(name: str) -> str:
    clean = "".join(char if char.isalnum() or char in ("_", "-", ".") else "_" for char in name.strip())
    return clean.strip("_") or "calibration"


def default_intrinsics_output_path(camera_name: str) -> Path:
    return DEFAULT_CALIBRATION_ROOT / f"{sanitize_name(camera_name)}_intrinsics.json"


def default_extrinsics_output_path(camera_name: str) -> Path:
    return DEFAULT_CALIBRATION_ROOT / f"{sanitize_name(camera_name)}_extrinsics.json"


def default_batch_extrinsics_output_path(camera_name: str, input_path: Path) -> Path:
    return input_path / f"{sanitize_name(camera_name)}_extrinsics_batch.json"


def looks_like_camera_folder(folder: Path) -> bool:
    return folder.is_dir() and any(path.suffix.lower() in IMAGE_EXTENSIONS for path in folder.iterdir() if path.is_file())


def default_extrinsics_image_for_camera(camera_name: str) -> Path | None:
    direct = DEFAULT_EXTRINSICS_ROOT / f"{camera_name}.png"
    if direct.exists():
        return direct

    if not DEFAULT_EXTRINSICS_ROOT.exists():
        return None

    candidates = sorted(
        path
        for path in DEFAULT_EXTRINSICS_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    normalized = camera_name.lower()
    digit_suffix = "".join(char for char in normalized if char.isdigit())
    preferred_tokens = [normalized]
    if digit_suffix:
        preferred_tokens.extend(
            [
                f"cam{digit_suffix}",
                f"cam_{digit_suffix}",
                f"camera{digit_suffix}",
                f"camera_{digit_suffix}",
            ]
        )

    for path in candidates:
        stem = path.stem.lower()
        if any(token in stem for token in preferred_tokens):
            return path

    return None


def default_multi_camera_payload(index: int) -> dict[str, str]:
    camera_name = f"cam{index + 1}"
    return {
        "enabled": "",
        "camera_name": camera_name,
        "intrinsics_path": "",
        "image_path": "",
    }


def transform_points(np, transform: list[list[float]] | Any, points: list[list[float]]) -> Any:
    matrix = np.array(transform, dtype=float)
    homogeneous = np.hstack([np.array(points, dtype=float), np.ones((len(points), 1), dtype=float)])
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def detect_charuco_in_image(
    image_path: Path,
    settings: BoardSettings,
    min_corners: int,
) -> DetectionSummary:
    cv2, _np = load_backend()
    image = cv2.imread(str(image_path))
    if image is None:
        return DetectionSummary(
            image_path=str(image_path),
            accepted=False,
            marker_count=0,
            charuco_corner_count=0,
            reason="Could not read image",
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board = create_board(cv2, settings)
    dictionary = get_dictionary(cv2, settings.dictionary_name)
    parameters = create_detector_parameters(cv2)

    corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    marker_count = 0 if ids is None else int(len(ids))
    if ids is None or marker_count == 0:
        return DetectionSummary(
            image_path=str(image_path),
            accepted=False,
            marker_count=0,
            charuco_corner_count=0,
            reason="No ArUco markers detected",
            image_size=(gray.shape[1], gray.shape[0]),
        )

    interpolated = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if len(interpolated) == 3:
        _retval, charuco_corners, charuco_ids = interpolated
    else:
        _retval, charuco_corners, charuco_ids, _ = interpolated

    charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
    if charuco_ids is None or charuco_corners is None or charuco_count < min_corners:
        return DetectionSummary(
            image_path=str(image_path),
            accepted=False,
            marker_count=marker_count,
            charuco_corner_count=charuco_count,
            reason=f"Only {charuco_count} ChArUco corners found",
            image_size=(gray.shape[1], gray.shape[0]),
        )

    return DetectionSummary(
        image_path=str(image_path),
        accepted=True,
        marker_count=marker_count,
        charuco_corner_count=charuco_count,
        reason="Accepted",
        image_size=(gray.shape[1], gray.shape[0]),
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
    )


def pose_dict_from_rvec_tvec(cv2, np, rvec, tvec) -> dict[str, Any]:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = tvec.reshape(3)
    inverse = np.linalg.inv(transform)
    return {
        "rvec": [float(value) for value in rvec.reshape(-1)],
        "tvec": [float(value) for value in tvec.reshape(-1)],
        "T_board_to_camera": [[float(value) for value in row] for row in transform.tolist()],
        "T_camera_to_board": [[float(value) for value in row] for row in inverse.tolist()],
    }


def calibrate_intrinsics(
    image_folder: Path,
    camera_name: str,
    board_settings: BoardSettings,
    min_corners: int,
    recursive: bool,
) -> dict[str, Any]:
    cv2, _np = load_backend()
    image_paths = iter_image_paths(image_folder, recursive)
    if not image_paths:
        raise RuntimeError(f"No supported image files found in {image_folder}")

    accepted_detections: list[DetectionSummary] = []
    rejected_detections: list[DetectionSummary] = []
    image_size = None

    for image_path in image_paths:
        detection = detect_charuco_in_image(image_path, board_settings, min_corners)
        if detection.image_size and image_size is None:
            image_size = detection.image_size
        if detection.accepted:
            accepted_detections.append(detection)
        else:
            rejected_detections.append(detection)

    if len(accepted_detections) < 3:
        raise RuntimeError(
            "Need at least 3 accepted ChArUco frames for intrinsics calibration.\n"
            f"Accepted: {len(accepted_detections)} / {len(image_paths)}"
        )

    board = create_board(cv2, board_settings)
    charuco_corners = [item.charuco_corners for item in accepted_detections]
    charuco_ids = [item.charuco_ids for item in accepted_detections]

    result = cv2.aruco.calibrateCameraCharuco(
        charuco_corners,
        charuco_ids,
        board,
        image_size,
        None,
        None,
    )
    reprojection_error, camera_matrix, distortion_coeffs, rvecs, tvecs = result[:5]

    return {
        "camera_name": camera_name,
        "mode": "intrinsics",
        "board": {
            "squares_x": board_settings.squares_x,
            "squares_y": board_settings.squares_y,
            "square_length": board_settings.square_length,
            "marker_length": board_settings.marker_length,
            "dictionary_name": board_settings.dictionary_name,
            "legacy_pattern": board_settings.legacy_pattern,
        },
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "reprojection_error": float(reprojection_error),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion_coeffs.reshape(-1).tolist(),
        "accepted_frame_count": len(accepted_detections),
        "rejected_frame_count": len(rejected_detections),
        "accepted_frames": [
            {
                "image_path": item.image_path,
                "marker_count": item.marker_count,
                "charuco_corner_count": item.charuco_corner_count,
            }
            for item in accepted_detections
        ],
        "rejected_frames": [
            {
                "image_path": item.image_path,
                "reason": item.reason,
                "marker_count": item.marker_count,
                "charuco_corner_count": item.charuco_corner_count,
            }
            for item in rejected_detections
        ],
        "view_poses": [
            {
                "image_path": accepted_detections[index].image_path,
                "rvec": [float(value) for value in rvec.reshape(-1)],
                "tvec": [float(value) for value in tvec.reshape(-1)],
            }
            for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs))
        ],
    }


def estimate_pose_from_image(
    image_path: Path,
    intrinsics_payload: dict[str, Any],
    board_settings: BoardSettings,
    min_corners: int,
) -> dict[str, Any]:
    cv2, np = load_backend()
    detection = detect_charuco_in_image(image_path, board_settings, min_corners)
    if not detection.accepted:
        raise RuntimeError(f"{image_path.name}: {detection.reason}")

    camera_matrix = np.array(intrinsics_payload["camera_matrix"], dtype=float)
    distortion_coeffs = np.array(intrinsics_payload["distortion_coefficients"], dtype=float).reshape(-1, 1)
    board = create_board(cv2, board_settings)

    try:
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            detection.charuco_corners,
            detection.charuco_ids,
            board,
            camera_matrix,
            distortion_coeffs,
            None,
            None,
        )
    except TypeError:
        rvec = np.zeros((3, 1), dtype=float)
        tvec = np.zeros((3, 1), dtype=float)
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            detection.charuco_corners,
            detection.charuco_ids,
            board,
            camera_matrix,
            distortion_coeffs,
            rvec,
            tvec,
            False,
        )

    if not ok:
        raise RuntimeError(f"{image_path.name}: pose estimation failed")

    payload = pose_dict_from_rvec_tvec(cv2, np, rvec, tvec)
    payload.update(
        {
            "image_path": str(image_path),
            "marker_count": detection.marker_count,
            "charuco_corner_count": detection.charuco_corner_count,
        }
    )
    return payload


def estimate_extrinsics_for_path(
    input_path: Path,
    camera_name: str,
    intrinsics_payload: dict[str, Any],
    board_settings: BoardSettings,
    min_corners: int,
    recursive: bool,
) -> dict[str, Any]:
    if input_path.is_file():
        poses = [estimate_pose_from_image(input_path, intrinsics_payload, board_settings, min_corners)]
    else:
        image_paths = iter_image_paths(input_path, recursive)
        if not image_paths:
            raise RuntimeError(f"No supported image files found in {input_path}")
        poses = []
        failures = []
        for image_path in image_paths:
            try:
                poses.append(estimate_pose_from_image(image_path, intrinsics_payload, board_settings, min_corners))
            except Exception as exc:
                failures.append({"image_path": str(image_path), "reason": str(exc)})

        if not poses:
            raise RuntimeError("No valid ChArUco poses could be estimated from the selected images.")

        return {
            "camera_name": camera_name,
            "mode": "extrinsics_batch",
            "intrinsics_source": intrinsics_payload.get("camera_name"),
            "estimated_pose_count": len(poses),
            "failed_pose_count": len(failures),
            "poses": poses,
            "failed_images": failures,
        }

    return {
        "camera_name": camera_name,
        "mode": "extrinsics_single",
        "intrinsics_source": intrinsics_payload.get("camera_name"),
        "pose": poses[0],
    }


def estimate_multi_camera_extrinsics(
    camera_entries: list[dict[str, str]],
    board_settings: BoardSettings,
    min_corners: int,
    reference_camera: str | None,
) -> dict[str, Any]:
    cv2, np = load_backend()
    active_entries = [entry for entry in camera_entries if entry["camera_name"].strip()]
    if len(active_entries) < 2:
        raise RuntimeError("Provide at least two cameras for multi-camera extrinsics.")

    per_camera = []
    for entry in active_entries:
        intrinsics_payload = read_json(Path(entry["intrinsics_path"]))
        pose = estimate_pose_from_image(
            Path(entry["image_path"]),
            intrinsics_payload,
            board_settings,
            min_corners,
        )
        per_camera.append(
            {
                "camera_name": entry["camera_name"].strip(),
                "image_path": entry["image_path"],
                "intrinsics_path": entry["intrinsics_path"],
                **pose,
            }
        )

    if reference_camera:
        reference = next((item for item in per_camera if item["camera_name"] == reference_camera), None)
        if reference is None:
            raise RuntimeError(f"Reference camera not found: {reference_camera}")
    else:
        reference = per_camera[0]

    reference_to_board = np.array(reference["T_camera_to_board"], dtype=float)
    reference_name = reference["camera_name"]

    for item in per_camera:
        board_to_camera = np.array(item["T_board_to_camera"], dtype=float)
        camera_to_board = np.array(item["T_camera_to_board"], dtype=float)
        camera_to_reference = np.linalg.inv(reference_to_board) @ camera_to_board
        reference_to_camera = board_to_camera @ reference_to_board
        item["T_camera_to_reference"] = camera_to_reference.tolist()
        item["T_reference_to_camera"] = reference_to_camera.tolist()
        item["reference_camera"] = reference_name

    return {
        "mode": "multi_camera_extrinsics",
        "reference_camera": reference_name,
        "board": {
            "squares_x": board_settings.squares_x,
            "squares_y": board_settings.squares_y,
            "square_length": board_settings.square_length,
            "marker_length": board_settings.marker_length,
            "dictionary_name": board_settings.dictionary_name,
            "legacy_pattern": board_settings.legacy_pattern,
        },
        "cameras": per_camera,
    }


class MultiCameraRow:
    def __init__(self, parent, index: int, payload: dict[str, str] | None = None):
        defaults = default_multi_camera_payload(index)
        payload = payload or defaults
        camera_name = payload.get("camera_name", defaults["camera_name"]) or defaults["camera_name"]
        self.enabled_var = tk.BooleanVar(value=bool(payload.get("enabled")))
        self.camera_name_var = tk.StringVar(value=camera_name)
        self.intrinsics_var = tk.StringVar(value=payload.get("intrinsics_path", defaults["intrinsics_path"]))
        self.image_var = tk.StringVar(value=payload.get("image_path", defaults["image_path"]))

        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(2, weight=1)
        self.frame.columnconfigure(4, weight=1)

        ttk.Checkbutton(self.frame, text=f"Camera {index + 1}", variable=self.enabled_var).grid(
            row=0, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Entry(self.frame, textvariable=self.camera_name_var, width=12).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Entry(self.frame, textvariable=self.intrinsics_var).grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(self.frame, text="Intrinsics", command=self._browse_intrinsics).grid(
            row=0, column=3, padx=4, pady=4, sticky="ew"
        )
        ttk.Entry(self.frame, textvariable=self.image_var).grid(row=0, column=4, padx=4, pady=4, sticky="ew")
        ttk.Button(self.frame, text="Image", command=self._browse_image).grid(
            row=0, column=5, padx=4, pady=4, sticky="ew"
        )

    def _browse_intrinsics(self):
        path = filedialog.askopenfilename(
            title="Choose intrinsics JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.intrinsics_var.set(path)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Choose synchronized ChArUco image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.image_var.set(path)

    def grid(self, row: int):
        self.frame.grid(row=row, column=0, sticky="ew")

    def destroy(self):
        self.frame.destroy()

    def payload(self) -> dict[str, str]:
        return {
            "enabled": "1" if self.enabled_var.get() else "",
            "camera_name": self.camera_name_var.get().strip(),
            "intrinsics_path": self.intrinsics_var.get().strip(),
            "image_path": self.image_var.get().strip(),
        }


class CharucoCalibratorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1320x940")

        self.status_var = tk.StringVar(value="Ready.")
        self.board_square_x = tk.IntVar(value=8)
        self.board_square_y = tk.IntVar(value=6)
        self.square_length = tk.DoubleVar(value=95.0)
        self.marker_length = tk.DoubleVar(value=71.0)
        self.dictionary_name = tk.StringVar(value="DICT_4X4_100")
        self.legacy_pattern = tk.BooleanVar(value=True)
        self.multi_camera_count = tk.IntVar(value=DEFAULT_MULTI_CAMERA_COUNT)
        self.multi_rows: list[MultiCameraRow] = []
        self.vis_scene_cameras: list[dict[str, Any]] = []
        self.vis_scene_shapes: list[dict[str, Any]] = []
        self.vis_scene_axes: list[dict[str, Any]] = []
        self.vis_scene_center = [0.0, 0.0, 0.0]
        self.vis_scene_radius = 1000.0
        self.vis_yaw = -0.9
        self.vis_pitch = 0.55
        self.vis_distance = 4000.0
        self.vis_pan_x = 0.0
        self.vis_pan_y = 0.0
        self.vis_drag_mode: str | None = None
        self.vis_last_mouse = (0, 0)

        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        settings_frame = ttk.LabelFrame(self.root, text="Board Settings")
        settings_frame.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        settings_frame.columnconfigure(7, weight=1)

        ttk.Label(settings_frame, text="Squares X").grid(row=0, column=0, padx=6, pady=8, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.board_square_x, width=8).grid(
            row=0, column=1, padx=6, pady=8, sticky="w"
        )
        ttk.Label(settings_frame, text="Squares Y").grid(row=0, column=2, padx=6, pady=8, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.board_square_y, width=8).grid(
            row=0, column=3, padx=6, pady=8, sticky="w"
        )
        ttk.Label(settings_frame, text="Square Length").grid(row=0, column=4, padx=6, pady=8, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.square_length, width=10).grid(
            row=0, column=5, padx=6, pady=8, sticky="w"
        )
        ttk.Label(settings_frame, text="Marker Length").grid(row=0, column=6, padx=6, pady=8, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.marker_length, width=10).grid(
            row=0, column=7, padx=6, pady=8, sticky="w"
        )
        ttk.Label(settings_frame, text="Dictionary").grid(row=0, column=8, padx=6, pady=8, sticky="w")
        ttk.Combobox(
            settings_frame,
            textvariable=self.dictionary_name,
            values=ARUCO_DICTIONARIES,
            state="readonly",
            width=18,
        ).grid(row=0, column=9, padx=6, pady=8, sticky="w")
        ttk.Checkbutton(
            settings_frame,
            text="Use Legacy Pattern",
            variable=self.legacy_pattern,
        ).grid(row=0, column=10, padx=6, pady=8, sticky="w")

        hint = (
            "Lengths can be meters, millimeters, or any other unit, as long as square and marker "
            "lengths use the same unit."
        )
        ttk.Label(settings_frame, text=hint).grid(row=1, column=0, columnspan=10, padx=6, pady=(0, 8), sticky="w")
        ttk.Label(
            settings_frame,
            text="Starting defaults: OpenCV 8 x 6 board, square length 95, marker length 71, DICT_4X4_100, legacy pattern on.",
        ).grid(row=2, column=0, columnspan=11, padx=6, pady=(0, 8), sticky="w")

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")

        self._build_intrinsics_tab(notebook)
        self._build_extrinsics_tab(notebook)
        self._build_multi_camera_tab(notebook)
        self._build_visualizer_tab(notebook)

        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

    def _build_intrinsics_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        notebook.add(tab, text="Intrinsics")

        self.intr_camera_name = tk.StringVar(value="cam1")
        self.intr_folder = tk.StringVar(value="")
        self.intr_output = tk.StringVar(value="")
        self.intr_min_corners = tk.IntVar(value=12)
        self.intr_recursive = tk.BooleanVar(value=False)

        form = ttk.Frame(tab)
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Camera Name").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.intr_camera_name).grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        ttk.Label(form, text="Frame Folder").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.intr_folder).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(form, text="Browse", command=self._browse_intrinsics_folder).grid(
            row=1, column=2, padx=4, pady=4, sticky="ew"
        )

        ttk.Label(form, text="Output JSON").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.intr_output).grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(form, text="Save As", command=self._browse_intrinsics_output).grid(
            row=2, column=2, padx=4, pady=4, sticky="ew"
        )

        ttk.Label(form, text="Min Corners").grid(row=3, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.intr_min_corners, width=10).grid(row=3, column=1, padx=4, pady=4, sticky="w")
        ttk.Checkbutton(form, text="Search Recursively", variable=self.intr_recursive).grid(
            row=3, column=2, padx=4, pady=4, sticky="w"
        )

        ttk.Button(form, text="Run Intrinsics Calibration", command=self.run_intrinsics).grid(
            row=4, column=0, columnspan=3, padx=4, pady=8, sticky="ew"
        )
        ttk.Button(form, text="Run All Camera Folders", command=self.run_all_intrinsics).grid(
            row=5, column=0, columnspan=3, padx=4, pady=(0, 8), sticky="ew"
        )

        self.intr_log = ScrolledText(tab, wrap="word", font=("Courier", 11))
        self.intr_log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _build_extrinsics_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        notebook.add(tab, text="Single-Camera Extrinsics")

        self.ext_camera_name = tk.StringVar(value="cam1")
        self.ext_intrinsics = tk.StringVar(value="")
        self.ext_input = tk.StringVar(value="")
        self.ext_output = tk.StringVar(value="")
        self.ext_min_corners = tk.IntVar(value=8)
        self.ext_recursive = tk.BooleanVar(value=False)

        form = ttk.Frame(tab)
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Camera Name").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.ext_camera_name).grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        ttk.Label(form, text="Intrinsics JSON").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.ext_intrinsics).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(form, text="Browse", command=self._browse_extrinsics_intrinsics).grid(
            row=1, column=2, padx=4, pady=4, sticky="ew"
        )

        ttk.Label(form, text="Image Or Folder").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.ext_input).grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        input_buttons = ttk.Frame(form)
        input_buttons.grid(row=2, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(input_buttons, text="Image", command=self._browse_extrinsics_image).grid(
            row=0, column=0, padx=(0, 4), sticky="ew"
        )
        ttk.Button(input_buttons, text="Folder", command=self._browse_extrinsics_folder).grid(
            row=0, column=1, sticky="ew"
        )

        ttk.Label(form, text="Output JSON").grid(row=3, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.ext_output).grid(row=3, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(form, text="Save As", command=self._browse_extrinsics_output).grid(
            row=3, column=2, padx=4, pady=4, sticky="ew"
        )

        ttk.Label(form, text="Min Corners").grid(row=4, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.ext_min_corners, width=10).grid(row=4, column=1, padx=4, pady=4, sticky="w")
        ttk.Checkbutton(form, text="Search Recursively", variable=self.ext_recursive).grid(
            row=4, column=2, padx=4, pady=4, sticky="w"
        )

        ttk.Button(form, text="Run Pose / Extrinsics Estimation", command=self.run_extrinsics).grid(
            row=5, column=0, columnspan=3, padx=4, pady=8, sticky="ew"
        )

        self.ext_log = ScrolledText(tab, wrap="word", font=("Courier", 11))
        self.ext_log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _build_multi_camera_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        notebook.add(tab, text="Multi-Camera Extrinsics")

        self.multi_output = tk.StringVar(value="")
        self.multi_reference = tk.StringVar(value="cam1")
        self.multi_min_corners = tk.IntVar(value=8)

        form = ttk.Frame(tab)
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(5, weight=1)

        ttk.Label(form, text="Reference Camera").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.multi_reference, width=14).grid(row=0, column=1, padx=4, pady=4, sticky="w")
        ttk.Label(form, text="Min Corners").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.multi_min_corners, width=10).grid(row=0, column=3, padx=4, pady=4, sticky="w")
        ttk.Label(form, text="Camera Count").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.multi_camera_count, width=10).grid(row=0, column=5, padx=4, pady=4, sticky="w")
        ttk.Button(form, text="Apply Count", command=self._apply_multi_camera_count).grid(
            row=0, column=6, padx=4, pady=4, sticky="ew"
        )
        ttk.Label(form, text="Output JSON").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.multi_output).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(form, text="Save As", command=self._browse_multi_output).grid(
            row=1, column=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(form, text="Run Multi-Camera Extrinsics", command=self.run_multi_camera).grid(
            row=1, column=3, padx=4, pady=4, sticky="ew"
        )

        rows_frame = ttk.LabelFrame(tab, text="One synchronized ChArUco image per camera")
        rows_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        rows_frame.columnconfigure(0, weight=1)

        ttk.Label(
            rows_frame,
            text="Set the camera count, then point each enabled row at that camera's intrinsics JSON and one synchronized ChArUco image.",
        ).grid(row=0, column=0, padx=4, pady=(4, 8), sticky="w")

        self.multi_rows_container = ttk.Frame(rows_frame)
        self.multi_rows_container.grid(row=1, column=0, sticky="ew")
        self.multi_rows_container.columnconfigure(0, weight=1)
        self._rebuild_multi_camera_rows(self.multi_camera_count.get())

        self.multi_log = ScrolledText(tab, wrap="word", font=("Courier", 11))
        self.multi_log.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _build_visualizer_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        notebook.add(tab, text="Visualize Extrinsics")

        self.vis_json_path = tk.StringVar(value="")

        form = ttk.Frame(tab)
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Extrinsics JSON").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(form, textvariable=self.vis_json_path).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(form, text="Browse", command=self._browse_visualizer_json).grid(
            row=0, column=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(form, text="Load And Draw", command=self.run_visualizer).grid(
            row=0, column=3, padx=4, pady=4, sticky="ew"
        )

        content = ttk.Frame(tab)
        content.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)
        content.rowconfigure(1, weight=0)

        self.vis_canvas = tk.Canvas(content, width=860, height=620, bg="#f8f6f1", highlightthickness=1)
        self.vis_summary = ScrolledText(content, wrap="word", font=("Courier", 10), height=18)
        self.vis_help = ttk.Label(
            content,
            text="Drag to orbit. Shift-drag to pan. Mouse wheel to zoom. Double-click to reset. Axes: X red, Y green, Z blue.",
        )

        self.vis_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self.vis_summary.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self.vis_help.grid(row=1, column=0, columnspan=2, sticky="w", padx=2, pady=(0, 2))

        self.vis_canvas.bind("<ButtonPress-1>", self._on_vis_mouse_down)
        self.vis_canvas.bind("<B1-Motion>", self._on_vis_mouse_drag)
        self.vis_canvas.bind("<Double-Button-1>", self._reset_visualizer_view)
        self.vis_canvas.bind("<MouseWheel>", self._on_vis_mouse_wheel)
        self.vis_canvas.bind("<Button-4>", self._on_vis_mouse_wheel)
        self.vis_canvas.bind("<Button-5>", self._on_vis_mouse_wheel)
        self.vis_canvas.bind("<Configure>", self._on_vis_canvas_resize)

    def board_settings(self) -> BoardSettings:
        settings = BoardSettings(
            squares_x=self.board_square_x.get(),
            squares_y=self.board_square_y.get(),
            square_length=self.square_length.get(),
            marker_length=self.marker_length.get(),
            dictionary_name=self.dictionary_name.get().strip(),
            legacy_pattern=bool(self.legacy_pattern.get()),
        )
        if settings.squares_x < 2 or settings.squares_y < 2:
            raise ValueError("Squares X and Y must both be at least 2.")
        if settings.square_length <= 0 or settings.marker_length <= 0:
            raise ValueError("Board lengths must be positive.")
        if settings.marker_length >= settings.square_length:
            raise ValueError("Marker length must be smaller than square length.")
        return settings

    def set_status(self, message: str):
        self.status_var.set(message)
        self.root.update_idletasks()

    def write_log(self, widget: ScrolledText, message: str):
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, message.rstrip() + "\n")
        widget.see(tk.END)

    def append_log(self, widget: ScrolledText, message: str):
        widget.insert(tk.END, message.rstrip() + "\n")
        widget.see(tk.END)

    def _require_path_value(self, value: str, label: str) -> Path:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"Choose {label}.")
        return Path(cleaned)

    def _resolve_intrinsics_output_path(self) -> Path:
        cleaned = self.intr_output.get().strip()
        return Path(cleaned) if cleaned else default_intrinsics_output_path(self.intr_camera_name.get())

    def _resolve_batch_intrinsics_output_dir(self) -> Path:
        cleaned = self.intr_output.get().strip()
        return Path(cleaned) if cleaned else DEFAULT_CALIBRATION_ROOT

    def _resolve_extrinsics_output_path(self, input_path: Path) -> Path:
        cleaned = self.ext_output.get().strip()
        if cleaned:
            return Path(cleaned)
        if input_path.is_dir():
            return default_batch_extrinsics_output_path(self.ext_camera_name.get(), input_path)
        return default_extrinsics_output_path(self.ext_camera_name.get())

    def _resolve_multi_output_path(self) -> Path:
        cleaned = self.multi_output.get().strip()
        return Path(cleaned) if cleaned else DEFAULT_CALIBRATION_ROOT / "multi_camera_extrinsics.json"

    def _current_multi_row_payloads(self) -> list[dict[str, str]]:
        return [row.payload() for row in self.multi_rows]

    def _rebuild_multi_camera_rows(self, count: int):
        existing_payloads = self._current_multi_row_payloads()
        camera_names = []
        for row in self.multi_rows:
            row.destroy()
        self.multi_rows = []

        for index in range(count):
            payload = existing_payloads[index] if index < len(existing_payloads) else default_multi_camera_payload(index)
            row = MultiCameraRow(self.multi_rows_container, index, payload)
            row.grid(index)
            self.multi_rows.append(row)
            camera_name = payload.get("camera_name", "").strip()
            if camera_name:
                camera_names.append(camera_name)

        if camera_names and self.multi_reference.get().strip() not in camera_names:
            self.multi_reference.set(camera_names[0])

    def _apply_multi_camera_count(self):
        count = int(self.multi_camera_count.get())
        if count < 2:
            raise ValueError("Camera count must be at least 2 for multi-camera extrinsics.")
        self._rebuild_multi_camera_rows(count)
        self.set_status(f"Configured {count} multi-camera rows.")

    def _browse_intrinsics_folder(self):
        path = filedialog.askdirectory(title="Choose ChArUco frame folder")
        if path:
            self.intr_folder.set(path)
            self.intr_output.set(str(default_intrinsics_output_path(self.intr_camera_name.get())))

    def _browse_intrinsics_output(self):
        path = filedialog.asksaveasfilename(
            title="Save intrinsics JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.intr_output.set(path)

    def _browse_extrinsics_intrinsics(self):
        path = filedialog.askopenfilename(
            title="Choose intrinsics JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.ext_intrinsics.set(path)

    def _browse_extrinsics_image(self):
        path = filedialog.askopenfilename(
            title="Choose ChArUco image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.ext_input.set(path)
            self.ext_output.set(str(default_extrinsics_output_path(self.ext_camera_name.get())))

    def _browse_extrinsics_folder(self):
        path = filedialog.askdirectory(title="Choose ChArUco image folder")
        if path:
            self.ext_input.set(path)
            self.ext_output.set(str(default_batch_extrinsics_output_path(self.ext_camera_name.get(), Path(path))))

    def _browse_extrinsics_output(self):
        path = filedialog.asksaveasfilename(
            title="Save extrinsics JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.ext_output.set(path)

    def _browse_multi_output(self):
        path = filedialog.asksaveasfilename(
            title="Save multi-camera extrinsics JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.multi_output.set(path)

    def _browse_visualizer_json(self):
        path = filedialog.askopenfilename(
            title="Choose extrinsics JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.vis_json_path.set(path)

    def run_intrinsics(self):
        try:
            board_settings = self.board_settings()
            image_folder = self._require_path_value(self.intr_folder.get(), "a ChArUco frame folder")
            output_path = self._resolve_intrinsics_output_path()
            camera_name = self.intr_camera_name.get().strip() or "camera"
            min_corners = int(self.intr_min_corners.get())
            self.intr_output.set(str(output_path))

            self.set_status("Running intrinsics calibration...")
            payload = calibrate_intrinsics(
                image_folder=image_folder,
                camera_name=camera_name,
                board_settings=board_settings,
                min_corners=min_corners,
                recursive=bool(self.intr_recursive.get()),
            )
            write_json(output_path, payload)

            lines = [
                f"Intrinsics calibration complete for {camera_name}",
                f"Saved: {output_path}",
                f"Accepted frames: {payload['accepted_frame_count']}",
                f"Rejected frames: {payload['rejected_frame_count']}",
                f"Reprojection error: {payload['reprojection_error']:.6f}",
                "",
                "Camera matrix:",
                json.dumps(payload["camera_matrix"], indent=2),
                "",
                "Distortion coefficients:",
                json.dumps(payload["distortion_coefficients"], indent=2),
            ]
            if payload["rejected_frames"]:
                lines.extend(["", "Rejected frames:"])
                lines.extend(
                    f"- {item['image_path']} :: {item['reason']} ({item['charuco_corner_count']} corners)"
                    for item in payload["rejected_frames"][:50]
                )

            self.write_log(self.intr_log, "\n".join(lines))
            self.ext_intrinsics.set(str(output_path))
            self.set_status(f"Intrinsics saved to {output_path}")
        except Exception as exc:
            self._handle_error(self.intr_log, "Intrinsics calibration failed", exc)

    def run_all_intrinsics(self):
        try:
            board_settings = self.board_settings()
            selected_folder = self._require_path_value(self.intr_folder.get(), "a ChArUco frame folder")
            output_path = self._resolve_batch_intrinsics_output_dir()
            min_corners = int(self.intr_min_corners.get())
            recursive = bool(self.intr_recursive.get())
            self.intr_output.set(str(output_path))

            if not selected_folder.exists():
                raise FileNotFoundError(f"Folder does not exist: {selected_folder}")

            candidate_root = selected_folder
            if looks_like_camera_folder(selected_folder) and selected_folder.parent.exists():
                sibling_folders = sorted(path for path in selected_folder.parent.iterdir() if looks_like_camera_folder(path))
                camera_folders = sibling_folders if sibling_folders else [selected_folder]
                search_root = selected_folder.parent
            else:
                camera_folders = sorted(path for path in selected_folder.iterdir() if looks_like_camera_folder(path))
                search_root = selected_folder

            if not camera_folders:
                raise RuntimeError(
                    f"No camera folders with images were found under {candidate_root}. "
                    "Point the Frame Folder at one camera folder like cam1 or at the parent intrinsics folder."
                )

            output_dir = output_path.parent if output_path.suffix.lower() == ".json" else output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            combined_output_path = output_dir / "all_intrinsics.json"

            self.set_status(f"Running batch intrinsics for {len(camera_folders)} camera folders...")

            results = {}
            log_lines = [
                f"Batch intrinsics root: {search_root}",
                f"Output directory: {output_dir}",
                "",
            ]

            for folder in camera_folders:
                camera_name = sanitize_name(folder.name)
                camera_output_path = output_dir / f"{camera_name}_intrinsics.json"
                payload = calibrate_intrinsics(
                    image_folder=folder,
                    camera_name=camera_name,
                    board_settings=board_settings,
                    min_corners=min_corners,
                    recursive=recursive,
                )
                write_json(camera_output_path, payload)
                results[camera_name] = payload
                log_lines.extend(
                    [
                        f"{camera_name}",
                        f"  frames folder: {folder}",
                        f"  saved: {camera_output_path}",
                        f"  accepted: {payload['accepted_frame_count']}",
                        f"  rejected: {payload['rejected_frame_count']}",
                        f"  reprojection error: {payload['reprojection_error']:.6f}",
                        "",
                    ]
                )

            combined_payload = {
                "mode": "multi_camera_intrinsics_bundle",
                "camera_count": len(results),
                "board": {
                    "squares_x": board_settings.squares_x,
                    "squares_y": board_settings.squares_y,
                    "square_length": board_settings.square_length,
                    "marker_length": board_settings.marker_length,
                    "dictionary_name": board_settings.dictionary_name,
                    "legacy_pattern": board_settings.legacy_pattern,
                },
                "cameras": results,
            }
            write_json(combined_output_path, combined_payload)

            first_camera = next(iter(results)) if results else None
            if first_camera:
                self.ext_intrinsics.set(str(output_dir / f"{first_camera}_intrinsics.json"))

            log_lines.extend(
                [
                    f"Combined bundle saved: {combined_output_path}",
                    f"Cameras solved: {len(results)}",
                ]
            )
            self.write_log(self.intr_log, "\n".join(log_lines))
            self.set_status(f"Batch intrinsics complete. Saved {len(results)} camera JSON files and all_intrinsics.json")
        except Exception as exc:
            self._handle_error(self.intr_log, "Batch intrinsics failed", exc)

    def run_extrinsics(self):
        try:
            board_settings = self.board_settings()
            camera_name = self.ext_camera_name.get().strip() or "camera"
            intrinsics_path = self._require_path_value(self.ext_intrinsics.get(), "an intrinsics JSON file")
            input_path = self._require_path_value(self.ext_input.get(), "an image or folder")
            output_path = self._resolve_extrinsics_output_path(input_path)
            min_corners = int(self.ext_min_corners.get())
            self.ext_output.set(str(output_path))

            intrinsics_payload = read_json(intrinsics_path)
            self.set_status("Running single-camera extrinsics estimation...")
            payload = estimate_extrinsics_for_path(
                input_path=input_path,
                camera_name=camera_name,
                intrinsics_payload=intrinsics_payload,
                board_settings=board_settings,
                min_corners=min_corners,
                recursive=bool(self.ext_recursive.get()),
            )
            write_json(output_path, payload)

            lines = [
                f"Extrinsics complete for {camera_name}",
                f"Saved: {output_path}",
                f"Mode: {payload['mode']}",
            ]
            if payload["mode"] == "extrinsics_single":
                lines.extend(
                    [
                        f"Image: {payload['pose']['image_path']}",
                        f"ChArUco corners: {payload['pose']['charuco_corner_count']}",
                        "",
                        "Board to camera transform:",
                        json.dumps(payload["pose"]["T_board_to_camera"], indent=2),
                    ]
                )
            else:
                lines.extend(
                    [
                        f"Estimated poses: {payload['estimated_pose_count']}",
                        f"Failed images: {payload['failed_pose_count']}",
                    ]
                )
                if payload["poses"]:
                    lines.extend(
                        [
                            "",
                            "First pose sample:",
                            json.dumps(payload["poses"][0], indent=2),
                        ]
                    )

            self.write_log(self.ext_log, "\n".join(lines))
            self.set_status(f"Extrinsics saved to {output_path}")
        except Exception as exc:
            self._handle_error(self.ext_log, "Extrinsics estimation failed", exc)

    def run_multi_camera(self):
        try:
            board_settings = self.board_settings()
            output_path = self._resolve_multi_output_path()
            camera_entries = []
            self.multi_output.set(str(output_path))
            for row in self.multi_rows:
                payload = row.payload()
                if not payload.get("enabled"):
                    continue
                if not payload["camera_name"]:
                    continue
                if not payload["intrinsics_path"] or not payload["image_path"]:
                    raise ValueError(
                        f"{payload['camera_name']}: both intrinsics JSON and synchronized image are required."
                    )
                camera_entries.append(payload)

            self.set_status("Running multi-camera extrinsics estimation...")
            payload = estimate_multi_camera_extrinsics(
                camera_entries=camera_entries,
                board_settings=board_settings,
                min_corners=int(self.multi_min_corners.get()),
                reference_camera=self.multi_reference.get().strip() or None,
            )
            write_json(output_path, payload)

            lines = [
                f"Multi-camera extrinsics complete",
                f"Saved: {output_path}",
                f"Reference camera: {payload['reference_camera']}",
                f"Cameras solved: {len(payload['cameras'])}",
                "",
            ]
            for item in payload["cameras"]:
                lines.extend(
                    [
                        f"{item['camera_name']}",
                        f"  image: {item['image_path']}",
                        f"  corners: {item['charuco_corner_count']}",
                        f"  T_camera_to_reference:",
                        json.dumps(item["T_camera_to_reference"], indent=2),
                        "",
                    ]
                )

            self.write_log(self.multi_log, "\n".join(lines))
            self.vis_json_path.set(str(output_path))
            self.set_status(f"Multi-camera extrinsics saved to {output_path}")
        except Exception as exc:
            self._handle_error(self.multi_log, "Multi-camera extrinsics failed", exc)

    def _camera_wireframe_points(self, scale: float) -> list[list[float]]:
        return [
            [0.0, 0.0, 0.0],
            [-0.6 * scale, -0.4 * scale, scale],
            [0.6 * scale, -0.4 * scale, scale],
            [0.6 * scale, 0.4 * scale, scale],
            [-0.6 * scale, 0.4 * scale, scale],
        ]

    def _board_outline_points(self, board_info: dict[str, Any]) -> list[list[float]]:
        width = float(board_info["squares_x"]) * float(board_info["square_length"])
        height = float(board_info["squares_y"]) * float(board_info["square_length"])
        return [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
        ]

    def _board_cuboid_shape(self, board_info: dict[str, Any], transform: list[list[float]] | Any | None = None) -> dict[str, Any]:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Visualization needs numpy installed. Import error: {exc}") from exc

        width = float(board_info["squares_x"]) * float(board_info["square_length"])
        height = float(board_info["squares_y"]) * float(board_info["square_length"])
        thickness = max(float(board_info.get("marker_length", 20.0)) * 0.12, 8.0)

        points = [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
            [0.0, 0.0, thickness],
            [width, 0.0, thickness],
            [width, height, thickness],
            [0.0, height, thickness],
        ]
        if transform is not None:
            points = transform_points(np, transform, points).tolist()

        return {
            "name": "board",
            "points": points,
            "edges": [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ],
            "color": "#7a3e1d",
        }

    def _axis_segments(self, scale: float) -> list[dict[str, Any]]:
        return [
            {"name": "X", "points": [[0.0, 0.0, 0.0], [scale, 0.0, 0.0]], "color": "#dc2626"},
            {"name": "Y", "points": [[0.0, 0.0, 0.0], [0.0, scale, 0.0]], "color": "#16a34a"},
            {"name": "Z", "points": [[0.0, 0.0, 0.0], [0.0, 0.0, scale]], "color": "#2563eb"},
        ]

    def _prepare_visual_scene(
        self, payload: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Visualization needs numpy installed. Import error: {exc}") from exc

        mode = payload.get("mode", "")
        scene_cameras: list[dict[str, Any]] = []
        scene_shapes: list[dict[str, Any]] = []
        scene_axes: list[dict[str, Any]] = []
        summary_lines: list[str] = [f"Mode: {mode}"]

        if mode == "multi_camera_extrinsics":
            board_info = payload.get("board", {})
            axis_scale = max(float(board_info.get("square_length", 100.0)) * 2.0, 200.0)
            scene_axes = self._axis_segments(axis_scale)
            reference_name = payload.get("reference_camera", "reference")
            reference_entry = next(
                (item for item in payload.get("cameras", []) if item.get("camera_name") == reference_name),
                None,
            )
            if reference_entry and reference_entry.get("T_board_to_camera"):
                scene_shapes.append(self._board_cuboid_shape(board_info, reference_entry["T_board_to_camera"]))

            for index, item in enumerate(payload.get("cameras", [])):
                transform = item.get("T_camera_to_reference")
                if transform is None:
                    continue
                color = ["#c2410c", "#0f766e", "#2563eb", "#7c3aed", "#b45309", "#be123c"][index % 6]
                scene_cameras.append(
                    {
                        "name": item["camera_name"],
                        "transform": transform,
                        "color": color,
                    }
                )
                summary_lines.extend(
                    [
                        f"{item['camera_name']}",
                        f"  image: {item.get('image_path', '-')}",
                        f"  corners: {item.get('charuco_corner_count', '-')}",
                    ]
                )
            summary_lines.extend(
                [
                    "",
                    "Reference-frame axes:",
                    "  X = red",
                    "  Y = green",
                    "  Z = blue",
                ]
            )
        elif mode == "extrinsics_single":
            pose = payload.get("pose", {})
            board_info = payload.get("board", {})
            axis_scale = max(float(board_info.get("square_length", 100.0)) * 2.0, 200.0) if board_info else 200.0
            scene_axes = self._axis_segments(axis_scale)
            if board_info:
                scene_shapes.append(self._board_cuboid_shape(board_info))
            scene_cameras.append(
                {
                    "name": payload.get("camera_name", "camera"),
                    "transform": pose.get("T_camera_to_board"),
                    "color": "#2563eb",
                }
            )
            summary_lines.extend(
                [
                    f"Camera: {payload.get('camera_name', 'camera')}",
                    f"Image: {pose.get('image_path', '-')}",
                    f"Corners: {pose.get('charuco_corner_count', '-')}",
                    "",
                    "Board-frame axes:",
                    "  X = red",
                    "  Y = green",
                    "  Z = blue",
                ]
            )
        else:
            raise RuntimeError(
                "Visualization currently supports `multi_camera_extrinsics` and `extrinsics_single` JSON outputs."
            )

        return mode, scene_cameras, scene_shapes, scene_axes, "\n".join(summary_lines)

    def _refresh_visual_scene_bounds(self):
        try:
            import numpy as np  # type: ignore
        except Exception:
            self.vis_scene_center = [0.0, 0.0, 0.0]
            self.vis_scene_radius = 1000.0
            return

        all_points = []
        for shape in self.vis_scene_shapes:
            all_points.extend(shape["points"])
        for axis in self.vis_scene_axes:
            all_points.extend(axis["points"])
        for cam in self.vis_scene_cameras:
            wire = transform_points(np, cam["transform"], self._camera_wireframe_points(180.0))
            cam["wireframe_points"] = wire.tolist()
            all_points.extend(wire.tolist())

        if not all_points:
            self.vis_scene_center = [0.0, 0.0, 0.0]
            self.vis_scene_radius = 1000.0
            return

        array = np.array(all_points, dtype=float)
        center = array.mean(axis=0)
        radius = float(np.max(np.linalg.norm(array - center, axis=1)))
        self.vis_scene_center = center.tolist()
        self.vis_scene_radius = max(radius, 500.0)

    def _reset_visualizer_view(self, _event=None):
        self.vis_yaw = -0.9
        self.vis_pitch = 0.55
        self.vis_pan_x = 0.0
        self.vis_pan_y = 0.0
        self.vis_distance = max(self.vis_scene_radius * 3.2, 1600.0)
        self._redraw_visualizer()
        return "break"

    def _on_vis_mouse_down(self, event):
        self.vis_drag_mode = "pan" if (event.state & 0x0001) else "orbit"
        self.vis_last_mouse = (event.x, event.y)

    def _on_vis_mouse_drag(self, event):
        dx = event.x - self.vis_last_mouse[0]
        dy = event.y - self.vis_last_mouse[1]
        self.vis_last_mouse = (event.x, event.y)

        if self.vis_drag_mode == "pan":
            self.vis_pan_x += dx
            self.vis_pan_y += dy
        else:
            self.vis_yaw += dx * 0.01
            self.vis_pitch = max(-1.45, min(1.45, self.vis_pitch + dy * 0.01))
        self._redraw_visualizer()

    def _on_vis_mouse_wheel(self, event):
        if getattr(event, "num", None) == 4:
            factor = 0.9
        elif getattr(event, "num", None) == 5:
            factor = 1.1
        else:
            factor = 0.9 if event.delta > 0 else 1.1
        self.vis_distance = max(200.0, self.vis_distance * factor)
        self._redraw_visualizer()
        return "break"

    def _on_vis_canvas_resize(self, _event):
        self._redraw_visualizer()

    def _project_points_3d(self, points: list[list[float]]):
        import numpy as np  # type: ignore

        width = int(self.vis_canvas.winfo_width() or self.vis_canvas["width"])
        height = int(self.vis_canvas.winfo_height() or self.vis_canvas["height"])
        pts = np.array(points, dtype=float) - np.array(self.vis_scene_center, dtype=float)
        pts[:, 0] *= -1.0
        pts[:, 1] *= -1.0

        cy = math.cos(self.vis_yaw)
        sy = math.sin(self.vis_yaw)
        cp = math.cos(self.vis_pitch)
        sp = math.sin(self.vis_pitch)
        rotate_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
        rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=float)
        rotated = pts @ (rotate_x @ rotate_y).T

        depth = rotated[:, 2] + self.vis_distance
        focal = min(width, height) * 0.9

        projected = []
        for point, z in zip(rotated, depth):
            if z <= 1.0:
                projected.append(None)
                continue
            x = width / 2 + self.vis_pan_x + (point[0] * focal / z)
            y = height / 2 + self.vis_pan_y - (point[1] * focal / z)
            projected.append((x, y, z))
        return projected

    def _redraw_visualizer(self):
        width = int(self.vis_canvas.winfo_width() or self.vis_canvas["width"])
        height = int(self.vis_canvas.winfo_height() or self.vis_canvas["height"])
        self.vis_canvas.delete("all")

        if not self.vis_scene_cameras and not self.vis_scene_shapes:
            self.vis_canvas.create_text(width / 2, height / 2, text="Load an extrinsics JSON to visualize", fill="#444")
            return

        self.vis_canvas.create_rectangle(0, 0, width, height, fill="#f8f6f1", outline="")
        self.vis_canvas.create_text(
            10,
            10,
            text="Interactive 3D View",
            anchor="nw",
            fill="#2d2418",
            font=("Helvetica", 12, "bold"),
        )

        segments = []
        labels = []

        for axis in self.vis_scene_axes:
            projected = self._project_points_3d(axis["points"])
            start = projected[0]
            end = projected[1]
            if start is not None and end is not None:
                depth = (start[2] + end[2]) / 2
                segments.append((depth, axis["color"], 3, start[:2], end[:2]))
                labels.append((end[2], axis["name"], axis["color"], end[0], end[1]))

        for shape in self.vis_scene_shapes:
            points = shape["points"]
            projected = self._project_points_3d(points)
            pairs = shape.get("edges")
            if not pairs:
                pairs = list(zip(range(len(points)), range(1, len(points))))
                if shape.get("closed", False) and len(points) > 2:
                    pairs.append((len(points) - 1, 0))
            for start_idx, end_idx in pairs:
                start = projected[start_idx]
                end = projected[end_idx]
                if start is None or end is None:
                    continue
                depth = (start[2] + end[2]) / 2
                segments.append((depth, shape["color"], 3, start[:2], end[:2]))

        for cam in self.vis_scene_cameras:
            points = cam.get("wireframe_points") or []
            if not points:
                continue
            projected = self._project_points_3d(points)
            edge_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
            for start_idx, end_idx in edge_pairs:
                start = projected[start_idx]
                end = projected[end_idx]
                if start is None or end is None:
                    continue
                depth = (start[2] + end[2]) / 2
                segments.append((depth, cam["color"], 2, start[:2], end[:2]))
            origin = projected[0]
            if origin is not None:
                labels.append((origin[2], cam["name"], cam["color"], origin[0], origin[1]))

        for _depth, color, line_width, start, end in sorted(segments, key=lambda item: item[0], reverse=True):
            self.vis_canvas.create_line(*start, *end, fill=color, width=line_width)

        for _depth, label, color, x, y in sorted(labels, key=lambda item: item[0], reverse=True):
            self.vis_canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
            self.vis_canvas.create_text(x + 8, y - 8, text=label, fill=color, anchor="sw", font=("Helvetica", 10, "bold"))

    def run_visualizer(self):
        try:
            json_path = self._require_path_value(self.vis_json_path.get(), "an extrinsics JSON file")
            payload = read_json(json_path)
            _mode, scene_cameras, scene_shapes, scene_axes, summary = self._prepare_visual_scene(payload)
            self.vis_scene_cameras = scene_cameras
            self.vis_scene_shapes = scene_shapes
            self.vis_scene_axes = scene_axes
            self._refresh_visual_scene_bounds()
            self._reset_visualizer_view()
            self.write_log(self.vis_summary, summary)
            self.vis_json_path.set(str(json_path))
            self.set_status(f"Extrinsics visualization loaded from {json_path}")
        except Exception as exc:
            self._handle_error(self.vis_summary, "Extrinsics visualization failed", exc)

    def _handle_error(self, widget: ScrolledText, title: str, exc: Exception):
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        self.write_log(widget, f"{title}\n\n{details}")
        self.set_status(details)
        messagebox.showerror(title, details)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--self-check", action="store_true", help="Exit after a lightweight import/syntax check.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        print("Self check OK")
        return 0

    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    app = CharucoCalibratorApp(root)
    app.set_status("Ready. Set your board parameters, then run calibration.")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
