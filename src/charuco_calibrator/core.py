"""Headless ChArUco detection, calibration and extrinsics (no tkinter dependency)."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpg", ".mts", ".webm"}
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
    "This calibrator needs numpy and opencv-contrib-python>=4.8.\n\n"
    "Install them with:\n"
    "python3 -m pip install -r requirements.txt"
)

FX_FY_RATIO_RANGE = (0.9, 1.1)
PRINCIPAL_POINT_MAX_OFFSET = 0.1
FOCAL_WIDTH_RATIO_RANGE = (0.25, 5.0)
MAX_RELATIVE_FOCAL_STD = 0.02
MIN_DISTINCT_ORIENTATIONS = 3
ORIENTATION_SEPARATION_DEG = 10.0
MIN_AXIS_TILT_RANGE_DEG = 10.0
MIN_COVERAGE_FRACTION = 0.3
COVERAGE_GRID = (8, 6)
DEFAULT_MAX_INTRINSICS_FRAMES = 40
DEFAULT_MAX_FRAME_SETS = 50

ProgressCallback = Optional[Callable[[str], None]]


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


@dataclass
class IntrinsicsOptions:
    fix_aspect_ratio: bool = False
    fix_k3: bool = True
    zero_tangential: bool = False
    fix_principal_point: bool = False
    max_frames: int = DEFAULT_MAX_INTRINSICS_FRAMES
    auto_constrain: bool = True


@dataclass
class FrameSampling:
    every: int = 1
    start: int = 0
    end: int | None = None


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
    if not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} is too old: cv2.aruco.CharucoDetector is missing.\n\n{DEPENDENCY_HINT}"
        )

    return cv2, np


def get_dictionary(cv2, dictionary_name: str):
    dict_id = getattr(cv2.aruco, dictionary_name, None)
    if dict_id is None:
        raise ValueError(f"Unsupported dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(dict_id)


def board_key(settings: BoardSettings) -> tuple:
    return (
        int(settings.squares_x),
        int(settings.squares_y),
        float(settings.square_length),
        float(settings.marker_length),
        settings.dictionary_name,
        bool(settings.legacy_pattern),
    )


def board_payload(settings: BoardSettings) -> dict[str, Any]:
    return {
        "squares_x": settings.squares_x,
        "squares_y": settings.squares_y,
        "square_length": settings.square_length,
        "marker_length": settings.marker_length,
        "dictionary_name": settings.dictionary_name,
        "legacy_pattern": settings.legacy_pattern,
    }


def create_board(cv2, settings: BoardSettings):
    board = cv2.aruco.CharucoBoard(
        (int(settings.squares_x), int(settings.squares_y)),
        float(settings.square_length),
        float(settings.marker_length),
        get_dictionary(cv2, settings.dictionary_name),
    )
    if settings.legacy_pattern:
        board.setLegacyPattern(True)
    return board


def create_detector_parameters(cv2):
    return cv2.aruco.DetectorParameters()


_ENGINES: dict[tuple, tuple[Any, Any]] = {}


def board_engine(cv2, settings: BoardSettings):
    key = board_key(settings)
    if key not in _ENGINES:
        board = create_board(cv2, settings)
        _ENGINES[key] = (board, cv2.aruco.CharucoDetector(board))
    return _ENGINES[key]


def iter_image_paths(folder: Path, recursive: bool) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def video_frame_label(video_path: Path, frame_index: int) -> str:
    return f"{video_path}#frame={frame_index}"


def iter_video_frames(
    video_path: Path,
    sampling: FrameSampling | None = None,
    frame_indices: list[int] | None = None,
) -> Iterator[tuple[int, Any]]:
    cv2, _np = load_backend()
    sampling = sampling or FrameSampling()
    every = max(1, int(sampling.every))
    wanted = sorted(set(int(index) for index in frame_indices)) if frame_indices else None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        index = 0
        while True:
            if sampling.end is not None and index >= sampling.end:
                break
            if wanted is not None:
                if index > wanted[-1]:
                    break
                take = index in wanted
            else:
                take = index >= sampling.start and (index - sampling.start) % every == 0
            if take:
                ok, frame = capture.read()
            else:
                ok, frame = capture.grab(), None
            if not ok:
                break
            if take:
                yield index, frame
            index += 1
    finally:
        capture.release()


def iter_source_frames(
    source: Path,
    recursive: bool = False,
    sampling: FrameSampling | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yield (label, BGR image or None) from an image file, an image folder, or a video file."""
    cv2, _np = load_backend()
    sampling = sampling or FrameSampling()
    if is_video_path(source):
        for index, frame in iter_video_frames(source, sampling):
            yield video_frame_label(source, index), frame
        return
    if source.is_file():
        yield str(source), cv2.imread(str(source))
        return
    paths = iter_image_paths(source, recursive)
    end = len(paths) if sampling.end is None else sampling.end
    for path in paths[sampling.start:end:max(1, int(sampling.every))]:
        yield str(path), cv2.imread(str(path))


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


def transform_points(np, transform: list[list[float]] | Any, points: list[list[float]]) -> Any:
    matrix = np.array(transform, dtype=float)
    homogeneous = np.hstack([np.array(points, dtype=float), np.ones((len(points), 1), dtype=float)])
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def detect_charuco_in_array(
    image,
    settings: BoardSettings,
    min_corners: int,
    label: str = "",
) -> DetectionSummary:
    cv2, _np = load_backend()
    if image is None:
        return DetectionSummary(label, False, 0, 0, "Could not read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    image_size = (int(gray.shape[1]), int(gray.shape[0]))
    board, detector = board_engine(cv2, settings)
    charuco_corners, charuco_ids, _marker_corners, marker_ids = detector.detectBoard(gray)

    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    if marker_count == 0:
        return DetectionSummary(label, False, 0, 0, "No ArUco markers detected", image_size)

    charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
    if charuco_ids is None or charuco_corners is None or charuco_count < max(int(min_corners), 4):
        return DetectionSummary(
            label, False, marker_count, charuco_count, f"Only {charuco_count} ChArUco corners found", image_size
        )
    if board.checkCharucoCornersCollinear(charuco_ids):
        return DetectionSummary(
            label, False, marker_count, charuco_count, "ChArUco corners are collinear", image_size
        )

    return DetectionSummary(
        image_path=label,
        accepted=True,
        marker_count=marker_count,
        charuco_corner_count=charuco_count,
        reason="Accepted",
        image_size=image_size,
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
    )


def detect_charuco_in_image(
    image_path: Path,
    settings: BoardSettings,
    min_corners: int,
) -> DetectionSummary:
    cv2, _np = load_backend()
    return detect_charuco_in_array(cv2.imread(str(image_path)), settings, min_corners, str(image_path))


def board_correspondences(cv2, np, settings: BoardSettings, charuco_corners, charuco_ids):
    board, _detector = board_engine(cv2, settings)
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return (
        np.asarray(object_points, dtype=np.float32).reshape(-1, 1, 3),
        np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
    )


def reprojection_rms(cv2, np, object_points, image_points, rvec, tvec, camera_matrix, distortion_coeffs) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coeffs)
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def solve_board_pose(cv2, np, object_points, image_points, camera_matrix, distortion_coeffs):
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, distortion_coeffs, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return False, None, None
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, distortion_coeffs, rvec, tvec)
    return True, rvec.reshape(3, 1), tvec.reshape(3, 1)


def pose_dict_from_rvec_tvec(cv2, np, rvec, tvec) -> dict[str, Any]:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    inverse = np.linalg.inv(transform)
    return {
        "rvec": [float(value) for value in np.asarray(rvec).reshape(-1)],
        "tvec": [float(value) for value in np.asarray(tvec).reshape(-1)],
        "T_board_to_camera": [[float(value) for value in row] for row in transform.tolist()],
        "T_camera_to_board": [[float(value) for value in row] for row in inverse.tolist()],
    }


def collect_detections(
    source: Path,
    board_settings: BoardSettings,
    min_corners: int,
    recursive: bool = False,
    sampling: FrameSampling | None = None,
    progress: ProgressCallback = None,
) -> tuple[list[DetectionSummary], list[DetectionSummary], tuple[int, int] | None]:
    accepted: list[DetectionSummary] = []
    rejected: list[DetectionSummary] = []
    image_size = None
    for count, (label, image) in enumerate(iter_source_frames(source, recursive, sampling), start=1):
        detection = detect_charuco_in_array(image, board_settings, min_corners, label)
        if detection.image_size and image_size is None:
            image_size = detection.image_size
        (accepted if detection.accepted else rejected).append(detection)
        if progress and count % 50 == 0:
            progress(f"Scanned {count} frames, accepted {len(accepted)}")
    return accepted, rejected, image_size


def board_normal(cv2, np, rvec):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    normal = rotation[:, 2].copy()
    return -normal if normal[2] > 0 else normal


def orientation_statistics(cv2, np, rvecs) -> dict[str, Any]:
    normals = [board_normal(cv2, np, rvec) for rvec in rvecs]
    representatives: list[Any] = []
    cos_limit = math.cos(math.radians(ORIENTATION_SEPARATION_DEG))
    for normal in normals:
        if all(float(normal @ rep) < cos_limit for rep in representatives):
            representatives.append(normal)
    stacked = np.array(normals, dtype=float)
    spread = 0.0
    if len(stacked) > 1:
        cosines = np.clip(stacked @ stacked.T, -1.0, 1.0)
        spread = float(np.degrees(np.arccos(cosines.min())))
    tilt_x = np.degrees(np.arcsin(np.clip(stacked[:, 1], -1.0, 1.0)))
    tilt_y = np.degrees(np.arcsin(np.clip(stacked[:, 0], -1.0, 1.0)))
    return {
        "distinct_orientations": len(representatives),
        "orientation_spread_deg": spread,
        "tilt_about_x_range_deg": float(tilt_x.max() - tilt_x.min()),
        "tilt_about_y_range_deg": float(tilt_y.max() - tilt_y.min()),
        "max_tilt_deg": float(np.degrees(np.arccos(np.clip(-stacked[:, 2], -1.0, 1.0))).max()),
    }


def corner_coverage(np, image_points_list, image_size: tuple[int, int]) -> dict[str, Any]:
    cols, rows = COVERAGE_GRID
    counts = np.zeros((rows, cols), dtype=int)
    width, height = image_size
    for points in image_points_list:
        xy = points.reshape(-1, 2)
        col = np.clip((xy[:, 0] / width * cols).astype(int), 0, cols - 1)
        row = np.clip((xy[:, 1] / height * rows).astype(int), 0, rows - 1)
        np.add.at(counts, (row, col), 1)
    return {
        "grid": [cols, rows],
        "counts": counts.tolist(),
        "fraction": float((counts > 0).mean()),
    }


def format_coverage_map(coverage: dict[str, Any]) -> str:
    lines = []
    for row in coverage["counts"]:
        lines.append(" ".join(f"{value:4d}" if value else "   ." for value in row))
    return "\n".join(lines)


def intrinsics_quality_warnings(
    camera_matrix,
    image_size: tuple[int, int],
    orientation: dict[str, Any],
    coverage: dict[str, Any],
    intrinsics_std: dict[str, float] | None,
    options: IntrinsicsOptions,
) -> list[str]:
    messages: list[str] = []
    width, height = image_size
    fx, fy = float(camera_matrix[0][0]), float(camera_matrix[1][1])
    cx, cy = float(camera_matrix[0][2]), float(camera_matrix[1][2])

    ratio = fx / fy if fy else float("inf")
    if not FX_FY_RATIO_RANGE[0] <= ratio <= FX_FY_RATIO_RANGE[1]:
        messages.append(
            f"fx/fy ratio {ratio:.3f} is outside {FX_FY_RATIO_RANGE[0]}-{FX_FY_RATIO_RANGE[1]}; "
            "the focal length is probably degenerate (enable Fix Aspect Ratio or add tilted views)."
        )
    for name, focal in (("fx", fx), ("fy", fy)):
        low, high = FOCAL_WIDTH_RATIO_RANGE
        if not low * width <= focal <= high * width:
            messages.append(
                f"{name}={focal:.1f} px is implausible for a {width} px wide image "
                f"(expected {low * width:.0f}-{high * width:.0f} px)."
            )
    if abs(cx - (width - 1) / 2.0) > PRINCIPAL_POINT_MAX_OFFSET * width or abs(
        cy - (height - 1) / 2.0
    ) > PRINCIPAL_POINT_MAX_OFFSET * height:
        messages.append(
            f"Principal point ({cx:.1f}, {cy:.1f}) is more than {PRINCIPAL_POINT_MAX_OFFSET:.0%} of the image "
            f"size away from the centre ({(width - 1) / 2:.1f}, {(height - 1) / 2:.1f})."
        )
    if orientation["distinct_orientations"] < MIN_DISTINCT_ORIENTATIONS:
        messages.append(
            f"Only {orientation['distinct_orientations']} distinct board orientation(s) "
            f"(>{ORIENTATION_SEPARATION_DEG:.0f} deg apart) were used; need at least {MIN_DISTINCT_ORIENTATIONS}. "
            "Tilt the board towards and away from the camera in different directions."
        )
    for axis, focal_name in (("x", "fy"), ("y", "fx")):
        tilt_range = orientation[f"tilt_about_{axis}_range_deg"]
        if tilt_range < MIN_AXIS_TILT_RANGE_DEG and not options.fix_aspect_ratio:
            messages.append(
                f"Board tilt about the image {axis}-axis varies by only {tilt_range:.1f} deg; "
                f"{focal_name} is poorly constrained."
            )
    if coverage["fraction"] < MIN_COVERAGE_FRACTION:
        messages.append(
            f"Detected corners cover only {coverage['fraction']:.0%} of the image grid; "
            "distortion is extrapolated in the empty regions."
        )
    if intrinsics_std:
        for name, focal in (("fx", fx), ("fy", fy)):
            std = intrinsics_std.get(name)
            if std is not None and focal and std / focal > MAX_RELATIVE_FOCAL_STD:
                messages.append(f"{name} standard deviation is {std / focal:.1%} of its value.")
    return messages


def calibration_flags(cv2, options: IntrinsicsOptions) -> int:
    flags = 0
    if options.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if options.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if options.zero_tangential:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if options.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    return flags


def initial_camera_matrix(np, image_size: tuple[int, int], focal: float | None = None):
    width, height = image_size
    focal = float(focal or width)
    return np.array([[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def rough_camera_matrix(cv2, np, object_points, image_points, image_size: tuple[int, int]):
    width = image_size[0]
    guess = initial_camera_matrix(np, image_size)
    subset = sorted(set(np.linspace(0, len(object_points) - 1, min(len(object_points), 40)).astype(int).tolist()))
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    try:
        _rms, camera_matrix, *_ = cv2.calibrateCamera(
            [object_points[index] for index in subset],
            [image_points[index] for index in subset],
            image_size,
            guess.copy(),
            np.zeros(5),
            flags=flags,
        )
    except cv2.error:
        return guess
    focal = float(camera_matrix[0, 0])
    low, high = FOCAL_WIDTH_RATIO_RANGE
    if not math.isfinite(focal) or not low * width <= focal <= high * width:
        return guess
    return camera_matrix


def select_diverse_frames(
    object_points,
    image_points,
    image_size: tuple[int, int],
    max_frames: int,
) -> list[int]:
    """Greedy farthest-point selection on board normal, image position and distance."""
    cv2, np = load_backend()
    count = len(object_points)
    if max_frames <= 0 or count <= max_frames:
        return list(range(count))

    camera_matrix = rough_camera_matrix(cv2, np, object_points, image_points, image_size)
    width, height = image_size
    features = []
    corner_counts = []
    for obj, img in zip(object_points, image_points):
        ok, rvec, tvec = solve_board_pose(cv2, np, obj, img, camera_matrix, None)
        normal = board_normal(cv2, np, rvec) if ok else np.array([0.0, 0.0, -1.0])
        depth = float(tvec[2, 0]) if ok and float(tvec[2, 0]) > 0 else 1.0
        centre = img.reshape(-1, 2).mean(axis=0) / np.array([width, height])
        features.append(
            np.concatenate(
                [
                    normal / math.radians(ORIENTATION_SEPARATION_DEG),
                    centre / 0.2,
                    [math.log(depth) / 0.25],
                ]
            )
        )
        corner_counts.append(len(obj))
    features = np.array(features)
    weights = np.sqrt(np.array(corner_counts, dtype=float) / max(corner_counts))

    selected = [int(np.argmax(corner_counts))]
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)
    while len(selected) < max_frames:
        score = min_dist * weights
        score[selected] = -1.0
        best = int(np.argmax(score))
        selected.append(best)
        min_dist = np.minimum(min_dist, np.linalg.norm(features - features[best], axis=1))
    return sorted(selected)


def calibrate_from_detections(
    detections: list[DetectionSummary],
    image_size: tuple[int, int],
    board_settings: BoardSettings,
    options: IntrinsicsOptions | None = None,
) -> dict[str, Any]:
    cv2, np = load_backend()
    options = options or IntrinsicsOptions()
    if len(detections) < 3:
        raise RuntimeError(f"Need at least 3 accepted ChArUco frames for intrinsics calibration. Got {len(detections)}.")

    object_points, image_points = [], []
    for item in detections:
        obj, img = board_correspondences(cv2, np, board_settings, item.charuco_corners, item.charuco_ids)
        object_points.append(obj)
        image_points.append(img)

    selected = select_diverse_frames(object_points, image_points, image_size, options.max_frames)
    sel_obj = [object_points[index] for index in selected]
    sel_img = [image_points[index] for index in selected]

    result = _solve_intrinsics(cv2, np, sel_obj, sel_img, image_size, options)
    result["auto_constrained"] = False
    degenerate = result["orientation"]["distinct_orientations"] < MIN_DISTINCT_ORIENTATIONS
    if options.auto_constrain and degenerate and not (options.fix_aspect_ratio and options.fix_principal_point):
        constrained = replace(options, fix_aspect_ratio=True, fix_principal_point=True, zero_tangential=True)
        unconstrained_matrix = result["camera_matrix"]
        result = _solve_intrinsics(cv2, np, sel_obj, sel_img, image_size, constrained)
        result["auto_constrained"] = True
        result["unconstrained_camera_matrix"] = unconstrained_matrix
        result["quality_warnings"].insert(
            0,
            "Degenerate view set: re-solved with fixed aspect ratio, centred principal point and zero tangential "
            f"distortion (unconstrained fx={unconstrained_matrix[0][0]:.1f}, fy={unconstrained_matrix[1][1]:.1f}).",
        )
    result["selected"] = selected
    result["frame_errors"] = {selected[index]: value for index, value in enumerate(result.pop("per_view_errors"))}
    return result


def _solve_intrinsics(cv2, np, object_points, image_points, image_size, options: IntrinsicsOptions) -> dict[str, Any]:
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    (
        reprojection_error,
        camera_matrix,
        distortion_coeffs,
        rvecs,
        tvecs,
        std_intrinsics,
        _std_extrinsics,
        per_view_errors,
    ) = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        initial_camera_matrix(np, image_size, focal=1.0),
        np.zeros(5),
        flags=calibration_flags(cv2, options),
        criteria=criteria,
    )

    std_names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    intrinsics_std = {
        name: float(value) for name, value in zip(std_names, np.asarray(std_intrinsics, dtype=float).reshape(-1))
    }
    orientation = orientation_statistics(cv2, np, rvecs)
    coverage = corner_coverage(np, image_points, image_size)
    return {
        "reprojection_error": float(reprojection_error),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion_coeffs.reshape(-1).tolist(),
        "per_view_errors": [float(value) for value in np.asarray(per_view_errors).reshape(-1)],
        "rvecs": rvecs,
        "tvecs": tvecs,
        "intrinsics_std": intrinsics_std,
        "orientation": orientation,
        "coverage": coverage,
        "options": options,
        "quality_warnings": intrinsics_quality_warnings(
            camera_matrix, image_size, orientation, coverage, intrinsics_std, options
        ),
    }


def calibrate_intrinsics(
    image_folder: Path,
    camera_name: str,
    board_settings: BoardSettings,
    min_corners: int,
    recursive: bool,
    options: IntrinsicsOptions | None = None,
    sampling: FrameSampling | None = None,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    cv2, _np = load_backend()
    options = options or IntrinsicsOptions()
    source = Path(image_folder)
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    accepted, rejected, image_size = collect_detections(
        source, board_settings, min_corners, recursive, sampling, progress
    )
    total = len(accepted) + len(rejected)
    if total == 0:
        raise RuntimeError(f"No supported image files or video frames found in {source}")
    if len(accepted) < 3:
        raise RuntimeError(
            "Need at least 3 accepted ChArUco frames for intrinsics calibration.\n"
            f"Accepted: {len(accepted)} / {total}"
        )

    result = calibrate_from_detections(accepted, image_size, board_settings, options)
    selected = result["selected"]
    selected_set = set(selected)
    return {
        "camera_name": camera_name,
        "mode": "intrinsics",
        "board": board_payload(board_settings),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "reprojection_error": result["reprojection_error"],
        "camera_matrix": result["camera_matrix"],
        "distortion_coefficients": result["distortion_coefficients"],
        "accepted_frame_count": len(accepted),
        "rejected_frame_count": len(rejected),
        "accepted_frames": [
            {
                "image_path": item.image_path,
                "marker_count": item.marker_count,
                "charuco_corner_count": item.charuco_corner_count,
                "used_for_calibration": index in selected_set,
                **(
                    {"reprojection_error": result["frame_errors"][index]}
                    if index in selected_set
                    else {}
                ),
            }
            for index, item in enumerate(accepted)
        ],
        "rejected_frames": [
            {
                "image_path": item.image_path,
                "reason": item.reason,
                "marker_count": item.marker_count,
                "charuco_corner_count": item.charuco_corner_count,
            }
            for item in rejected
        ],
        "view_poses": [
            {
                "image_path": accepted[frame_index].image_path,
                "rvec": [float(value) for value in rvec.reshape(-1)],
                "tvec": [float(value) for value in tvec.reshape(-1)],
                "reprojection_error": result["frame_errors"][frame_index],
            }
            for frame_index, rvec, tvec in zip(selected, result["rvecs"], result["tvecs"])
        ],
        "used_frame_count": len(selected),
        "frame_selection": {
            "method": "pose_diversity" if len(selected) < len(accepted) else "all",
            "max_frames": options.max_frames,
            "candidates": len(accepted),
            "selected": len(selected),
        },
        "calibration_options": asdict(options),
        "effective_calibration_options": asdict(result["options"]),
        "auto_constrained": result["auto_constrained"],
        **(
            {"unconstrained_camera_matrix": result["unconstrained_camera_matrix"]}
            if result["auto_constrained"]
            else {}
        ),
        "intrinsics_std": result["intrinsics_std"],
        "orientation_diversity": result["orientation"],
        "coverage": result["coverage"],
        "quality_warnings": result["quality_warnings"],
        "opencv_version": cv2.__version__,
    }


def intrinsics_arrays(np, intrinsics_payload: dict[str, Any]):
    camera_matrix = np.array(intrinsics_payload["camera_matrix"], dtype=float)
    distortion_coeffs = np.array(intrinsics_payload.get("distortion_coefficients", []), dtype=float).reshape(-1, 1)
    if distortion_coeffs.size == 0:
        distortion_coeffs = np.zeros((5, 1), dtype=float)
    return camera_matrix, distortion_coeffs


def estimate_pose_from_detection(
    detection: DetectionSummary,
    intrinsics_payload: dict[str, Any],
    board_settings: BoardSettings,
) -> dict[str, Any]:
    cv2, np = load_backend()
    camera_matrix, distortion_coeffs = intrinsics_arrays(np, intrinsics_payload)
    obj, img = board_correspondences(cv2, np, board_settings, detection.charuco_corners, detection.charuco_ids)
    ok, rvec, tvec = solve_board_pose(cv2, np, obj, img, camera_matrix, distortion_coeffs)
    if not ok:
        raise RuntimeError(f"{detection.image_path}: pose estimation failed")

    payload = pose_dict_from_rvec_tvec(cv2, np, rvec, tvec)
    payload.update(
        {
            "image_path": detection.image_path,
            "marker_count": detection.marker_count,
            "charuco_corner_count": detection.charuco_corner_count,
            "reprojection_error": reprojection_rms(cv2, np, obj, img, rvec, tvec, camera_matrix, distortion_coeffs),
        }
    )
    return payload


def estimate_pose_from_image(
    image_path: Path,
    intrinsics_payload: dict[str, Any],
    board_settings: BoardSettings,
    min_corners: int,
) -> dict[str, Any]:
    detection = detect_charuco_in_image(image_path, board_settings, min_corners)
    if not detection.accepted:
        raise RuntimeError(f"{Path(image_path).name}: {detection.reason}")
    return estimate_pose_from_detection(detection, intrinsics_payload, board_settings)


def estimate_extrinsics_for_path(
    input_path: Path,
    camera_name: str,
    intrinsics_payload: dict[str, Any],
    board_settings: BoardSettings,
    min_corners: int,
    recursive: bool,
    sampling: FrameSampling | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    if input_path.is_file() and not is_video_path(input_path):
        return {
            "camera_name": camera_name,
            "mode": "extrinsics_single",
            "intrinsics_source": intrinsics_payload.get("camera_name"),
            "board": board_payload(board_settings),
            "pose": estimate_pose_from_image(input_path, intrinsics_payload, board_settings, min_corners),
        }

    poses = []
    failures = []
    for label, image in iter_source_frames(input_path, recursive, sampling):
        detection = detect_charuco_in_array(image, board_settings, min_corners, label)
        if not detection.accepted:
            failures.append({"image_path": label, "reason": f"{Path(label).name}: {detection.reason}"})
            continue
        try:
            poses.append(estimate_pose_from_detection(detection, intrinsics_payload, board_settings))
        except Exception as exc:
            failures.append({"image_path": label, "reason": str(exc)})

    if not poses and not failures:
        raise RuntimeError(f"No supported image files or video frames found in {input_path}")
    if not poses:
        raise RuntimeError("No valid ChArUco poses could be estimated from the selected images.")

    return {
        "camera_name": camera_name,
        "mode": "extrinsics_batch",
        "intrinsics_source": intrinsics_payload.get("camera_name"),
        "board": board_payload(board_settings),
        "estimated_pose_count": len(poses),
        "failed_pose_count": len(failures),
        "poses": poses,
        "failed_images": failures,
    }


def rt_to_matrix(cv2, np, rvec, tvec):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return transform


def matrix_to_rt(cv2, np, transform):
    rvec = cv2.Rodrigues(np.asarray(transform, dtype=float)[:3, :3])[0].reshape(3)
    return rvec, np.asarray(transform, dtype=float)[:3, 3].copy()


def rotation_angle_deg(np, rotation_a, rotation_b) -> float:
    cos_value = (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def robust_mean_transform(np, transforms: list[Any]):
    """Chordal rotation mean and translation mean after rejecting outliers around the median."""

    def chordal_mean(rotations):
        u, _s, vt = np.linalg.svd(np.sum(rotations, axis=0))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return rotation

    stack = np.array(transforms, dtype=float)
    if len(stack) == 1:
        return stack[0].copy(), 1
    rotations, translations = stack[:, :3, :3], stack[:, :3, 3]
    rotation = chordal_mean(rotations)
    translation = np.median(translations, axis=0)
    angle = np.array([rotation_angle_deg(np, rotation, item) for item in rotations])
    offset = np.linalg.norm(translations - translation, axis=1)
    keep = (angle <= max(1.0, 3.0 * np.median(angle))) & (offset <= max(1e-9, 3.0 * np.median(offset)))
    if keep.sum() < max(1, len(stack) // 2):
        keep = angle <= np.median(angle)
    result = np.eye(4)
    result[:3, :3] = chordal_mean(rotations[keep])
    result[:3, 3] = translations[keep].mean(axis=0)
    return result, int(keep.sum())


def _source_kind(path: Path) -> str:
    if is_video_path(path):
        return "video"
    if path.is_dir():
        return "folder"
    if path.is_file():
        return "image"
    raise FileNotFoundError(f"Input does not exist: {path}")


def _camera_frame_sources(entries: list[dict[str, Any]], recursive: bool, sampling: FrameSampling):
    """Return per camera a list of (frame_key, label, loader) aligned across cameras by frame key."""
    cv2, _np = load_backend()
    kinds = {_source_kind(Path(entry["image_path"])) for entry in entries}
    if len(kinds) > 1 and kinds != {"image"}:
        raise RuntimeError(
            "All cameras must use the same input type (one image each, image folders, or videos). "
            f"Got: {', '.join(sorted(kinds))}"
        )
    kind = kinds.pop()
    if kind == "image":
        return kind, [[("0", entry["image_path"], None)] for entry in entries]
    if kind == "video":
        return kind, [None for _entry in entries]

    listings = [iter_image_paths(Path(entry["image_path"]), recursive) for entry in entries]
    stems = [{path.stem for path in paths} for paths in listings]
    counts: dict[str, int] = {}
    for camera_stems in stems:
        for stem in camera_stems:
            counts[stem] = counts.get(stem, 0) + 1
    shared = {stem for stem, count in counts.items() if count >= 2}
    by_name = len(shared) > 0
    shared_keys = sorted(shared, key=_frame_sort_key)
    per_camera = []
    for paths in listings:
        end = (len(shared_keys) if by_name else len(paths)) if sampling.end is None else sampling.end
        rows = []
        for position, path in enumerate(paths):
            key = path.stem if by_name else f"{position:06d}"
            rows.append((key, str(path), path))
        if by_name:
            allowed = set(shared_keys[sampling.start:end:max(1, sampling.every)])
            rows = [row for row in rows if row[0] in allowed]
        else:
            rows = rows[sampling.start:end:max(1, sampling.every)]
        per_camera.append(rows)
    return kind, per_camera


def _detect_camera_observations(
    entry: dict[str, Any],
    rows,
    kind: str,
    board_settings: BoardSettings,
    min_corners: int,
    sampling: FrameSampling,
    camera_matrix,
    distortion_coeffs,
    progress: ProgressCallback,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    cv2, np = load_backend()
    observations: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    if kind == "video":
        source = Path(entry["image_path"])
        frames = ((str(index), video_frame_label(source, index), image) for index, image in iter_video_frames(source, sampling))
    else:
        frames = ((key, label, cv2.imread(label)) for key, label, _path in rows)

    for count, (key, label, image) in enumerate(frames, start=1):
        detection = detect_charuco_in_array(image, board_settings, min_corners, label)
        if progress and count % 100 == 0:
            progress(f"{entry['camera_name']}: scanned {count} frames, {len(observations)} board poses")
        if not detection.accepted:
            failures.append({"image_path": label, "reason": detection.reason})
            continue
        obj, img = board_correspondences(cv2, np, board_settings, detection.charuco_corners, detection.charuco_ids)
        ok, rvec, tvec = solve_board_pose(cv2, np, obj, img, camera_matrix, distortion_coeffs)
        if not ok:
            failures.append({"image_path": label, "reason": "pose estimation failed"})
            continue
        observations[key] = {
            "label": label,
            "object_points": obj,
            "image_points": img,
            "T_board_to_camera": rt_to_matrix(cv2, np, rvec, tvec),
            "marker_count": detection.marker_count,
            "charuco_corner_count": detection.charuco_corner_count,
            "reprojection_error": reprojection_rms(cv2, np, obj, img, rvec, tvec, camera_matrix, distortion_coeffs),
        }
    return observations, failures


def _frame_sort_key(key: str):
    return (0, int(key), "") if key.isdigit() else (1, 0, key)


def select_frame_sets(observations: list[dict[str, dict[str, Any]]], max_frame_sets: int) -> list[str]:
    """Pick frame sets seen by >= 2 cameras, balancing per-camera coverage and spreading over time."""
    camera_count = len(observations)
    keys = sorted({key for obs in observations for key in obs}, key=_frame_sort_key)
    seen_by = {key: [index for index in range(camera_count) if key in observations[index]] for key in keys}
    candidates = [key for key in keys if len(seen_by[key]) >= 2]
    if max_frame_sets <= 0 or len(candidates) <= max_frame_sets:
        return candidates

    position = {key: index for index, key in enumerate(candidates)}
    selected: list[str] = []
    per_camera = [0] * camera_count
    remaining = set(candidates)
    while remaining and len(selected) < max_frame_sets:
        def score(key):
            coverage = sum(1.0 / (1 + per_camera[index]) for index in seen_by[key])
            gap = min((abs(position[key] - position[other]) for other in selected), default=len(candidates))
            return (round(coverage, 6), gap)

        best = max(remaining, key=score)
        selected.append(best)
        remaining.discard(best)
        for index in seen_by[best]:
            per_camera[index] += 1
    return sorted(selected, key=_frame_sort_key)


def _initial_camera_poses(np, observations, frame_keys, reference_index):
    """Maximum spanning tree over co-observation counts; relative poses by robust averaging."""
    camera_count = len(observations)
    shared = [[[key for key in frame_keys if key in observations[a] and key in observations[b]] for b in range(camera_count)] for a in range(camera_count)]
    reference_to_camera = {reference_index: np.eye(4)}
    pair_info = {}
    while len(reference_to_camera) < camera_count:
        best = None
        for parent in reference_to_camera:
            for child in range(camera_count):
                if child in reference_to_camera or not shared[parent][child]:
                    continue
                if best is None or len(shared[parent][child]) > len(shared[best[0]][best[1]]):
                    best = (parent, child)
        if best is None:
            break
        parent, child = best
        relative = [
            observations[child][key]["T_board_to_camera"] @ np.linalg.inv(observations[parent][key]["T_board_to_camera"])
            for key in shared[parent][child]
        ]
        parent_to_child, inliers = robust_mean_transform(np, relative)
        reference_to_camera[child] = parent_to_child @ reference_to_camera[parent]
        pair_info[child] = {"parent": parent, "shared_frames": len(relative), "inliers": inliers}
    return reference_to_camera, pair_info


def _initial_board_poses(np, observations, frame_keys, reference_to_camera):
    board_to_reference = {}
    for key in frame_keys:
        candidates = [
            np.linalg.inv(reference_to_camera[index]) @ observations[index][key]["T_board_to_camera"]
            for index in reference_to_camera
            if key in observations[index]
        ]
        if candidates:
            board_to_reference[key], _ = robust_mean_transform(np, candidates)
    return board_to_reference


def _residual_blocks(cv2, np, observations, intrinsics, reference_to_camera, board_to_reference):
    per_camera = {}
    for index, transform in reference_to_camera.items():
        errors = []
        for key, board_pose in board_to_reference.items():
            item = observations[index].get(key)
            if item is None:
                continue
            board_to_camera = transform @ board_pose
            rvec, tvec = matrix_to_rt(cv2, np, board_to_camera)
            projected, _ = cv2.projectPoints(item["object_points"], rvec, tvec, *intrinsics[index])
            errors.append(projected.reshape(-1, 2) - item["image_points"].reshape(-1, 2))
        per_camera[index] = np.concatenate(errors) if errors else np.zeros((0, 2))
    return per_camera


def _rms(np, residuals) -> float | None:
    if residuals is None or len(residuals) == 0:
        return None
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def _bundle_adjust(cv2, np, observations, intrinsics, reference_index, reference_to_camera, board_to_reference):
    from scipy.optimize import least_squares  # type: ignore
    from scipy.sparse import lil_matrix  # type: ignore

    cameras = [index for index in sorted(reference_to_camera) if index != reference_index]
    frames = sorted(board_to_reference, key=_frame_sort_key)
    camera_slot = {index: slot for slot, index in enumerate(cameras)}
    frame_slot = {key: len(cameras) + slot for slot, key in enumerate(frames)}

    x0 = []
    for index in cameras:
        x0.extend(np.concatenate(matrix_to_rt(cv2, np, reference_to_camera[index])))
    for key in frames:
        x0.extend(np.concatenate(matrix_to_rt(cv2, np, board_to_reference[key])))
    x0 = np.array(x0, dtype=float)

    blocks = []
    for index in sorted(reference_to_camera):
        for key in frames:
            item = observations[index].get(key)
            if item is not None:
                blocks.append((index, key, item))

    rows = sum(2 * len(item["object_points"]) for _index, _key, item in blocks)
    sparsity = lil_matrix((rows, len(x0)), dtype=int)
    row = 0
    for index, key, item in blocks:
        size = 2 * len(item["object_points"])
        if index in camera_slot:
            start = 6 * camera_slot[index]
            sparsity[row:row + size, start:start + 6] = 1
        start = 6 * frame_slot[key]
        sparsity[row:row + size, start:start + 6] = 1
        row += size

    def residuals(params):
        out = []
        for index, key, item in blocks:
            frame_params = params[6 * frame_slot[key]:6 * frame_slot[key] + 6]
            rvec, tvec = frame_params[:3], frame_params[3:]
            if index in camera_slot:
                cam_params = params[6 * camera_slot[index]:6 * camera_slot[index] + 6]
                rvec, tvec = cv2.composeRT(rvec, tvec, cam_params[:3], cam_params[3:])[:2]
            projected, _ = cv2.projectPoints(item["object_points"], rvec, tvec, *intrinsics[index])
            out.append((projected.reshape(-1, 2) - item["image_points"].reshape(-1, 2)).reshape(-1))
        return np.concatenate(out)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = least_squares(
            residuals, x0, jac_sparsity=sparsity, loss="huber", f_scale=1.0, method="trf", max_nfev=200
        )
    refined_cameras = {reference_index: np.eye(4)}
    for index in cameras:
        params = result.x[6 * camera_slot[index]:6 * camera_slot[index] + 6]
        refined_cameras[index] = rt_to_matrix(cv2, np, params[:3], params[3:])
    refined_frames = {}
    for key in frames:
        params = result.x[6 * frame_slot[key]:6 * frame_slot[key] + 6]
        refined_frames[key] = rt_to_matrix(cv2, np, params[:3], params[3:])
    return refined_cameras, refined_frames


def scipy_available() -> bool:
    try:
        import scipy.optimize  # noqa: F401  # type: ignore
    except Exception:
        return False
    return True


def estimate_multi_camera_extrinsics(
    camera_entries: list[dict[str, str]],
    board_settings: BoardSettings,
    min_corners: int,
    reference_camera: str | None,
    sampling: FrameSampling | None = None,
    max_frame_sets: int = DEFAULT_MAX_FRAME_SETS,
    bundle_adjust: bool = True,
    recursive: bool = False,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    """Relative camera poses from one or more synchronized frame sets.

    Each entry's image_path may be a single image (one frame set, as before), a folder of images
    (frame sets matched by file name, or by sorted order) or a video (frame sets matched by frame index).
    """
    cv2, np = load_backend()
    sampling = sampling or FrameSampling()
    active_entries = [entry for entry in camera_entries if entry["camera_name"].strip()]
    if len(active_entries) < 2:
        raise RuntimeError("Provide at least two cameras for multi-camera extrinsics.")

    names = [entry["camera_name"].strip() for entry in active_entries]
    if reference_camera:
        if reference_camera not in names:
            raise RuntimeError(f"Reference camera not found: {reference_camera}")
        reference_index = names.index(reference_camera)
    else:
        reference_index = 0

    intrinsics = []
    for entry in active_entries:
        camera_matrix, distortion_coeffs = intrinsics_arrays(np, read_json(Path(entry["intrinsics_path"])))
        intrinsics.append((camera_matrix, distortion_coeffs))

    kind, frame_rows = _camera_frame_sources(active_entries, recursive, sampling)
    observations = []
    failures = []
    for index, entry in enumerate(active_entries):
        obs, fails = _detect_camera_observations(
            entry, frame_rows[index], kind, board_settings, min_corners, sampling, *intrinsics[index], progress
        )
        if kind == "image" and not obs:
            raise RuntimeError(f"{Path(entry['image_path']).name}: {fails[0]['reason'] if fails else 'no board'}")
        observations.append(obs)
        failures.append(fails)

    frame_keys = select_frame_sets(observations, max_frame_sets)
    if not frame_keys:
        raise RuntimeError("No frame set shows the board in at least two cameras.")

    reference_to_camera, pair_info = _initial_camera_poses(np, observations, frame_keys, reference_index)
    missing = [names[index] for index in range(len(names)) if index not in reference_to_camera]
    if missing:
        raise RuntimeError(
            "These cameras never see the board together with a connected camera: " + ", ".join(missing)
        )
    board_to_reference = _initial_board_poses(np, observations, frame_keys, reference_to_camera)
    initial_residuals = _residual_blocks(cv2, np, observations, intrinsics, reference_to_camera, board_to_reference)

    messages: list[str] = []
    refinement = "single_frame" if len(frame_keys) == 1 else "robust_average"
    if len(frame_keys) > 1 and bundle_adjust:
        if scipy_available():
            reference_to_camera, board_to_reference = _bundle_adjust(
                cv2, np, observations, intrinsics, reference_index, reference_to_camera, board_to_reference
            )
            refinement = "bundle_adjustment"
        else:
            messages.append("scipy is not installed; used robust averaging without bundle adjustment.")
    final_residuals = _residual_blocks(cv2, np, observations, intrinsics, reference_to_camera, board_to_reference)

    reference_name = names[reference_index]
    per_camera = []
    for index, entry in enumerate(active_entries):
        keys = [key for key in frame_keys if key in observations[index]]
        first = observations[index][keys[0]]
        board_to_camera = reference_to_camera[index] @ board_to_reference[keys[0]]
        rvec, tvec = matrix_to_rt(cv2, np, board_to_camera)
        camera_to_reference = np.linalg.inv(reference_to_camera[index])
        item = {
            "camera_name": names[index],
            "image_path": first["label"],
            "intrinsics_path": entry["intrinsics_path"],
            **pose_dict_from_rvec_tvec(cv2, np, rvec, tvec),
            "marker_count": first["marker_count"],
            "charuco_corner_count": first["charuco_corner_count"],
            "T_camera_to_reference": camera_to_reference.tolist(),
            "T_reference_to_camera": reference_to_camera[index].tolist(),
            "reference_camera": reference_name,
            "source_path": entry["image_path"],
            "frame_count": len(keys),
            "detected_frame_count": len(observations[index]),
            "failed_frame_count": len(failures[index]),
            "reprojection_error_px": _rms(np, final_residuals[index]),
            "reprojection_error_px_initial": _rms(np, initial_residuals[index]),
            "frames": [
                {
                    "frame": key,
                    "image_path": observations[index][key]["label"],
                    "charuco_corner_count": observations[index][key]["charuco_corner_count"],
                    "single_view_reprojection_error_px": observations[index][key]["reprojection_error"],
                }
                for key in keys
            ],
        }
        if index in pair_info:
            info = pair_info[index]
            item["initialized_from"] = {
                "camera_name": names[info["parent"]],
                "shared_frames": info["shared_frames"],
                "inlier_frames": info["inliers"],
            }
        per_camera.append(item)

    all_final = np.concatenate([value for value in final_residuals.values() if len(value)])
    return {
        "mode": "multi_camera_extrinsics",
        "reference_camera": reference_name,
        "board": board_payload(board_settings),
        "cameras": per_camera,
        "input_type": kind,
        "frame_set_count": len(frame_keys),
        "frame_sets": frame_keys,
        "refinement": refinement,
        "reprojection_error_px": _rms(np, all_final),
        "warnings": messages,
        "opencv_version": cv2.__version__,
    }


def render_board_view(
    board_settings: BoardSettings,
    camera_matrix,
    rvec,
    tvec,
    image_size: tuple[int, int],
    distortion_coeffs=None,
    pixels_per_square: int = 120,
    supersample: int = 2,
    background: int = 96,
):
    """Render a synthetic grayscale view of the board seen by a pinhole camera (optionally distorted)."""
    cv2, np = load_backend()
    board, _detector = board_engine(cv2, board_settings)
    margin = pixels_per_square // 2
    board_image = board.generateImage(
        (board_settings.squares_x * pixels_per_square + 2 * margin, board_settings.squares_y * pixels_per_square + 2 * margin),
        marginSize=margin,
        borderBits=1,
    )
    scale = float(board_settings.square_length) / pixels_per_square
    pixel_to_board = np.array([[scale, 0, (0.5 - margin) * scale], [0, scale, (0.5 - margin) * scale], [0, 0, 1.0]])
    rotation = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    homography = np.asarray(camera_matrix, dtype=float) @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], np.asarray(tvec, dtype=float).reshape(3)]
    ) @ pixel_to_board

    width, height = image_size
    xs = (np.arange(width * supersample, dtype=np.float64) + 0.5) / supersample - 0.5
    ys = (np.arange(height * supersample, dtype=np.float64) + 0.5) / supersample - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    if distortion_coeffs is not None and np.any(np.asarray(distortion_coeffs) != 0):
        step = 8
        fine_w, fine_h = width * supersample, height * supersample
        coarse_w, coarse_h = -(-fine_w // step), -(-fine_h // step)
        coarse_x, coarse_y = np.meshgrid(
            (np.arange(coarse_w) + 0.5) * step / supersample - 0.5,
            (np.arange(coarse_h) + 0.5) * step / supersample - 0.5,
        )
        coarse = np.stack([coarse_x.ravel(), coarse_y.ravel()], axis=1).reshape(-1, 1, 2)
        matrix = np.asarray(camera_matrix, dtype=float)
        coeffs = np.asarray(distortion_coeffs, dtype=float)
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 1e-8)
        if hasattr(cv2, "undistortPointsIter"):
            ideal = cv2.undistortPointsIter(coarse, matrix, coeffs, None, matrix, criteria)
        else:
            ideal = cv2.undistortPoints(coarse, matrix, coeffs, R=None, P=matrix, criteria=criteria)
        delta = (ideal - coarse).reshape(coarse_h, coarse_w, 2).astype(np.float32)
        delta = cv2.resize(delta, (coarse_w * step, coarse_h * step), interpolation=cv2.INTER_LINEAR)
        pixels = pixels + delta[:fine_h, :fine_w].reshape(-1, 2)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(homography).T
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = homogeneous[:, 0] / homogeneous[:, 2]
        map_y = homogeneous[:, 1] / homogeneous[:, 2]
    behind = homogeneous[:, 2] <= 0
    map_x[behind] = -1e6
    map_y[behind] = -1e6
    shape = (height * supersample, width * supersample)
    rendered = cv2.remap(
        board_image,
        map_x.reshape(shape).astype(np.float32),
        map_y.reshape(shape).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    if supersample > 1:
        rendered = cv2.resize(rendered, (width, height), interpolation=cv2.INTER_AREA)
    return rendered


def run_self_check(verbose: bool = True) -> dict[str, Any]:
    """Render a synthetic board with a known camera and pose, then detect it and recover the pose."""
    cv2, np = load_backend()
    settings = BoardSettings(8, 6, 95.0, 71.0, "DICT_4X4_100", legacy_pattern=True)
    camera_matrix = np.array([[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]])
    rvec = np.array([0.35, -0.25, 0.1])
    tvec = np.array([-330.0, -260.0, 1500.0])
    image = render_board_view(settings, camera_matrix, rvec, tvec, (1280, 720))
    detection = detect_charuco_in_array(image, settings, 12, "synthetic")
    if not detection.accepted:
        raise RuntimeError(f"Self check failed: {detection.reason}")
    pose = estimate_pose_from_detection(
        detection, {"camera_matrix": camera_matrix.tolist(), "distortion_coefficients": [0.0] * 5}, settings
    )
    truth = rt_to_matrix(cv2, np, rvec, tvec)
    estimate = np.array(pose["T_board_to_camera"])
    rotation_error = rotation_angle_deg(np, truth[:3, :3], estimate[:3, :3])
    translation_error = float(np.linalg.norm(truth[:3, 3] - estimate[:3, 3]))
    expected = (settings.squares_x - 1) * (settings.squares_y - 1)
    result = {
        "opencv_version": cv2.__version__,
        "charuco_corners": detection.charuco_corner_count,
        "expected_corners": expected,
        "rotation_error_deg": rotation_error,
        "translation_error": translation_error,
        "reprojection_error_px": pose["reprojection_error"],
    }
    if detection.charuco_corner_count < expected or rotation_error > 0.5 or translation_error > 0.01 * tvec[2]:
        raise RuntimeError(f"Self check failed: {result}")
    if verbose:
        print(
            f"Self check OK (OpenCV {cv2.__version__}): detected {detection.charuco_corner_count}/{expected} corners, "
            f"pose error {rotation_error:.3f} deg / {translation_error:.2f} units, "
            f"reprojection {pose['reprojection_error']:.3f} px"
        )
    return result
