"""Command-line interface mirroring the GUI tabs. Runs headless: tkinter is only imported for the GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core import (
    ARUCO_DICTIONARIES,
    DEFAULT_MAX_FRAME_SETS,
    DEFAULT_MAX_INTRINSICS_FRAMES,
    BoardSettings,
    FrameSampling,
    IntrinsicsOptions,
    calibrate_intrinsics,
    estimate_extrinsics_for_path,
    estimate_multi_camera_extrinsics,
    format_coverage_map,
    read_json,
    run_self_check,
    sanitize_name,
    write_json,
)


APP_TITLE = "ChArUco Intrinsics / Extrinsics Calibrator"
CLI_OUTPUT_ROOT = Path("charuco_calibration_output")


def add_board_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("board")
    group.add_argument("--squares-x", type=int, default=8)
    group.add_argument("--squares-y", type=int, default=6)
    group.add_argument("--square-length", type=float, default=95.0, help="Same unit as --marker-length.")
    group.add_argument("--marker-length", type=float, default=71.0)
    group.add_argument("--dictionary", default="DICT_4X4_100", choices=ARUCO_DICTIONARIES)
    group.add_argument("--legacy-pattern", action=argparse.BooleanOptionalAction, default=True)


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("frame sampling")
    group.add_argument("--every", type=int, default=1, help="Use every Nth video frame or image.")
    group.add_argument("--start", type=int, default=0, help="First video frame (or image index).")
    group.add_argument("--end", type=int, default=None, help="Stop before this video frame (or image index).")


def board_from_args(args: argparse.Namespace) -> BoardSettings:
    settings = BoardSettings(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary_name=args.dictionary,
        legacy_pattern=args.legacy_pattern,
    )
    if settings.squares_x < 2 or settings.squares_y < 2:
        raise ValueError("--squares-x and --squares-y must both be at least 2.")
    if settings.square_length <= 0 or settings.marker_length <= 0:
        raise ValueError("Board lengths must be positive.")
    if settings.marker_length >= settings.square_length:
        raise ValueError("--marker-length must be smaller than --square-length.")
    return settings


def sampling_from_args(args: argparse.Namespace) -> FrameSampling:
    if args.every < 1:
        raise ValueError("--every must be at least 1.")
    return FrameSampling(every=args.every, start=args.start, end=args.end)


def progress_printer(args: argparse.Namespace):
    if getattr(args, "quiet", False):
        return None
    return lambda message: print(message, file=sys.stderr, flush=True)


def run_intrinsics(args: argparse.Namespace) -> int:
    source = Path(args.images or args.video)
    camera_name = args.camera_name or sanitize_name(source.stem if args.video else source.name)
    output = Path(args.output) if args.output else CLI_OUTPUT_ROOT / f"{sanitize_name(camera_name)}_intrinsics.json"
    options = IntrinsicsOptions(
        fix_aspect_ratio=args.fix_aspect_ratio,
        fix_k3=args.fix_k3,
        zero_tangential=args.zero_tangential,
        fix_principal_point=args.fix_principal_point,
        max_frames=args.max_frames,
        auto_constrain=args.auto_constrain,
    )
    payload = calibrate_intrinsics(
        source,
        camera_name,
        board_from_args(args),
        args.min_corners,
        args.recursive,
        options=options,
        sampling=sampling_from_args(args),
        progress=progress_printer(args),
    )
    write_json(output, payload)
    print("\n".join(format_intrinsics_summary(payload, output)))
    return 2 if args.fail_on_warnings and payload["quality_warnings"] else 0


def format_intrinsics_summary(payload: dict[str, Any], output: Path | None = None) -> list[str]:
    matrix = payload["camera_matrix"]
    lines = [f"Intrinsics calibration complete for {payload['camera_name']}"]
    if output is not None:
        lines.append(f"Saved: {output}")
    lines.extend(
        [
            f"Accepted frames: {payload['accepted_frame_count']} (used {payload.get('used_frame_count', payload['accepted_frame_count'])})",
            f"Rejected frames: {payload['rejected_frame_count']}",
            f"Reprojection error: {payload['reprojection_error']:.6f}",
            f"fx={matrix[0][0]:.2f} fy={matrix[1][1]:.2f} cx={matrix[0][2]:.2f} cy={matrix[1][2]:.2f}",
            "Distortion coefficients: " + json.dumps([round(value, 6) for value in payload["distortion_coefficients"]]),
        ]
    )
    diversity = payload.get("orientation_diversity")
    if diversity:
        lines.append(
            f"Board orientations: {diversity['distinct_orientations']} distinct, "
            f"spread {diversity['orientation_spread_deg']:.1f} deg"
        )
    used = [item for item in payload.get("accepted_frames", []) if "reprojection_error" in item]
    if used:
        worst = sorted(used, key=lambda item: item["reprojection_error"], reverse=True)[:5]
        lines.append("Worst frames (RMS px):")
        lines.extend(f"  {item['reprojection_error']:.3f}  {item['image_path']}" for item in worst)
    coverage = payload.get("coverage")
    if coverage:
        lines.append(f"Corner coverage ({coverage['fraction']:.0%} of grid cells):")
        lines.append(format_coverage_map(coverage))
    warnings = payload.get("quality_warnings", [])
    lines.append("Quality warnings:" if warnings else "Quality warnings: none")
    lines.extend(f"  WARNING: {message}" for message in warnings)
    return lines


def run_extrinsics(args: argparse.Namespace) -> int:
    source = Path(args.image or args.images or args.video)
    intrinsics_payload = read_json(Path(args.intrinsics))
    camera_name = args.camera_name or intrinsics_payload.get("camera_name") or "camera"
    if args.output:
        output = Path(args.output)
    elif args.image:
        output = CLI_OUTPUT_ROOT / f"{sanitize_name(camera_name)}_extrinsics.json"
    else:
        output = CLI_OUTPUT_ROOT / f"{sanitize_name(camera_name)}_extrinsics_batch.json"
    payload = estimate_extrinsics_for_path(
        source,
        camera_name,
        intrinsics_payload,
        board_from_args(args),
        args.min_corners,
        args.recursive,
        sampling=sampling_from_args(args),
    )
    write_json(output, payload)
    print(f"Extrinsics complete for {camera_name}\nSaved: {output}\nMode: {payload['mode']}")
    if payload["mode"] == "extrinsics_single":
        pose = payload["pose"]
        print(f"Corners: {pose['charuco_corner_count']}  reprojection {pose['reprojection_error']:.3f} px")
    else:
        print(f"Estimated poses: {payload['estimated_pose_count']}  failed: {payload['failed_pose_count']}")
    return 0


def run_multi(args: argparse.Namespace) -> int:
    names, intrinsics, sources = args.camera, args.intrinsics, args.source
    if not (len(names) == len(intrinsics) == len(sources)):
        raise ValueError("Give one --intrinsics and one --source for every --camera.")
    entries = [
        {"camera_name": name, "intrinsics_path": intr, "image_path": source}
        for name, intr, source in zip(names, intrinsics, sources)
    ]
    output = Path(args.output) if args.output else CLI_OUTPUT_ROOT / "multi_camera_extrinsics.json"
    payload = estimate_multi_camera_extrinsics(
        entries,
        board_from_args(args),
        args.min_corners,
        args.reference,
        sampling=sampling_from_args(args),
        max_frame_sets=args.max_frame_sets,
        bundle_adjust=args.bundle_adjustment,
        recursive=args.recursive,
        progress=progress_printer(args),
    )
    write_json(output, payload)
    print("\n".join(format_multi_summary(payload, output)))
    return 0


def format_multi_summary(payload: dict[str, Any], output: Path | None = None) -> list[str]:
    import numpy as np  # type: ignore

    lines = ["Multi-camera extrinsics complete"]
    if output is not None:
        lines.append(f"Saved: {output}")
    lines.extend(
        [
            f"Reference camera: {payload['reference_camera']}",
            f"Frame sets: {payload.get('frame_set_count', 1)}  refinement: {payload.get('refinement', 'single_frame')}",
            f"Overall reprojection error: {payload.get('reprojection_error_px') or 0.0:.3f} px",
            "",
            f"{'camera':<12}{'frames':>8}{'rms px':>10}{'initial px':>12}{'distance to ref':>18}",
        ]
    )
    for item in payload["cameras"]:
        centre = np.array(item["T_camera_to_reference"], dtype=float)[:3, 3]
        rms = item.get("reprojection_error_px")
        initial = item.get("reprojection_error_px_initial")
        lines.append(
            f"{item['camera_name']:<12}{item.get('frame_count', 1):>8}"
            f"{(f'{rms:.3f}' if rms is not None else '-'):>10}"
            f"{(f'{initial:.3f}' if initial is not None else '-'):>12}"
            f"{float(np.linalg.norm(centre)):>18.4f}"
        )
    for message in payload.get("warnings", []):
        lines.append(f"WARNING: {message}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="charuco_calibrator",
        description=f"{APP_TITLE}. Without a command the Tkinter GUI starts.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Render a synthetic board, detect it and estimate its pose, then exit.",
    )
    commands = parser.add_subparsers(dest="command")

    intr = commands.add_parser("intrinsics", help="Calibrate one camera's intrinsics.")
    source = intr.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", help="Folder of ChArUco images.")
    source.add_argument("--video", help="Video file of the moving ChArUco board.")
    intr.add_argument("--camera-name")
    intr.add_argument("--output", help="Output JSON path.")
    intr.add_argument("--min-corners", type=int, default=12)
    intr.add_argument("--recursive", action="store_true")
    intr.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_INTRINSICS_FRAMES,
        help="Pick at most this many pose-diverse frames (0 = use all).",
    )
    intr.add_argument("--fix-aspect-ratio", action="store_true")
    intr.add_argument("--fix-k3", action=argparse.BooleanOptionalAction, default=True)
    intr.add_argument("--zero-tangential", action="store_true")
    intr.add_argument("--fix-principal-point", action="store_true")
    intr.add_argument(
        "--auto-constrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-solve with fixed aspect ratio / principal point when fewer than 3 board orientations are seen.",
    )
    intr.add_argument("--fail-on-warnings", action="store_true", help="Exit with status 2 if quality warnings exist.")
    intr.add_argument("--quiet", action="store_true")
    add_sampling_arguments(intr)
    add_board_arguments(intr)
    intr.set_defaults(handler=run_intrinsics)

    ext = commands.add_parser("extrinsics", help="Board pose for one camera (one image, a folder or a video).")
    ext.add_argument("--intrinsics", required=True, help="Intrinsics JSON.")
    source = ext.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--images")
    source.add_argument("--video")
    ext.add_argument("--camera-name")
    ext.add_argument("--output")
    ext.add_argument("--min-corners", type=int, default=8)
    ext.add_argument("--recursive", action="store_true")
    add_sampling_arguments(ext)
    add_board_arguments(ext)
    ext.set_defaults(handler=run_extrinsics)

    multi = commands.add_parser(
        "multi",
        help="Relative extrinsics of several cameras from synchronized images, image folders or videos.",
    )
    multi.add_argument("--camera", action="append", required=True, help="Camera name (repeat per camera).")
    multi.add_argument("--intrinsics", action="append", required=True, help="Intrinsics JSON (repeat per camera).")
    multi.add_argument(
        "--source",
        action="append",
        required=True,
        help="One image, an image folder (frames matched by file name) or a video (matched by frame index).",
    )
    multi.add_argument("--reference", help="Reference camera name (default: first camera).")
    multi.add_argument("--output")
    multi.add_argument("--min-corners", type=int, default=8)
    multi.add_argument("--recursive", action="store_true")
    multi.add_argument(
        "--max-frame-sets",
        type=int,
        default=DEFAULT_MAX_FRAME_SETS,
        help="Use at most this many synchronized frame sets (0 = all).",
    )
    multi.add_argument("--bundle-adjustment", action=argparse.BooleanOptionalAction, default=True)
    multi.add_argument("--quiet", action="store_true")
    add_sampling_arguments(multi)
    add_board_arguments(multi)
    multi.set_defaults(handler=run_multi)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        try:
            run_self_check()
        except Exception as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    if args.command is None:
        from .gui import run_gui

        return run_gui()
    try:
        return args.handler(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
