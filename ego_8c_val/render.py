#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


EGO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EGO_ROOT.parent
LEGACY_ROOT = REPO_ROOT / "Issacsim_data_generation"
LEGACY_SRC = LEGACY_ROOT / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))
if str(EGO_ROOT) not in sys.path:
    sys.path.insert(0, str(EGO_ROOT))

from geosim.blenderproc_cache import build_blenderproc_motion_cache_from_amass
from geosim.config import load_config
from geosim.smplx_numpy import load_smplx_model
from geosim.tag_rig import make_wrist_tag_rig
from head_ring import load_head_ring_cameras, load_head_ring_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render AMASS/SMPL-X motion with the HEAD_RING_MODEL_8 camera rig."
    )
    parser.add_argument("--motion", required=True, help="AMASS-style .npz containing poses and trans.")
    parser.add_argument(
        "--poses-csv",
        default=str(EGO_ROOT / "HEAD_RING_MODEL_8_NEW_camera_poses_head_frame.csv"),
        help="Eight camera-to-head poses. Positions are interpreted as millimetres.",
    )
    parser.add_argument(
        "--config",
        default=str(LEGACY_ROOT / "configs/default_geometry.json"),
        help="Shared image size, fisheye FOV and wrist-tag geometry.",
    )
    parser.add_argument(
        "--smplx-model",
        default=str(LEGACY_ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"),
    )
    parser.add_argument("--output-dir", default="", help="Defaults to ego_8c_val/outputs/<motion_stem>.")
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920, help="Final cropped video width.")
    parser.add_argument("--height", type=int, default=1080, help="Final cropped video height.")
    parser.add_argument("--max-output-frames", type=int, default=0)
    parser.add_argument(
        "--head-frame",
        default="smplx_relative",
        choices=("smplx_relative", "smplx", "shoulders"),
    )
    parser.add_argument("--isaacsim-python", default="")
    parser.add_argument(
        "--renderer",
        default="RayTracedLighting",
        choices=("RayTracedLighting", "PathTracing"),
    )
    parser.add_argument("--rt-subframes", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=12)
    parser.add_argument("--video-format", default="mp4", choices=("mp4", "avi"))
    parser.add_argument("--camera-name", default="", help="Render one camera, for example cam_01.")
    parser.add_argument("--hide-wrist-tags", action="store_true")
    parser.add_argument("--parallel-cameras", type=int, default=0)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--random-seed", type=int, default=20260609)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    motion_path = Path(args.motion).expanduser().resolve()
    poses_csv = Path(args.poses_csv).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else EGO_ROOT / "outputs" / motion_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model = load_smplx_model(args.smplx_model)
    poses = load_head_ring_poses(poses_csv)
    cameras = load_head_ring_cameras(
        poses_csv,
        image_width=args.width,
        image_height=args.width,
        fov_deg=config.camera_rig.fisheye_fov_deg,
    )
    legacy_render = _load_legacy_render_module()
    rng = np.random.default_rng(int(args.random_seed))
    cache_result = build_blenderproc_motion_cache_from_amass(
        motion_path=motion_path,
        model=model,
        config=config,
        tag_rig=make_wrist_tag_rig(config.tag_rig),
        output_dir=output_dir,
        output_fps=args.output_fps,
        max_output_frames=args.max_output_frames,
        video_width=args.width,
        video_height=args.height,
        head_frame_mode=args.head_frame,
        body_face_groups=legacy_render._make_body_face_groups(model),
        body_group_colors=legacy_render._make_body_group_colors(rng),
        scene_config=legacy_render._make_random_scene_config(rng),
        head_cameras=cameras,
        head_camera_labels={camera.name: camera.name for camera in cameras},
        include_wrist_cameras=False,
    )
    _record_camera_source(
        cache_result.metadata_path,
        poses_csv=poses_csv,
        source_face_indices={pose.camera_id: pose.source_face_index for pose in poses},
    )

    summary = {
        "backend": "isaacsim",
        "camera_rig": "HEAD_RING_MODEL_8_NEW",
        "camera_pose_csv": str(poses_csv),
        "cache": str(cache_result.cache_path),
        "metadata": str(cache_result.metadata_path),
        "output_dir": str(output_dir),
        "frames": cache_result.frame_count,
        "fps": cache_result.output_fps,
        "cameras": list(cache_result.camera_names),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.prepare_only:
        return 0

    camera_names = tuple(
        name for name in cache_result.camera_names if not args.camera_name or name == args.camera_name
    )
    if args.camera_name and not camera_names:
        raise ValueError(
            f"Unknown camera {args.camera_name!r}. "
            f"Available cameras: {', '.join(cache_result.camera_names)}"
        )

    isaacsim_python = legacy_render._resolve_isaacsim_python(args.isaacsim_python)
    gpu_ids = legacy_render._resolve_gpu_ids(args.gpu_ids)
    parallel_cameras = legacy_render._resolve_parallel_camera_count(
        args.parallel_cameras,
        "GPU",
        gpu_ids,
        len(camera_names),
    )
    legacy_render._run_isaacsim_camera_jobs(
        isaacsim_python=isaacsim_python,
        runner=LEGACY_SRC / "geosim/isaacsim_runner.py",
        cache_path=cache_result.cache_path,
        output_dir=output_dir,
        renderer=args.renderer,
        rt_subframes=args.rt_subframes,
        warmup_frames=args.warmup_frames,
        video_format=args.video_format,
        hide_wrist_tags=args.hide_wrist_tags,
        motion_label=motion_path.stem,
        camera_names=camera_names,
        parallel_cameras=parallel_cameras,
        gpu_ids=gpu_ids,
    )
    return 0


def _load_legacy_render_module():
    module_path = LEGACY_ROOT / "scripts/render.py"
    spec = importlib.util.spec_from_file_location("legacy_geosim_render", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy renderer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record_camera_source(
    metadata_path: Path,
    *,
    poses_csv: Path,
    source_face_indices: dict[str, int],
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "camera_rig": "HEAD_RING_MODEL_8_NEW",
            "camera_pose_csv": str(poses_csv),
            "camera_pose_units": "millimetres_in_csv_metres_in_cache",
            "camera_pose_axis_conversion": "none",
            "source_face_indices": source_face_indices,
            "include_wrist_cameras": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
