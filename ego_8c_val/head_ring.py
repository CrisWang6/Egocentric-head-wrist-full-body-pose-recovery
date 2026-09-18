from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geosim.camera import FisheyeCamera


REQUIRED_COLUMNS = {
    "camera_id",
    "source_face_index",
    "x_mm",
    "y_mm",
    "z_mm",
    "optical_x",
    "optical_y",
    "optical_z",
    "r00",
    "r01",
    "r02",
    "r10",
    "r11",
    "r12",
    "r20",
    "r21",
    "r22",
}


@dataclass(frozen=True)
class HeadRingPose:
    camera_id: str
    source_face_index: int
    position_head_m: np.ndarray
    rotation_cam_to_head: np.ndarray
    optical_axis_head: np.ndarray


def load_head_ring_poses(
    csv_path: str | Path,
    *,
    expected_count: int = 8,
) -> list[HeadRingPose]:
    """Load camera-to-head poses without changing the CSV's head-frame axes."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if len(rows) != expected_count:
        raise ValueError(f"{path} contains {len(rows)} camera rows; expected {expected_count}.")

    poses: list[HeadRingPose] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        camera_id = str(row["camera_id"]).strip()
        if not camera_id:
            raise ValueError(f"{path}:{row_number} has an empty camera_id.")
        if camera_id in seen_ids:
            raise ValueError(f"{path}:{row_number} repeats camera_id {camera_id!r}.")
        seen_ids.add(camera_id)

        try:
            position_head_m = np.array(
                [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])],
                dtype=float,
            ) / 1000.0
            optical_axis_head = np.array(
                [float(row["optical_x"]), float(row["optical_y"]), float(row["optical_z"])],
                dtype=float,
            )
            rotation_cam_to_head = np.array(
                [
                    [float(row["r00"]), float(row["r01"]), float(row["r02"])],
                    [float(row["r10"]), float(row["r11"]), float(row["r12"])],
                    [float(row["r20"]), float(row["r21"]), float(row["r22"])],
                ],
                dtype=float,
            )
            source_face_index = int(row["source_face_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{row_number} contains an invalid numeric value.") from exc

        _validate_pose(
            path=path,
            row_number=row_number,
            rotation_cam_to_head=rotation_cam_to_head,
            optical_axis_head=optical_axis_head,
            position_head_m=position_head_m,
        )
        poses.append(
            HeadRingPose(
                camera_id=camera_id,
                source_face_index=source_face_index,
                position_head_m=position_head_m,
                rotation_cam_to_head=rotation_cam_to_head,
                optical_axis_head=optical_axis_head,
            )
        )
    return poses


def load_head_ring_cameras(
    csv_path: str | Path,
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> list[FisheyeCamera]:
    poses = load_head_ring_poses(csv_path)
    return [
        FisheyeCamera(
            name=pose.camera_id,
            position_head=pose.position_head_m,
            rotation_cam_to_head=pose.rotation_cam_to_head,
            image_width=int(image_width),
            image_height=int(image_height),
            fov_deg=float(fov_deg),
        )
        for pose in poses
    ]


def _validate_pose(
    *,
    path: Path,
    row_number: int,
    rotation_cam_to_head: np.ndarray,
    optical_axis_head: np.ndarray,
    position_head_m: np.ndarray,
) -> None:
    if not (
        np.isfinite(position_head_m).all()
        and np.isfinite(optical_axis_head).all()
        and np.isfinite(rotation_cam_to_head).all()
    ):
        raise ValueError(f"{path}:{row_number} contains a non-finite pose value.")

    if not np.allclose(rotation_cam_to_head.T @ rotation_cam_to_head, np.eye(3), atol=1e-6):
        raise ValueError(f"{path}:{row_number} rotation matrix is not orthonormal.")
    determinant = float(np.linalg.det(rotation_cam_to_head))
    if not np.isclose(determinant, 1.0, atol=1e-6):
        raise ValueError(f"{path}:{row_number} rotation determinant is {determinant:.6f}, not +1.")

    optical_norm = float(np.linalg.norm(optical_axis_head))
    if not np.isclose(optical_norm, 1.0, atol=1e-6):
        raise ValueError(f"{path}:{row_number} optical axis is not unit length.")
    if not np.allclose(rotation_cam_to_head[:, 2], optical_axis_head, atol=1e-6):
        raise ValueError(f"{path}:{row_number} optical axis does not match rotation column r02/r12/r22.")
