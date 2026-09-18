from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent
LEGACY_SRC = ROOT.parent / "Issacsim_data_generation/src"
for path in (ROOT, LEGACY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geosim.blenderproc_cache import _camera_pose_arrays_from_sampled
from geosim.config import SimulationConfig
from head_ring import load_head_ring_cameras, load_head_ring_poses


POSES_CSV = ROOT / "HEAD_RING_MODEL_8_NEW_camera_poses_head_frame.csv"


class HeadRingLoaderTest(unittest.TestCase):
    def test_loads_all_eight_camera_poses_in_csv_order(self) -> None:
        poses = load_head_ring_poses(POSES_CSV)

        self.assertEqual([pose.camera_id for pose in poses], [f"cam_{idx:02d}" for idx in range(1, 9)])
        np.testing.assert_allclose(
            poses[0].position_head_m,
            [-0.060000028422, -0.159972466364, 0.082546157216],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            poses[-1].position_head_m,
            [0.109260159299, 0.175999337595, 0.0425],
            atol=1e-12,
        )

    def test_rotation_third_column_matches_optical_axis(self) -> None:
        for pose in load_head_ring_poses(POSES_CSV):
            np.testing.assert_allclose(
                pose.rotation_cam_to_head[:, 2],
                pose.optical_axis_head,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                pose.rotation_cam_to_head.T @ pose.rotation_cam_to_head,
                np.eye(3),
                atol=1e-9,
            )
            self.assertAlmostEqual(float(np.linalg.det(pose.rotation_cam_to_head)), 1.0, places=8)

    def test_custom_rig_generates_head_only_cache_pose_arrays(self) -> None:
        cameras = load_head_ring_cameras(
            POSES_CSV,
            image_width=1920,
            image_height=1920,
            fov_deg=220.0,
        )
        frame_count = 2
        head_pos = np.array([[0.0, 0.0, 1.6], [0.1, -0.2, 1.7]])
        head_rot = np.repeat(np.eye(3)[None, :, :], frame_count, axis=0)
        unused_positions = np.zeros((frame_count, 3))
        unused_rotations = np.repeat(np.eye(3)[None, :, :], frame_count, axis=0)

        names, positions, rotations = _camera_pose_arrays_from_sampled(
            config=SimulationConfig(),
            head_pos=head_pos,
            head_rot=head_rot,
            left_wrist_pos=unused_positions,
            left_wrist_rot=unused_rotations,
            right_wrist_pos=unused_positions,
            right_wrist_rot=unused_rotations,
            width=1920,
            height=1920,
            head_cameras=cameras,
            head_camera_labels={camera.name: camera.name for camera in cameras},
            include_wrist_cameras=False,
        )

        self.assertEqual(names, [f"cam_{idx:02d}" for idx in range(1, 9)])
        self.assertEqual(positions.shape, (8, frame_count, 3))
        self.assertEqual(rotations.shape, (8, frame_count, 3, 3))
        np.testing.assert_allclose(positions[0, 0], head_pos[0] + cameras[0].position_head)
        np.testing.assert_allclose(rotations[0, 0], cameras[0].rotation_cam_to_head)


if __name__ == "__main__":
    unittest.main()
