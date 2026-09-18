# HEAD_RING_MODEL_8 Isaac Sim 验证程序

本目录在原 `Issacsim_data_generation` 管线基础上，将四路矩形头戴相机替换为
`HEAD_RING_MODEL_8_NEW_camera_poses_head_frame.csv` 中的八路头环相机。原四相机入口及默认配置保持不变。

## 相机数据约定

- `x_mm/y_mm/z_mm` 是相机中心在 SMPL-X Head rig 局部坐标系中的位置，加载时从毫米转换为米。
- `r00` 至 `r22` 按行组成 `rotation_cam_to_head`。
- 相机局部 `+Z` 是光轴；程序会校验旋转矩阵第三列与 CSV 的
  `optical_x/optical_y/optical_z` 一致。
- 文件名明确声明姿态位于 head frame，因此程序不额外翻转或交换坐标轴。
- 八路输出名称依次为 `cam_01` 至 `cam_08`，不再生成腕部相机视频。

相机世界姿态仍按原管线计算：

```text
camera_position_world = head_position_world + head_rotation_world @ position_head
camera_rotation_world = head_rotation_world @ rotation_cam_to_head
```

## 仅生成运动缓存

当前仓库不包含 AMASS/HumanEva 动作与 SMPL-X 模型，需要先将它们放到本机：

```bash
python3 ego_8c_val/render.py \
  --motion /path/to/AMASS/motion_stageii.npz \
  --smplx-model /path/to/SMPLX_NEUTRAL_2020.npz \
  --output-dir ego_8c_val/outputs/motion \
  --prepare-only
```

生成的 `blenderproc_motion_cache.npz` 中，`camera_names`、`camera_positions` 和
`camera_rotations` 仅包含八路头环相机。

## Isaac Sim 渲染

```bash
python3 ego_8c_val/render.py \
  --motion /path/to/AMASS/motion_stageii.npz \
  --smplx-model /path/to/SMPLX_NEUTRAL_2020.npz \
  --isaacsim-python /path/to/isaacsim/python.sh \
  --output-dir ego_8c_val/outputs/motion \
  --parallel-cameras 8 \
  --gpu-ids 0,1,2,3
```

只渲染一路：

```bash
python3 ego_8c_val/render.py ... --camera-name cam_01
```

输出文件名形如：

```text
motion_stageii_cam_01.mp4
motion_stageii_cam_01_isaacsim_stats.json
logs/cam_01.log
```

默认沿用原程序的 220° OpenCV 鱼眼模型、1920×1080 输出和
`smplx_relative` 头部方向模式。可用 `--head-frame shoulders` 或
`--head-frame smplx` 切换原有的头部方向算法。

## 无 Isaac Sim 的几何测试

```bash
python3 -m unittest ego_8c_val/test_head_ring.py
```
