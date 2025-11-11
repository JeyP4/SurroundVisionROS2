# Surround Vision for ROS 2

An open-source surround-view visualisation stack for ROS 2 Jazzy. The package fuses four fisheye cameras, renders a calibrated top-down view, and overlays a textured 3D vehicle model. It ships with a fast JPEG decoder based on libturbojpeg, a standalone OBJ inspector, and a demo bag so you can evaluate the pipeline in minutes.

> 🎬 **Poster video:** [`sample_bag/PosterVideo.mp4`](sample_bag/PosterVideo.mp4)

## Highlights

- Real-time stitched bird’s-eye view with vehicle overlay.
- Works with either `sensor_msgs/msg/Image` (raw) or `sensor_msgs/msg/CompressedImage` (JPEG) inputs per camera.
- TurboJPEG-accelerated decoding path with automatic OpenCV fallback when streams are corrupted.
- Interactive OpenGL viewer (orbit / pan / zoom) and an OBJ-only node to inspect vehicle meshes.
- Configuration-first: all topics, decoding modes, window settings, and file paths live in YAML.

## ROS 2 Nodes

| Node | Executable | Purpose |
|------|------------|---------|
| `surround_vision_node` | `surround_vision` | Main compositor: camera ingestion, undistortion/IPM, surround-view rendering, 3D model overlay. |
| `fast_image_republisher` | `fast_image_republisher` | High-throughput JPEG decoding + republishing node built around libturbojpeg (ideal for replaying legacy JPEG bags). |
| `obj_viewer_node` | `obj_viewer_node` | Lightweight OpenGL viewer to inspect the vehicle OBJ, orbit controls, axes overlay, parameterised camera defaults. |

## Sample Bag

A short demo bag is included to validate the pipeline without hardware:

```bash
ros2 bag info sample_bag/
```
```
Files:             sample_bag_0.mcap
Duration:          2.69 s
Messages:          176
Topics:            /frontCamera/v4l2/compressed
                   /leftCamera/v4l2/compressed
                   /rightCamera/v4l2/compressed
                   /rearCamera/v4l2/compressed
                   /insideCamera/v4l2/compressed
                   /groundCamera/v4l2/compressed
```

## Quick Start

```bash
# Clone into your workspace
cd ~/ros2_ws/src
git clone https://github.com/your-org/surround_vision.git

# Install runtime dependencies
sudo apt update
sudo apt install -y \
    libopencv-dev \
    libyaml-cpp-dev \
    libglew-dev \
    libglm-dev \
    libassimp-dev \
    libturbojpeg0-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    ros-jazzy-cv-bridge \
    ros-jazzy-sensor-msgs

# Build
cd ~/ros2_ws
colcon build --packages-select surround_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### Demo Run (compressed sample bag)

```bash
ros2 bag play -l sample_bag
ros2 launch surround_vision surround_vision_rover.launch.py
```

The launch file loads `config/surround_vision_compressed.yaml` by default, which subscribes to the `/v4l2/compressed` topics and enables the TurboJPEG decoding path.

## Configuration

Two YAML configuration presets are provided:

- `config/surround_vision_compressed.yaml` — expects JPEG (`sensor_msgs/msg/CompressedImage`) topics.
- `config/surround_vision_raw.yaml` — uses raw `sensor_msgs/msg/Image` topics instead.

Each camera block exposes fields:

```yaml
front_camera:
  topic: "/frontCamera/v4l2/compressed"   # or "/frontCamera/image"
  transport: "compressed"   # or "raw"
```

Switching between raw and compressed streams is as simple as editing the YAML or supplying an alternative file:

```bash
ros2 launch surround_vision surround_vision_rover.launch.py \
  config:=/path/to/surround_vision_raw.yaml
```

### Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `config` | `config/surround_vision_compressed.yaml` | Parameter file for `surround_vision_node`. |
| `use_sim_time` | `false` | Enable simulation clock. |
| `log_level` | `info` | ROS 2 node log level. |

## Camera Calibration

Surround-view accuracy is tied to **intrinsic** and **extrinsic** calibration quality. The package assumes correct, precomputed parameters and ships a sample `config/camerasParam.yaml`. Calibration workflows (checkerboards, optimisation, etc.) are intentionally out of scope—replace this file with values from your calibration pipeline before using the system on a vehicle.

## Poster Video

Playback `sample_bag/PosterVideo.mp4` locally or embed it in your project page to preview the stitched surround view.

## Controls (Surround Viewer & OBJ Viewer)

- **Left drag**: orbit around the vehicle.
- **Right drag**: pan.
- **Scroll**: zoom.
- **R** (OBJ viewer): reset camera.
- **F** (surround viewer): fullscreen toggle.

## Dependencies at a Glance

- ROS 2 Jazzy (`rclcpp`, `sensor_msgs`, `cv_bridge`)
- OpenGL 3.3+, GLEW, GLFW, GLM
- OpenCV
- yaml-cpp
- assimp
- libturbojpeg (optional but strongly recommended)
- bundled headers: `tiny_obj_loader`, `stb_image`

Consult `CMakeLists.txt` if you are packaging for another platform or need additional compiler flags.

## Repository Layout

```
config/                 # YAML presets (topics, decoding modes, window defaults)
launch/                 # Minimal launch files
models/smartCar/        # Default vehicle OBJ + textures
sample_bag/             # Demo bag + poster video
src/
  surround_vision_node.cpp
  fast_image_republisher_optimized.cpp
  obj_viewer_node.cpp
```

## Tips & Troubleshooting

- **Non-JPEG data on compressed topics** → The decoder logs and skips the frame, falling back to OpenCV only when TurboJPEG fails. Verify your camera driver publishes valid JPEG payloads.
- **Raw topics** → Switch the YAML transport to `raw`; the node automatically skips JPEG checks and consumes `sensor_msgs/msg/Image`.
- **Rendering issues** → Ensure you have an OpenGL 3.3 compatible driver and run with a local display (`ssh -X` for remote sessions).
- **Model alignment** → Adjust scale/orientation in your OBJ or update the parameter file if you replace `smartCar`.

## Contributing

Improvements, bug reports, and calibration workflows are welcome. Please open a GitHub issue or PR with clear descriptions and, where possible, sample data that reproduces the behaviour.

---

**License:** Apache-2.0  