# Vision Grasp Troubleshooting

## Environment conflicts

ROS `cv_bridge`, NumPy, OpenCV, and Ultralytics must be loaded from a compatible
Python environment. Use the workspace helper rather than installing packages
into the system interpreter:

```bash
cd ~/aries
./scripts/vision/install_dependencies.sh
source scripts/vision/setup_environment.bash
```

The installer pins NumPy 1.26.4 and OpenCV 4.8.1.78 in `vision_venv/`. The
activation helper then sources the workspace's `install/setup.bash`.

## Workspace is not built

If the activation helper warns that `install/setup.bash` is missing:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install --packages-up-to aries_vision_grasp
source scripts/vision/setup_environment.bash
```

## Model does not load

The default model is installed at
`install/aries_vision_grasp/share/aries_vision_grasp/models/grasp.pt`. Rebuild the package
after pulling or changing the model:

```bash
colcon build --symlink-install --packages-up-to aries_vision_grasp
python3 scripts/vision/check_model.py
```

To use different weights, pass an absolute path:

```bash
ros2 launch aries_vision_grasp vision_grasp.launch.py model_path:=/path/to/model.pt
```

## Camera or ROS topics are missing

```bash
ros2 topic list | grep gripper_camera
ros2 topic hz /gripper_camera/color/image_raw
```

Confirm that the robot/camera launch is running before starting the vision
node. See the [usage guide](usage.md) for expected topics and parameters.
