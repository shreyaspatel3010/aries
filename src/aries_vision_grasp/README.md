# aries_vision_grasp

Standalone ROS 2 package for Aries camera visualization, YOLO inference, and
autonomous MoveIt grasp execution.

```bash
colcon build --symlink-install --packages-up-to aries_vision_grasp
source install/setup.bash
ros2 launch aries_vision_grasp vision_grasp.launch.py
```

The default model is installed as `models/grasp.pt`. It is the model previously
named `best(1).pt`: a YOLO26l segmentation model trained for the `probe` class.
The old `best(2).pt` is a different, smaller detection-only model and is not the
default. Override `model_path` when testing other weights.
