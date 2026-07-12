# Vision-Based Grasping - Complete Implementation

## What It Does

**YES - The entire arm now moves to detected objects and grasps them!**

When an object is detected by YOLO:
1. ✅ **Detects object** using gripper camera + YOLO
2. ✅ **Converts to 3D position** using depth data
3. ✅ **Opens gripper** to prepare for grasp
4. ✅ **Moves arm** to pre-grasp position (15cm above object)
5. ✅ **Lowers arm** to grasp position (10cm above object)
6. ✅ **Closes gripper** to grab object
7. ✅ **Lifts arm up** 25cm with object

## Complete Workflow

### State Machine
```
IDLE → DETECTING → APPROACHING → GRASPING → LIFTING → DONE
```

### Terminal 1: Launch Robot
```bash
# Start Gazebo + Robot + MoveIt
ros2 launch aries my_robot.launch.py
```

### Terminal 2: Activate Vision Environment & Launch Grasp Node
```bash
# Activate vision environment (REQUIRED for YOLO + OpenCV)
source ~/aries/scripts/vision/setup_environment.bash

# Launch vision grasp with default settings (detects "bottle")
ros2 launch aries_vision_grasp vision_grasp.launch.py

# OR specify custom target class and model
ros2 launch aries_vision_grasp vision_grasp.launch.py \
    target_class:=cup \
    confidence_threshold:=0.6 \
    model_path:=/path/to/custom_yolo.pt
```

### Terminal 3: Monitor Detection (Optional)
```bash
source ~/aries/scripts/vision/setup_environment.bash
ros2 run aries_vision_grasp camera_viewer.py
```

## How the Arm Movement Works

### MoveIt Integration
The system uses **MoveIt's action server** to plan and execute arm motions:

```python
# Each detected object position triggers:
1. Convert pixel (u, v) → 3D point (x, y, z) in camera frame
2. Create PoseStamped target in world frame
3. Send MoveGroup.Goal to /move_action
4. MoveIt plans collision-free path
5. Arm executes motion via rebel_arm_trajectory_controller
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_class` | "bottle" | YOLO class name to detect |
| `confidence_threshold` | 0.5 | Minimum detection confidence |
| `grasp_offset_z` | 0.10m | Height above object center for grasp |
| `pre_grasp_offset_z` | 0.15m | Approach height before lowering |
| `gripper_open_width` | 0.08 | Open gripper position (8cm) |
| `gripper_close_width` | 0.02 | Closed gripper position (2cm) |

### Gripper Control
```bash
# Gripper commands published to:
/aries/arm_gripper_left_joint/cmd_pos
```

## Testing Different Objects

### Default YOLO Classes (80 COCO classes)
```bash
# Common objects you can detect:
- bottle
- cup
- bowl
- apple
- orange
- banana
- cell phone
- keyboard
- mouse
- book
- scissors
- teddy bear
```

### Spawn Test Object in Gazebo
```bash
# In Gazebo GUI: Insert → select object model
# Or use gz service call to spawn
```

### Train Custom Model for "Probe"
```bash
# 1. Collect images from gripper camera
ros2 topic echo /gripper_camera/color/image_raw --once > image.raw

# 2. Annotate on Roboflow or Label Studio
# 3. Train YOLOv8:
yolo train data=probe_dataset.yaml model=yolov8n.pt epochs=100

# 4. Use custom model:
ros2 launch aries_vision_grasp vision_grasp.launch.py \
    model_path:=~/models/probe_yolov8.pt \
    target_class:=probe
```

## Debugging

### Check if Camera is Working
```bash
source ~/aries/scripts/vision/setup_environment.bash
ros2 run aries_vision_grasp camera_viewer.py

# Should show:
# - Color image from gripper camera
# - Depth visualization with center distance
# - Crosshair at image center
```

### Check MoveIt Action Server
```bash
ros2 action list
# Should show: /move_action
```

### Check Controller Status
```bash
ros2 control list_controllers

# Should show:
# - rebel_arm_trajectory_controller [active]
# - joint_state_broadcaster [active]
```

### Monitor Grasp States
```bash
ros2 topic echo /rosout | grep vision_grasp

# You'll see state transitions:
# - "Detected bottle at pixel..."
# - "Moving arm to pre-grasp position..."
# - "MoveIt goal accepted..."
# - "MoveIt motion completed successfully!"
# - "Closing gripper..."
# - "Lifting object up..."
# - "Grasp sequence completed!"
```

## Advanced Usage

### Custom Grasp Parameters
```bash
ros2 launch aries_vision_grasp vision_grasp.launch.py \
    grasp_offset_z:=0.05 \
    pre_grasp_offset_z:=0.20 \
    gripper_close_width:=0.01
```

### Faster/Slower Motion
Edit in code:
```python
goal.request.max_velocity_scaling_factor = 0.5  # 0.1-1.0 (default 0.3)
goal.request.max_acceleration_scaling_factor = 0.5  # 0.1-1.0 (default 0.3)
```

### Continuous Detection Mode
The node continuously detects in IDLE state and executes when target found.
After DONE state, it stays completed. To reset:
```bash
# Stop and restart the node
Ctrl+C in vision terminal
ros2 launch aries_vision_grasp vision_grasp.launch.py
```

## Architecture Summary

```
┌─────────────────────────────────────────────┐
│  Gripper D435i Camera                       │
│  - Color: 640x480 @ 30Hz                    │
│  - Depth: 640x480 @ 30Hz                    │
│  - Mounted on arm_gripper_base_link         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  vision_grasp_node.py                       │
│  - YOLOv8 object detection                  │
│  - Pixel → 3D conversion (depth + K matrix) │
│  - State machine control                    │
│  - MoveIt action client                     │
└──────────────┬──────────────────────────────┘
               │
               ├──► /move_action (MoveGroup)
               │    └──► MoveIt Motion Planning
               │         └──► rebel_arm_trajectory_controller
               │              └──► Gazebo Arm Joints
               │
               └──► /aries/arm_gripper_left_joint/cmd_pos
                    └──► Gripper Position Control
```

## Notes

- **Vision environment required**: Always `source ~/aries/scripts/vision/setup_environment.bash` before running vision nodes
- **Robot launch doesn't need venv**: Just vision nodes need the virtual environment
- **Camera frame**: All detections are in `gripper_camera_depth_optical_frame`
- **Planning group**: Uses `rebel_arm` MoveIt group for arm motion
- **End-effector**: Plans to `arm_gripper_base_link` pose
- **Collision checking**: MoveIt automatically avoids collisions during motion

## Next Steps

1. **Test with simple object**: Launch robot, spawn a bottle in Gazebo near gripper
2. **Verify detection**: Check camera_viewer.py shows object in view
3. **Run grasp**: Launch vision_grasp.launch.py and watch arm autonomously grasp
4. **Train custom model**: Collect probe images and train YOLOv8 for your specific object
5. **Tune parameters**: Adjust grasp offsets and speed for your use case

Enjoy autonomous robotic grasping! 🦾🎯
