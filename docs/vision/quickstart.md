# Quick Start Guide: Vision-Based Grasping

## Setup

### 1. Install Dependencies (Workspace-Specific Virtual Environment)

Run the automated installation script:
```bash
cd ~/aries
./scripts/vision/install_dependencies.sh
```

This creates a virtual environment at `~/aries/vision_venv` with all required packages.

### 2. Activate Environment

**Every time you want to use vision grasp**, activate the environment:
```bash
source ~/aries/vision_venv/bin/activate
source ~/aries/install/setup.bash
```

**Tip:** Add an alias to your `~/.bashrc` for convenience:
```bash
echo "alias aries_vision='source ~/aries/vision_venv/bin/activate && source ~/aries/install/setup.bash'" >> ~/.bashrc
source ~/.bashrc

# Then you can just run:
aries_vision
```

### 2. Build Workspace
```bash
cd ~/aries
colcon build --symlink-install --packages-up-to aries_vision_grasp
```

## Using Vision Grasp

### Always activate the environment first:
```bash
# Option 1: Manual activation
source ~/aries/vision_venv/bin/activate
source ~/aries/install/setup.bash

# Option 2: Use the alias (if you added it)
aries_vision
```

## Testing the Camera

### Step 1: Launch Robot
```bash
# Terminal 1: Standard ROS2 launch (no vision env needed)
ros2 launch aries my_robot.launch.py
```

### Step 2: Test Camera Feed
In a new terminal:
```bash
# Activate vision environment
source ~/aries/vision_venv/bin/activate
source ~/aries/install/setup.bash

# Run camera viewer
ros2 run aries_vision_grasp camera_viewer.py
```

You should see two windows:
- **Gripper Color**: RGB camera view
- **Gripper Depth**: Depth visualization with center distance

Press 'q' to quit.

### Step 3: Check Camera Topics
```bash
# List camera topics
ros2 topic list | grep gripper_camera

# Check image rate
ros2 topic hz /gripper_camera/color/image_raw

# View single frame
ros2 topic echo /gripper_camera/color/image_raw --once
```

## Running Vision Grasp

### Basic Usage
```bash
# Terminal 1: Launch robot (no vision env needed)
ros2 launch aries my_robot.launch.py

# Terminal 2: Activate vision environment and launch vision grasp
source ~/aries/vision_venv/bin/activate
source ~/aries/install/setup.bash
ros2 launch aries_vision_grasp vision_grasp.launch.py
```

### Custom Object Detection
```bash
# Activate environment first
source ~/aries/vision_venv/bin/activate
source ~/aries/install/setup.bash

# Detect cups instead of bottles
ros2 launch aries_vision_grasp vision_grasp.launch.py target_class:=cup

# Detect person
ros2 launch aries_vision_grasp vision_grasp.launch.py target_class:=person confidence_threshold:=0.7
```

### Available Object Classes (YOLOv8 COCO)
Common objects you can detect:
- **People**: person
- **Containers**: bottle, cup, bowl, vase
- **Electronics**: laptop, mouse, keyboard, cell phone, remote
- **Food**: banana, apple, sandwich, orange, broccoli, carrot
- **Tools**: scissors, knife, spoon, fork
- **Sports**: baseball, tennis racket, sports ball
- **Furniture**: chair, couch, bed, dining table
- **Vehicles**: car, truck, bus, motorcycle, bicycle

Full list of 80 classes available in COCO dataset.

## Visualizing Detections

### Method 1: RViz
1. Open RViz (already running with my_robot.launch.py)
2. Click "Add" → "By topic"
3. Select `/vision_grasp/detection_image` → Image
4. You'll see bounding boxes around detected objects

### Method 2: rqt_image_view
```bash
ros2 run rqt_image_view rqt_image_view /vision_grasp/detection_image
```

## Manual Gripper Control

Test gripper commands manually:

```bash
# Open gripper
ros2 topic pub --once /aries/arm_gripper_left_joint/cmd_pos std_msgs/Float64 "data: 0.08"

# Close gripper
ros2 topic pub --once /aries/arm_gripper_left_joint/cmd_pos std_msgs/Float64 "data: 0.02"

# Partial close
ros2 topic pub --once /aries/arm_gripper_left_joint/cmd_pos std_msgs/Float64 "data: 0.04"
```

## Training Custom Model for Probes

### 1. Collect Images
Move robot around and capture images:
```bash
# Save images from camera
ros2 run image_view image_saver --ros-args -r image:=/gripper_camera/color/image_raw
```

Capture 100-500 images of probes in different:
- Positions
- Orientations
- Lighting conditions
- Backgrounds

### 2. Annotate with Roboflow
1. Create account at https://roboflow.com
2. Create new project "Probe Detection"
3. Upload images
4. Draw bounding boxes around probes
5. Label as "probe"
6. Export in "YOLOv8" format

### 3. Train Model
```python
from ultralytics import YOLO

# Train
model = YOLO('yolov8n.pt')
results = model.train(
    data='probe_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    name='probe_detector'
)
```

### 4. Use Trained Model
```bash
ros2 launch aries_vision_grasp vision_grasp.launch.py \
    model_path:=~/runs/detect/probe_detector/weights/best.pt \
    target_class:=probe
```

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip3 install ultralytics
```

### Camera not publishing
Check if Gazebo is running:
```bash
ros2 topic list | grep camera
```

If no topics, restart launch file.

### Low detection rate
1. Lower confidence: `confidence_threshold:=0.3`
2. Better lighting in Gazebo
3. Move camera closer to objects
4. Train custom model for your specific objects

### Gripper not moving
Check controller status:
```bash
ros2 control list_controllers

# Should see:
# rebel_gripper_controller[joint_trajectory_controller/JointTrajectoryController] active
```

### MoveIt not planning
Ensure robot spawned correctly:
```bash
ros2 topic echo /joint_states
```

## Example Workflow

1. **Launch robot**:
   ```bash
   ros2 launch aries my_robot.launch.py
   ```

2. **Test camera view**:
   ```bash
   ros2 run aries_vision_grasp camera_viewer.py
   ```

3. **Place object in front of gripper in Gazebo**
   - Use Gazebo GUI to spawn a bottle or other object
   - Position it within camera view (0.2-2.0 meters)

4. **Start vision grasp**:
   ```bash
   ros2 launch aries_vision_grasp vision_grasp.launch.py
   ```

5. **Monitor detections**:
   ```bash
   ros2 topic echo /vision_grasp/detection_image
   ```

6. **Watch the robot**:
   - Detects object
   - Opens gripper
   - Approaches object
   - Closes gripper
   - Lifts object

## Next Steps

- Train custom YOLO model for probes
- Integrate with navigation for mobile manipulation
- Add force control for delicate grasping
- Implement object pose estimation
- Add collision avoidance with depth sensing

## Support

For issues or questions, see the full documentation in:
`docs/vision/usage.md`
