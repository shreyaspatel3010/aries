# aries

Robot description and Gazebo simulation package for the Aries rover.

## Layout

- `urdf/`: composable rover, arm, gripper, camera, sensor, and control Xacros.
- `meshes/`: runtime geometry referenced by the robot descriptions.
- `models/`: runtime Gazebo terrain and grasp-object assets.
- `worlds/`: test and Mars-yard Gazebo worlds.
- `config/`: world-specific Gazebo bridges, GUI, teleop, and controller parameters.
- `launch/`: simulation, visualization, waypoint, and legacy Mars-yard launch files.
- `rviz/`: RViz display configurations.
- `scripts/`: small simulation and navigation support nodes.

Use `my_robot.launch.py` for the maintained full simulation. The XML launch is
retained as `legacy_marsyard.launch.xml` because it uses the older Mars-yard
spawn defaults and controller startup sequence.

```bash
colcon build --symlink-install --packages-up-to aries
source install/setup.bash
ros2 launch aries my_robot.launch.py
```
