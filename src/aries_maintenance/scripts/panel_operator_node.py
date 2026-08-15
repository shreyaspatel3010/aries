#!/usr/bin/env python3
"""Find the maintenance panel with the gripper camera and operate its controls.

    ros2 run aries_vision_grasp panel_operator_node.py

Watches /gripper_camera for ArUco 11/13/14/15, solves the panel's pose, publishes
it as a TF frame, and drives the arm through any control in `panel_task.json`:

    ros2 topic pub --once /panel/operate std_msgs/String "data: 'mcb_3'"
    ros2 topic pub --once /panel/operate std_msgs/String "data: 'all_breakers'"

Three shapes of action, taken from the table rather than decided here:

  flick  14 MCB toggles - jaws CLOSED, drive into the toggle and sweep up-slope.
         They sit on a 17.7 mm pitch (3.9 mm between modules) so they cannot be
         grasped, only pushed. Same for the 5 buttons, which are pressed along
         the console normal.
  turn   5 selectors and 2 red disconnects - grip, then roll the wrist about the
         approach axis. The jaw line MUST run up-slope: the disconnects are
         65 mm across on a 76.7 mm pitch, leaving 11.7 mm between them, so a
         grasp closing across the console cannot fit, while up-slope has 35.8 mm
         clear above and 36.2 mm below.
  press  the 5 push buttons, 4 mm of travel along the normal.

WHAT THIS NODE DELIBERATELY DOES NOT DO
It never plans free-space to a contact pose. Every motion from the standoff pose
inward is a Cartesian path, and a fraction < 1.0 is reported and refused rather
than executed part-way - the same rule the grasp node follows, for the same
reason: a joint-space plan that ends at the right pose can sweep the tool
through the console on the way there.
"""

import math
import pathlib
import sys
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from aries_vision_grasp.image_bridge import NumpyImageBridge
from aries_maintenance.panel_alignment import (
    control_waypoints, detect_markers, load_task_table, panel_pose_from_markers,
    quaternion_from_matrix,
)


def _pose_msg(transform):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(v) for v in np.asarray(transform)[:3, 3])
    x, y, z, w = quaternion_from_matrix(transform)
    pose.orientation.x, pose.orientation.y = float(x), float(y)
    pose.orientation.z, pose.orientation.w = float(z), float(w)
    return pose


def _rotation_about(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) * c + s * cross + (1 - c) * np.outer(axis, axis)


class PanelOperator(Node):

    def __init__(self):
        super().__init__('panel_operator')
        group = ReentrantCallbackGroup()
        p = self.declare_parameter

        # Installed, this script lives in lib/, so a path relative to __file__
        # only resolves when running from the source tree. Ask ament first.
        try:
            from ament_index_python.packages import get_package_share_directory
            share = pathlib.Path(get_package_share_directory('aries'))
        except Exception:
            share = pathlib.Path(__file__).resolve().parents[2] / 'aries'
        default_table = share / 'models' / 'maintenance_panel' / 'panel_task.json'
        p('task_table', str(default_table))
        p('planning_frame', 'base_link')
        p('planning_group', 'igus_rebel_arm')
        p('tool_frame', 'gripper_tcp')
        # BOTH cameras feed the localiser. The gripper camera is the accurate
        # one - it gets close - but it is only aimed at the panel some of the
        # time and loses the tags entirely once the tool closes in. The rover's
        # body camera sees the whole console from further back, so it is what
        # actually acquires the panel. Whichever gives the better fit wins.
        p('image_topics', ['/gripper_camera/color/image_raw',
                           '/camera/color/image_raw'])
        p('camera_info_topics', ['/gripper_camera/color/camera_info',
                                 '/camera/color/camera_info'])
        p('camera_frames', ['gripper_camera_color_optical_frame',
                            'camera_color_optical_frame'])
        p('cartesian_service_name', '/compute_cartesian_path')
        p('execute_action_name', '/execute_trajectory')
        p('gripper_topic', '/aries/gripper_gear_left_joint/cmd_pos')
        p('gripper_open_width', -1.20)
        p('gripper_close_width', 0.07)
        p('gripper_settle_sec', 1.5)
        # Cartesian steps this fine keep the tool on the straight line between
        # waypoints; a coarse step lets MoveIt cut the corner into the console.
        p('cartesian_step_m', 0.002)
        p('cartesian_jump_threshold', 0.0)
        p('min_cartesian_fraction', 0.99)
        # A pose fitted from one tag carries the planar-PnP ambiguity, so by
        # default the arm refuses to move on it. Two tags remove it.
        p('min_markers', 2)
        p('max_reprojection_px', 3.0)
        # The panel is bolted to the world, and the gripper camera LOSES the
        # tags as soon as the tool closes on a control - they leave the frame
        # long before contact. So the pose is latched once acquired and reused,
        # rather than required fresh at command time. Requiring freshness meant
        # every command was refused with "pose is stale" and the arm never
        # moved. Set `require_fresh_pose` if the panel might have been nudged.
        p('require_fresh_pose', False)
        p('pose_timeout_sec', 2.0)

        def get(name):
            return self.get_parameter(name).value

        self.get = get
        self.table = load_task_table(get('task_table'))
        self.controls = {c['name']: c for c in self.table['controls']}
        self.get_logger().info(
            f"loaded {len(self.controls)} controls from {get('task_table')}")

        self.bridge = NumpyImageBridge()
        self.cameras = {}
        for topic, info_topic, frame in zip(get('image_topics'),
                                            get('camera_info_topics'),
                                            get('camera_frames')):
            self.cameras[topic] = dict(info=None, frame=frame, frames=0, tags=[])
        self.panel_pose = None          # 4x4, planning frame from panel
        self.panel_stamp = None
        self.lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        for topic, info_topic in zip(get('image_topics'), get('camera_info_topics')):
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, t=topic: self.cameras[t].__setitem__('info', msg),
                qos_profile_sensor_data)
            self.create_subscription(
                Image, topic, lambda msg, t=topic: self._image_cb(msg, t),
                qos_profile_sensor_data)
        self.get_logger().info(
            f"localising from {len(self.cameras)} cameras: "
            f"{', '.join(self.cameras)}")
        self.create_subscription(String, '/panel/operate', self._operate_cb, 1,
                                 callback_group=group)
        self.status_pub = self.create_publisher(String, '/panel/status', 10)
        self.gripper_pub = self.create_publisher(Float64, get('gripper_topic'), 10)

        self.create_timer(5.0, self._heartbeat)
        self.cartesian = self.create_client(
            GetCartesianPath, get('cartesian_service_name'), callback_group=group)
        self.execute = ActionClient(
            self, ExecuteTrajectory, get('execute_action_name'),
            callback_group=group)

    # ------------------------------------------------------------------ vision

    def _heartbeat(self):
        """Say what is going on, every few seconds.

        Without this the node is silent whenever it cannot see the panel, which
        is indistinguishable from it being broken - and "no tags in view" is by
        far the most likely reason a command appears to do nothing.
        """
        seen = {t: c for t, c in self.cameras.items() if c['frames']}
        if not seen:
            self.get_logger().warn(
                f"no images yet on any of {', '.join(self.cameras)}")
            return
        view = '; '.join(f"{t.split('/')[1]}: {c['tags'] or 'none'}"
                         for t, c in seen.items())
        with self.lock:
            have = self.panel_pose is not None
            age = (0.0 if not have else
                   (self.get_clock().now() - self.panel_stamp).nanoseconds / 1e9)
            source = getattr(self, 'panel_source', '?')
        if not have:
            self.get_logger().warn(
                f"panel NOT localised. Tags in view - {view}. Point a camera at "
                f"the panel; two of 11/13/14/15 must be visible to one of them.")
        elif age > self.get('pose_timeout_sec'):
            self.get_logger().info(
                f"panel localised by {source} (latched {age:.0f} s ago); ready "
                f"for /panel/operate. Tags in view now - {view}")

    def _image_cb(self, msg, topic):
        camera = self.cameras[topic]
        if camera['info'] is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = image if image.ndim == 2 else image[..., :3].mean(2).astype('uint8')
        detections = detect_markers(gray)
        camera['frames'] += 1
        camera['tags'] = sorted(detections)
        if not detections:
            return
        camera_from_panel, info = panel_pose_from_markers(
            detections, self.table,
            np.asarray(camera['info'].k, float).reshape(3, 3),
            np.asarray(camera['info'].d, float).ravel(),
            min_markers=1)
        if camera_from_panel is None:
            return
        if info['reprojection_px'] > self.get('max_reprojection_px'):
            self.get_logger().warn(
                f"{topic}: panel fit rejected, {info['reprojection_px']:.1f} px "
                f"over {len(info['markers'])} tags {info['markers']}",
                throttle_duration_sec=5.0)
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.get('planning_frame'), camera['frame'], rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warn(f"no transform for {camera['frame']}: {exc}",
                                   throttle_duration_sec=5.0)
            return

        base_from_camera = np.eye(4)
        t, q = tf.transform.translation, tf.transform.rotation
        base_from_camera[:3, :3] = self._matrix_from_quaternion(q)
        base_from_camera[:3, 3] = (t.x, t.y, t.z)
        quality = (len(info['markers']), -info['reprojection_px'])
        with self.lock:
            previous = getattr(self, 'panel_quality', None)
            fresh = (self.panel_stamp is None or
                     (self.get_clock().now() - self.panel_stamp).nanoseconds / 1e9
                     > self.get('pose_timeout_sec'))
            if previous is not None and not fresh and quality < previous:
                return          # a worse view of a panel we already have
            self.panel_quality = quality
            self.panel_source = topic
            first = self.panel_pose is None
            self.panel_pose = base_from_camera @ camera_from_panel
            self.panel_stamp = self.get_clock().now()
            self.panel_info = info
        if first:
            position = self.panel_pose[:3, 3]
            self.get_logger().info(
                f"panel localised by {topic} from tags {info['markers']} at "
                f"[{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}], "
                f"{info['reprojection_px']:.2f} px; pose latched")
        self._broadcast(self.panel_pose)

    @staticmethod
    def _matrix_from_quaternion(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def _broadcast(self, pose):
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get('planning_frame')
        msg.child_frame_id = 'maintenance_panel'
        msg.transform.translation.x = float(pose[0, 3])
        msg.transform.translation.y = float(pose[1, 3])
        msg.transform.translation.z = float(pose[2, 3])
        x, y, z, w = quaternion_from_matrix(pose)
        msg.transform.rotation.x, msg.transform.rotation.y = float(x), float(y)
        msg.transform.rotation.z, msg.transform.rotation.w = float(z), float(w)
        self.tf_broadcaster.sendTransform(msg)

    def _fresh_pose(self):
        with self.lock:
            if self.panel_pose is None:
                return None, ('panel has never been seen - aim the gripper '
                              'camera at it until "panel localised" is logged')
            age = (self.get_clock().now() - self.panel_stamp).nanoseconds / 1e9
            markers = len(self.panel_info['markers'])
            pose = self.panel_pose.copy()
        if markers < self.get('min_markers'):
            return None, (f'the latched pose rests on {markers} tag(s); a '
                          f"single tag's pose is ambiguous, so show it another")
        if age > self.get('pose_timeout_sec'):
            if self.get('require_fresh_pose'):
                return None, f'panel pose is {age:.1f} s stale'
            # Expected, not a fault: the tool occludes the tags on approach.
            self.get_logger().info(
                f'using the latched panel pose, {age:.1f} s old')
        return pose, None

    # ------------------------------------------------------------------ motion

    def _move_through(self, poses):
        """Cartesian move through `poses`, refusing an incomplete path."""
        request = GetCartesianPath.Request()
        request.header.frame_id = self.get('planning_frame')
        request.group_name = self.get('planning_group')
        request.link_name = self.get('tool_frame')
        request.waypoints = [_pose_msg(p) for p in poses]
        request.max_step = float(self.get('cartesian_step_m'))
        request.jump_threshold = float(self.get('cartesian_jump_threshold'))
        request.avoid_collisions = True
        if not self.cartesian.wait_for_service(timeout_sec=5.0):
            return False, 'no /compute_cartesian_path'
        response = self.cartesian.call(request)
        if response is None:
            return False, 'cartesian service call failed'
        if response.fraction < self.get('min_cartesian_fraction'):
            # Not a planner hiccup: a short fraction here means the straight
            # line is geometrically blocked or out of reach, and executing the
            # prefix would park the tool somewhere arbitrary.
            return False, f'cartesian fraction {response.fraction:.2f}'
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        if not self.execute.wait_for_server(timeout_sec=5.0):
            return False, 'no /execute_trajectory'
        handle = self.execute.send_goal(goal)
        if not handle.accepted:
            return False, 'trajectory rejected'
        result = handle.get_result()
        ok = result.result.error_code.val == 1
        return ok, '' if ok else f'execute error {result.result.error_code.val}'

    def _gripper(self, width):
        self.gripper_pub.publish(Float64(data=float(width)))
        self.get_clock().sleep_for(
            rclpy.duration.Duration(seconds=float(self.get('gripper_settle_sec'))))

    def operate(self, name):
        control = self.controls.get(name)
        if control is None:
            return False, f'unknown control {name!r}'
        base_from_panel, why = self._fresh_pose()
        if base_from_panel is None:
            return False, why

        way = control_waypoints(control, self.table)
        approach = base_from_panel @ way['approach']
        contact = base_from_panel @ way['contact']
        operate = base_from_panel @ way['operate']

        # Jaws are set BEFORE going in: opening or closing them against the
        # console is how you shear a fingertip off a control.
        self._gripper(self.get('gripper_open_width') if way['grip']
                      else self.get('gripper_close_width'))

        ok, why = self._move_through([approach, contact])
        if not ok:
            return False, f'approach: {why}'

        if way['action'] == 'turn':
            self._gripper(self.get('gripper_close_width'))
            steps = []
            for k in range(1, 9):
                angle = way['turn_about_approach'] * k / 8.0
                spun = contact.copy()
                spun[:3, :3] = _rotation_about(contact[:3, 2], angle) @ contact[:3, :3]
                steps.append(spun)
            ok, why = self._move_through(steps)
            if ok:
                self._gripper(self.get('gripper_open_width'))
        else:
            ok, why = self._move_through([operate])
        if not ok:
            return False, f'{way["action"]}: {why}'

        # Retrace the way in rather than lifting: a world-vertical retreat from
        # a control that is still gripped or still in its detent pries at it.
        back, _ = self._move_through([approach])
        return True, f'{name} operated' + ('' if back else ' (retract failed)')

    # ------------------------------------------------------------------ command

    def _operate_cb(self, msg):
        target = msg.data.strip()
        groups = {
            'all_breakers': [c['name'] for c in self.table['controls']
                             if c['action'] == 'flick'],
            'all_buttons': [c['name'] for c in self.table['controls']
                            if c['action'] == 'press'],
            'all_switches': [c['name'] for c in self.table['controls']
                             if c['action'] == 'turn'],
        }
        names = groups.get(target, [target])
        for name in names:
            self.get_logger().info(f'[{name}] starting')
            ok, detail = self.operate(name)
            level = self.get_logger().info if ok else self.get_logger().error
            level(f'[{name}] {detail}')
            self.status_pub.publish(String(data=f'{name}: {"ok" if ok else detail}'))
            if not ok and len(names) > 1:
                self.get_logger().error('stopping the sequence on first failure')
                return


def main():
    rclpy.init()
    node = PanelOperator()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
