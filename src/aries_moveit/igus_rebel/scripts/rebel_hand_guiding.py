#!/usr/bin/env python3
"""
Safely coordinate Rebel ZeroTorque mode with ros2_control.

The public service stops the arm trajectory controller before requesting
ZeroTorque from the hardware plugin.  On exit it re-enables the motors first,
then starts a fresh controller instance so no pre-guiding trajectory resumes.
"""

import threading
import time

from controller_manager_msgs.srv import SwitchController
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


class RebelHandGuiding(Node):
    def __init__(self) -> None:
        super().__init__('rebel_hand_guiding')
        self.declare_parameter('arm_controller', 'rebel_arm_trajectory_controller')
        self.declare_parameter(
            'controller_manager_service', '/controller_manager/switch_controller'
        )
        self.declare_parameter('hardware_service', '/igus_rebel/set_hand_guiding')
        self.declare_parameter('service_timeout', 3.0)
        self.declare_parameter('motor_settle_time', 0.5)
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('hand_guiding_button', 3)
        self.declare_parameter('joy_timeout', 2.0)

        self._arm_controller = str(self.get_parameter('arm_controller').value)
        self._service_timeout = float(self.get_parameter('service_timeout').value)
        self._motor_settle_time = float(self.get_parameter('motor_settle_time').value)
        self._hand_guiding_button = int(
            self.get_parameter('hand_guiding_button').value
        )
        self._joy_timeout = float(self.get_parameter('joy_timeout').value)
        callback_group = ReentrantCallbackGroup()

        self._switch_client = self.create_client(
            SwitchController,
            str(self.get_parameter('controller_manager_service').value),
            callback_group=callback_group,
        )
        self._hardware_client = self.create_client(
            SetBool,
            str(self.get_parameter('hardware_service').value),
            callback_group=callback_group,
        )
        self._service = self.create_service(
            SetBool,
            '/arm/set_hand_guiding',
            self._handle_request,
            callback_group=callback_group,
        )
        self._status_pub = self.create_publisher(
            Bool,
            '/arm/hand_guiding_active',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._transition_lock = threading.Lock()
        self._joy_lock = threading.Lock()
        self._joy_pressed = False
        self._joy_generation = 0
        self._joy_worker_running = False
        self._last_joy_message = time.monotonic()
        self._active = False
        self._joy_subscription = self.create_subscription(
            Joy,
            str(self.get_parameter('joy_topic').value),
            self._joy_callback,
            10,
        )
        self._joy_watchdog = self.create_timer(0.1, self._check_joy_timeout)
        self._publish_status()
        self.get_logger().info(
            'Hand guiding ready: hold Y/button '
            f'{self._hand_guiding_button}, or use /arm/set_hand_guiding'
        )

    def _publish_status(self) -> None:
        msg = Bool()
        msg.data = self._active
        self._status_pub.publish(msg)

    def _wait_for_service(self, client, name: str) -> bool:
        if client.wait_for_service(timeout_sec=self._service_timeout):
            return True
        self.get_logger().error(f'Timed out waiting for {name}')
        return False

    def _call(self, client, request, name: str):
        if not self._wait_for_service(client, name):
            return None
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(self._service_timeout):
            self.get_logger().error(f'Timed out calling {name}')
            return None
        try:
            return future.result()
        except Exception as exc:  # rclpy transports service exceptions via Future
            self.get_logger().error(f'{name} failed: {exc}')
            return None

    def _switch_controller(self, activate: bool) -> tuple[bool, str]:
        request = SwitchController.Request()
        if activate:
            request.activate_controllers = [self._arm_controller]
        else:
            request.deactivate_controllers = [self._arm_controller]
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout.sec = int(self._service_timeout)
        response = self._call(
            self._switch_client, request, '/controller_manager/switch_controller'
        )
        if response is None:
            return False, 'controller manager did not respond'
        return bool(response.ok), response.message

    def _set_hardware_mode(self, enabled: bool) -> tuple[bool, str]:
        request = SetBool.Request()
        request.data = enabled
        response = self._call(
            self._hardware_client, request, '/igus_rebel/set_hand_guiding'
        )
        if response is None:
            return False, 'Rebel hardware service did not respond'
        return bool(response.success), response.message

    def _transition(self, enabled: bool) -> tuple[bool, str]:
        with self._transition_lock:
            if enabled:
                switched, detail = self._switch_controller(activate=False)
                if not switched:
                    return False, f'Could not stop arm controller: {detail}'

                hardware_enabled, detail = self._set_hardware_mode(enabled=True)
                if not hardware_enabled:
                    # Deliberately leave the trajectory controller inactive.  A
                    # timed-out ZeroTorque request may still have reached the robot.
                    # If the Rebel explicitly confirmed that ZeroTorque is inactive,
                    # however, it is safe to restore the controller immediately.
                    confirmed_inactive = detail.startswith(
                        ('ZeroTorque unavailable', 'ZeroTorque allowed but')
                    )
                    if confirmed_inactive:
                        restored, restore_detail = self._switch_controller(activate=True)
                        if restored:
                            return False, (
                                f'ZeroTorque was not activated; normal control restored. {detail}'
                            )
                        detail = (
                            f'{detail}; normal controller could not be restored: '
                            f'{restore_detail}'
                        )
                    return False, f'Arm controller is stopped; {detail}'

                self._active = True
                self._publish_status()
                message = (
                    'Hand guiding active. Support the arm and move it slowly by hand.'
                )
                self.get_logger().warn(message)
                return True, message

            disabled, detail = self._set_hardware_mode(enabled=False)
            if not disabled:
                return False, f'Arm controller remains stopped; {detail}'

            # This status describes ZeroTorque, independently of whether the ROS
            # trajectory controller can be restarted below.
            self._active = False
            self._publish_status()

            # Give the drives time to transition from Disabled to Enabled before
            # ros2_control claims the velocity interfaces again.
            time.sleep(self._motor_settle_time)
            switched, detail = self._switch_controller(activate=True)
            if not switched:
                return False, (
                    'ZeroTorque is off and motors are enabled, but the arm controller '
                    f'could not restart: {detail}'
                )

            message = 'Hand guiding stopped; normal arm control restored'
            self.get_logger().info(message)
            return True, message

    def _handle_request(self, request: SetBool.Request, response: SetBool.Response):
        response.success, response.message = self._transition(request.data)
        return response

    def _joy_callback(self, msg: Joy) -> None:
        pressed = (
            0 <= self._hand_guiding_button < len(msg.buttons)
            and bool(msg.buttons[self._hand_guiding_button])
        )
        start_worker = False
        with self._joy_lock:
            self._last_joy_message = time.monotonic()
            if pressed == self._joy_pressed:
                return
            self._joy_pressed = pressed
            self._joy_generation += 1
            if not self._joy_worker_running:
                self._joy_worker_running = True
                start_worker = True

        if start_worker:
            threading.Thread(target=self._joy_worker, daemon=True).start()

    def _check_joy_timeout(self) -> None:
        if self._joy_timeout <= 0.0:
            return

        start_worker = False
        with self._joy_lock:
            timed_out = (
                self._joy_pressed
                and time.monotonic() - self._last_joy_message > self._joy_timeout
            )
            if not timed_out:
                return
            self._joy_pressed = False
            self._joy_generation += 1
            if not self._joy_worker_running:
                self._joy_worker_running = True
                start_worker = True

        self.get_logger().error(
            'Joystick messages timed out while Y was held; restoring arm torque'
        )
        if start_worker:
            threading.Thread(target=self._joy_worker, daemon=True).start()

    def _joy_worker(self) -> None:
        while rclpy.ok():
            with self._joy_lock:
                generation = self._joy_generation
                requested_state = self._joy_pressed

            self.get_logger().info(
                'Y pressed: requesting ZeroTorque'
                if requested_state
                else 'Y released: restoring normal arm control'
            )
            success, message = self._transition(requested_state)
            if not success:
                self.get_logger().error(f'Y hand-guiding transition failed: {message}')

            with self._joy_lock:
                if generation == self._joy_generation:
                    self._joy_worker_running = False
                    return

        with self._joy_lock:
            self._joy_worker_running = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RebelHandGuiding()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
