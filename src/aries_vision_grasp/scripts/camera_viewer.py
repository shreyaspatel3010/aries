#!/usr/bin/env python3
"""
Simple camera viewer for testing gripper camera feed.
Displays color image, depth image, and basic info.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from aries_vision_grasp.image_bridge import NumpyImageBridge
import cv2
import numpy as np


class CameraViewerNode(Node):
    def __init__(self):
        super().__init__('camera_viewer_node')

        self.bridge = NumpyImageBridge()
        self.latest_color = None
        self.latest_depth = None
        self.camera_info = None

        # Subscribers (best-effort, newest frame only — it's a live viewer)
        self.color_sub = self.create_subscription(
            Image,
            '/gripper_camera/color/image_raw',
            self.color_callback,
            qos_profile_sensor_data,
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/gripper_camera/depth/image_rect_raw',
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/gripper_camera/depth/camera_info',
            self.info_callback,
            qos_profile_sensor_data,
        )

        # Timer for display
        self.timer = self.create_timer(0.1, self.display_callback)

        self.get_logger().info('Camera Viewer started')
        self.get_logger().info('Press "q" to quit')

    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Color conversion error: {e}')

    def depth_callback(self, msg):
        try:
            if msg.encoding == '32FC1':
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                depth_mm = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.latest_depth = depth_mm.astype(np.float32) / 1000.0
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Camera Info:')
            self.get_logger().info(f'  Resolution: {msg.width}x{msg.height}')
            self.get_logger().info(f'  FX: {msg.k[0]:.2f}, FY: {msg.k[4]:.2f}')
            self.get_logger().info(f'  CX: {msg.k[2]:.2f}, CY: {msg.k[5]:.2f}')

    def display_callback(self):
        if self.latest_color is not None:
            # Add info overlay
            color_display = self.latest_color.copy()
            cv2.putText(
                color_display,
                'Gripper Camera - Color',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow('Gripper Color', color_display)

        if self.latest_depth is not None:
            depth_valid = self.latest_depth.copy()

            valid_mask = ~np.isnan(depth_valid) & ~np.isinf(depth_valid) & (depth_valid > 0)
            if np.any(valid_mask):
                min_depth = float(np.min(depth_valid[valid_mask]))
                max_depth = float(np.max(depth_valid[valid_mask]))
                depth_valid[~valid_mask] = max_depth

                # Normalize to the actual valid range so a 0.1–2 m workspace
                # uses the full colormap instead of a fixed 0–10 m slice.
                span = max(max_depth - min_depth, 1e-6)
                depth_normalized = (
                    np.clip((depth_valid - min_depth) / span, 0.0, 1.0) * 255
                ).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                # Add center crosshair and depth value
                h, w = depth_colored.shape[:2]
                center_x, center_y = w // 2, h // 2
                cv2.drawMarker(depth_colored, (center_x, center_y), (0, 255, 0),
                              cv2.MARKER_CROSS, 20, 2)

                # Get depth at center
                if 0 <= center_y < self.latest_depth.shape[0] and 0 <= center_x < self.latest_depth.shape[1]:
                    center_depth = self.latest_depth[center_y, center_x]
                    if not np.isnan(center_depth) and not np.isinf(center_depth):
                        cv2.putText(
                            depth_colored,
                            f'Center Depth: {center_depth:.3f}m',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                # Show depth range
                cv2.putText(
                    depth_colored,
                    f'Range: {min_depth:.2f}m - {max_depth:.2f}m',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.imshow('Gripper Depth', depth_colored)
            else:
                # No valid depth data
                empty = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(empty, 'No valid depth data', (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Gripper Depth', empty)

        key = cv2.waitKey(1)
        if key == ord('q'):
            self.get_logger().info('Quitting...')
            # Raising SystemExit unwinds cleanly out of rclpy.spin();
            # calling rclpy.shutdown() mid-callback leaves an unhandled
            # ExternalShutdownException traceback on exit.
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewerNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
