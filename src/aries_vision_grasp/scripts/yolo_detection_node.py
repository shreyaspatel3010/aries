#!/usr/bin/env python3
"""
Minimal YOLO detection node — GPU-accelerated.

Topics
  IN  /gripper_camera/color/image_raw   (sensor_msgs/Image)
  OUT /gripper_camera/yolo/image_raw    (sensor_msgs/Image)  — annotated
  OUT /gripper_camera/yolo/detections   (std_msgs/String)    — JSON per frame

Parameters
  model_path           path to .pt weights  (default: packaged grasp model)
  confidence_threshold minimum confidence   (default: 0.50)
  input_topic          colour image topic   (default: /gripper_camera/color/image_raw)
  output_topic         annotated image      (default: /gripper_camera/yolo/image_raw)
  detections_topic     JSON detections      (default: derived from output_topic)
  device               torch device string  (default: cuda:0, CPU fallback)
  imgsz                inference image size (default: 640)
"""

import json
import threading
import time
import cv2
import numpy as np
import rclpy
from aries_vision_grasp.image_bridge import NumpyImageBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from aries_vision_grasp.inference import default_model_path, load_yolo_model


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        self.declare_parameter("model_path",           default_model_path())
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("input_topic",  "/gripper_camera/color/image_raw")
        self.declare_parameter("output_topic", "/gripper_camera/yolo/image_raw")
        self.declare_parameter("detections_topic", "")
        self.declare_parameter("device",       "cuda:0")
        self.declare_parameter("imgsz",        640)

        model_path = self.get_parameter("model_path").value
        self.conf  = self.get_parameter("confidence_threshold").value
        in_topic   = self.get_parameter("input_topic").value
        out_topic  = self.get_parameter("output_topic").value
        det_topic  = self.get_parameter("detections_topic").value
        self.dev   = self.get_parameter("device").value
        self.imgsz = self.get_parameter("imgsz").value

        if not det_topic:
            det_topic = out_topic.replace("image_raw", "detections")
            if det_topic == out_topic:
                # output_topic without an "image_raw" suffix would otherwise
                # silently publish two message types on one topic name.
                det_topic = out_topic.rstrip("/") + "/detections"

        self.get_logger().info(
            f"YOLO node starting | model={model_path} | device={self.dev} | imgsz={self.imgsz}"
        )

        # Fails loudly (raises) when the model cannot be loaded at all, so a
        # misconfigured node dies visibly instead of idling forever.
        self.model, self.dev = load_yolo_model(
            model_path, device=self.dev, imgsz=self.imgsz, logger=self.get_logger()
        )

        self.bridge   = NumpyImageBridge()
        self.img_pub  = self.create_publisher(Image,  out_topic,  1)
        self.det_pub  = self.create_publisher(String, det_topic,  1)

        # depth=1, best-effort: always get the newest frame, drop stale ones
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Image, in_topic, self._image_cb, qos)

        self._lock       = threading.Lock()
        self._latest_msg = None
        self._shutdown   = False
        self._worker     = threading.Thread(target=self._infer_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"yolo_detection_node ready\n"
            f"  subscribing : {in_topic}\n"
            f"  publishing  : {out_topic}\n"
            f"  detections  : {det_topic}"
        )

    def stop(self):
        self._shutdown = True

    def _image_cb(self, msg: Image):
        # Just store the latest frame; inference thread picks it up
        with self._lock:
            self._latest_msg = msg

    def _infer_loop(self):
        while rclpy.ok() and not self._shutdown:
            with self._lock:
                msg = self._latest_msg
                self._latest_msg = None
            if msg is None:
                time.sleep(0.005)
                continue

            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                self.get_logger().error(f"cv_bridge error: {e}", throttle_duration_sec=5.0)
                continue

            try:
                results = self.model(frame, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.dev)
            except Exception as e:
                self.get_logger().error(f"YOLO inference error: {e}", throttle_duration_sec=5.0)
                continue

            # Draw + publish — wrapped so a single bad frame can't kill the thread
            try:
                base     = frame.copy()
                overlay  = frame.copy()   # filled masks go here; blended once at the end
                dets     = []
                masks    = results[0].masks  # None if model has no seg head
                boxes    = results[0].boxes

                # Collect per-detection data first
                det_data = []
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    label  = self.model.names.get(cls_id, str(cls_id))
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    contours = []
                    if masks is not None and i < len(masks):
                        raw = masks[i].xy  # list of numpy polygon arrays
                        contours = [c.astype(np.int32).reshape(-1, 1, 2)
                                    for c in raw if len(c) >= 3]
                    det_data.append((label, conf, x1, y1, x2, y2, contours))
                    dets.append({"label": label, "confidence": round(conf, 3),
                                 "bbox_xyxy": [round(v, 1) for v in [x1, y1, x2, y2]]})

                # Fill all masks onto overlay in one pass, then blend once
                has_masks = any(d[6] for d in det_data)
                if has_masks:
                    for label, conf, x1, y1, x2, y2, contours in det_data:
                        for pts in contours:
                            cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    annotated = cv2.addWeighted(overlay, 0.4, base, 0.6, 0)
                else:
                    annotated = base

                # Draw outlines / bbox and labels on top of the blended image
                for label, conf, x1, y1, x2, y2, contours in det_data:
                    if contours:
                        cv2.polylines(annotated, contours, isClosed=True,
                                      color=(0, 255, 0), thickness=2)
                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}",
                                (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)

                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out_msg.header = msg.header
                self.img_pub.publish(out_msg)
                self.det_pub.publish(String(data=json.dumps(dets)))
            except Exception as e:
                self.get_logger().error(f"Draw/publish error: {e}", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
