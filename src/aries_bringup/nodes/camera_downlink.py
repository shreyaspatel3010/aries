#!/usr/bin/env python3
"""Rover-side reducer for the operator camera downlink.

The problem this exists to solve: moveit.rviz keeps four camera displays live
(colour + aligned depth on both D435is). Subscribed raw, that is

    2 x 640*480*3*15*8 = 221 Mbit/s of colour
    2 x 640*480*2*15*8 = 147 Mbit/s of depth
                       = 369 Mbit/s

which is one to two orders of magnitude past what the antenna link carries. On
top of the volume, every raw frame is far bigger than a UDP datagram, so each
one is fragmented; lose a single fragment and the whole sample is discarded, and
under Reliable QoS the writer then retransmits into a link that is already
saturated. That is the "lag" -- it is congestion collapse, not slow rendering.

This node produces a second, view-only stream sized for the link. It never
touches the topics the grasp/maintenance pipelines read: those stay at full rate
and full resolution on the rover, where they cost nothing. Reduction happens in
this order, cheapest first:

  1. rate     -- drop whole frames before doing any pixel work
  2. spatial  -- integer decimation, with the intrinsics rescaled to match
  3. range    -- depth outside [depth_min_m, depth_max_m] becomes 0 (invalid)
  4. quantise -- depth rounded to a step, which is what actually makes the PNG
                 small; raw sensor depth is noisy in its low bits and noise is
                 incompressible

Compression itself is deliberately NOT done here. image_transport's C++
plugins do it, driven off the topics this node publishes -- see
camera_downlink.launch.py. That keeps the codecs out of Python (cv_bridge is
banned under NumPy 2.x here) and gets the encoding for free only when someone is
actually subscribed.

Colour and depth are synchronised before they are emitted and re-stamped
identically, so RViz's DepthCloud ApproximateTime sync always finds its pair --
gating the two streams independently would let them drift a frame apart and the
cloud would stall.

  IN   <camera>/color/image_raw                     (full rate, rover-local)
       <camera>/aligned_depth_to_color/image_raw    (full rate, rover-local)
       <camera>/color/camera_info
  OUT  <output_ns>/color_src        reduced rgb8
       <output_ns>/depth_src        reduced 16UC1, millimetres
       <output_ns>/camera_info      intrinsics rescaled to the reduced size

COLOUR-ONLY CAMERAS
    Set depth_topic to the empty string and this runs without depth at all: one
    plain colour subscription, no synchroniser, and no depth publisher. That is
    the mode the rear camera uses -- a Logitech Brio 100 under the tail aimed at
    the drill, which is a UVC webcam and has no depth sensor to pair with.

    It is not a degraded version of the paired mode, it is a different one. The
    synchroniser is the whole reason the paired mode exists: it guarantees
    colour and depth leave here on one stamp so RViz's DepthCloud can match
    them. With no depth there is nothing to match, and feeding a single stream
    through a two-input ApproximateTimeSynchronizer would simply never fire --
    the node would sit there publishing nothing while both topics looked
    healthy. depth_rate_hz, depth_min_m, depth_max_m and
    depth_quantization_mm are all inert here.

    The topic names differ too, and the launch file passes them in: usb_cam
    publishes <camera>/image_raw and <camera>/camera_info, with no /color/
    segment for a colour stream that is the only stream there is.
"""

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from aries_common.image_bridge import NumpyImageBridge


# Publish Reliable, not SensorData. These topics are consumed on the rover by
# image_transport's republish nodes, whose subscriptions are Reliable: a
# best-effort writer would be QoS-incompatible with them and silently deliver
# nothing. Reliable costs nothing over loopback. What crosses the antenna is the
# republishers' compressed output, and that is where best-effort belongs.
def _local_qos(depth=1):
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=DurabilityPolicy.VOLATILE,
        depth=depth,
    )


class CameraDownlink(Node):

    def __init__(self):
        super().__init__('camera_downlink')

        camera = self._param('camera', 'gripper_camera').strip('/')
        color_topic = self._param('color_topic', f'/{camera}/color/image_raw')
        # An empty depth topic is the switch, not a missing value: it says this
        # camera HAS no depth sensor. See the module docstring.
        depth_topic = self._param(
            'depth_topic', f'/{camera}/aligned_depth_to_color/image_raw').strip()
        self.color_only = not depth_topic
        info_topic = self._param('camera_info_topic', f'/{camera}/color/camera_info')
        # Two namespaces on purpose, because they have opposite audiences.
        #
        # output_ns holds the reduced RAW frames, which exist only so the
        # republisher next door has something to encode. They are full-size
        # uncompressed images and must never be subscribed across the link.
        #
        # link_ns holds what actually crosses: the compressed streams and the
        # intrinsics beside them. Keeping it a single top-level prefix means
        # the operator can list exactly what the antenna carries --
        # `ros2 topic list | grep ^/downlink/` -- instead of picking it out of
        # the camera's own tree, where every raw topic also advertises four
        # image_transport codec names that nothing publishes.
        out_ns = self._param('output_ns', f'/{camera}/downlink_src').rstrip('/')
        link_ns = self._param('link_ns', f'/downlink/{camera}').rstrip('/')

        self.rate_hz = float(self._param('rate_hz', 15.0))
        # Colour and depth do not need the same rate, and splitting them is what
        # buys full resolution. A 640x480 depth frame is ~91 kB against ~98 kB of
        # colour, so running depth at a third of the rate frees roughly a third
        # of the budget while the image the operator drives by stays at the full
        # 15 Hz. Raise this to 15.0 for an equally smooth cloud, at about 35%
        # more bandwidth.
        self.depth_rate_hz = float(self._param('depth_rate_hz', 5.0))
        # Full sensor resolution by default. Resolution is what the operator
        # actually perceives as quality -- 640x480 at JPEG q80 reads better than
        # 320x240 at q95 and costs about the same, because halving resolution
        # throws away detail no quality setting can put back.
        self.decimation = max(1, int(self._param('decimation', 1)))
        self.depth_min_mm = int(round(float(self._param('depth_min_m', 0.15)) * 1000.0))
        self.depth_max_mm = int(round(float(self._param('depth_max_m', 6.0)) * 1000.0))
        self.depth_step_mm = max(0, int(self._param('depth_quantization_mm', 10)))
        sync_slop = float(self._param('sync_slop_s', 0.05))
        queue = max(1, int(self._param('queue_size', 5)))

        self.bridge = NumpyImageBridge()
        self.info = None
        self._scaled_info = None
        self._min_period = 0.0 if self.rate_hz <= 0.0 else 1.0 / self.rate_hz
        self._next_due = None
        self._last_arrival = None
        self._arrival_dt = None
        # Send depth on every Nth colour frame rather than running a second
        # independent gate. A separate gate would drift out of phase and emit
        # depth with a stamp no colour frame shares, which is exactly the pairing
        # RViz's DepthCloud needs; a counter keeps every depth frame on a stamp
        # that has a colour frame beside it.
        self._depth_every = 1
        if self.depth_rate_hz > 0.0 and self.rate_hz > self.depth_rate_hz:
            self._depth_every = max(1, int(round(self.rate_hz / self.depth_rate_hz)))
        self._frame = 0
        self._sent = 0
        self._seen = 0

        self.pub_color = self.create_publisher(Image, f'{out_ns}/color', _local_qos())
        # No depth publisher on a colour-only camera. Advertising one would put
        # an empty topic in front of the republisher next door, which would sit
        # warning about a stream that is never going to arrive -- the same
        # failure aries_hardware.launch.py avoids by not starting a downlink for
        # an absent camera.
        self.pub_depth = None if self.color_only else \
            self.create_publisher(Image, f'{out_ns}/depth', _local_qos())
        # CameraInfo is tiny and the viewer needs it once; keep it latched so an
        # RViz started mid-run gets intrinsics without waiting for the next frame.
        self.pub_info = self.create_publisher(
            CameraInfo, f'{link_ns}/camera_info',
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       depth=1))

        # Subscribe best-effort: the RealSense driver publishes sensor-data QoS.
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST,
                                durability=DurabilityPolicy.VOLATILE,
                                depth=queue)
        self.create_subscription(CameraInfo, info_topic, self._on_info, sensor_qos)
        if self.color_only:
            # A plain subscription, deliberately not a one-input synchroniser.
            self.sync = None
            self.create_subscription(Image, color_topic, self._on_color, sensor_qos)
        else:
            self.sync = ApproximateTimeSynchronizer(
                [Subscriber(self, Image, color_topic, qos_profile=sensor_qos),
                 Subscriber(self, Image, depth_topic, qos_profile=sensor_qos)],
                queue, sync_slop)
            self.sync.registerCallback(self._on_pair)

        self.create_timer(10.0, self._report)
        if self.color_only:
            self.get_logger().info(
                f'downlink {camera}: {color_topic} -> {link_ns}/* at '
                f'{self.rate_hz:g} Hz colour, 1/{self.decimation} scale '
                '(colour-only camera, no depth)')
        else:
            self.get_logger().info(
                f'downlink {camera}: {color_topic} + {depth_topic} -> {link_ns}/* '
                f'at {self.rate_hz:g} Hz colour / {self.rate_hz / self._depth_every:g} Hz '
                f'depth, 1/{self.decimation} scale, depth '
                f'{self.depth_min_mm}-{self.depth_max_mm} mm'
                + (f' rounded to {self.depth_step_mm} mm' if self.depth_step_mm > 1 else ''))

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _on_info(self, msg):
        if self.info is None or msg.width != self.info.width or msg.height != self.info.height:
            self._scaled_info = None
        self.info = msg

    def _on_color(self, color_msg):
        """Colour-only cameras. Same gate and same decimation, no depth."""
        self._seen += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self._gate_open(now):
            return
        self._frame += 1

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, 'rgb8')
        except ValueError as exc:
            self.get_logger().warn(f'unsupported encoding, skipping frame: {exc}',
                                   throttle_duration_sec=10.0)
            return

        color = self._reduce_color(color)

        stamp = color_msg.header.stamp
        out_color = self.bridge.cv2_to_imgmsg(color, 'rgb8')
        out_color.header.stamp = stamp
        out_color.header.frame_id = color_msg.header.frame_id
        self.pub_color.publish(out_color)

        # Sized off the colour image here. In the paired mode the depth image is
        # what gets unprojected, so its dimensions are the ones that matter;
        # with no depth, colour is the only image there is.
        info = self._scaled_camera_info(color.shape[1], color.shape[0])
        if info is not None:
            info.header.stamp = stamp
            info.header.frame_id = color_msg.header.frame_id
            self.pub_info.publish(info)
        self._sent += 1

    def _on_pair(self, color_msg, depth_msg):
        self._seen += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self._gate_open(now):
            return

        self._frame += 1
        send_depth = (self._frame % self._depth_every) == 0

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, 'rgb8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
        except ValueError as exc:
            self.get_logger().warn(f'unsupported encoding, skipping frame: {exc}',
                                   throttle_duration_sec=10.0)
            return
        depth = self._depth_to_mm(depth)
        if depth is None:
            self.get_logger().warn(
                f'depth must be 16UC1 millimetres or 32FC1 metres, got '
                f'{depth_msg.encoding}; skipping',
                throttle_duration_sec=10.0)
            return

        color, depth = self._reduce(color, depth)

        # One stamp for all three messages. RViz pairs colour and depth with an
        # ApproximateTime sync; identical stamps make that pairing exact instead
        # of merely probable, which matters once the link starts dropping frames.
        stamp = color_msg.header.stamp
        out_color = self.bridge.cv2_to_imgmsg(color, 'rgb8')
        out_color.header.stamp = stamp
        out_color.header.frame_id = color_msg.header.frame_id
        self.pub_color.publish(out_color)

        # Serialising the depth image copies it; skip that on the frames the
        # depth rate is dropping rather than building a message to discard.
        if send_depth:
            out_depth = self.bridge.cv2_to_imgmsg(depth, '16UC1')
            out_depth.header.stamp = stamp
            out_depth.header.frame_id = depth_msg.header.frame_id
            self.pub_depth.publish(out_depth)

        info = self._scaled_camera_info(depth.shape[1], depth.shape[0])
        if info is not None:
            info.header.stamp = stamp
            info.header.frame_id = depth_msg.header.frame_id
            self.pub_info.publish(info)
        self._sent += 1

    def _depth_to_mm(self, depth):
        """Depth as uint16 millimetres, the unit everything below here works in.

        The D435i already publishes 16UC1 millimetres. GAZEBO DOES NOT: its
        depth camera publishes 32FC1 METRES, and the bridge hands that straight
        through. Rejecting it dropped the whole synchronised pair, colour
        included, so in simulation every frame was skipped and the operator
        view stayed blank - which is exactly how it looked in RViz.

        Non-finite pixels are folded onto 0 BEFORE the cast, not after: gz
        writes +Inf where the ray hit nothing, and casting that to uint16 is
        undefined - in practice it lands on 65535, a 65 m wall of phantom
        surface. 0 is the "no reading" value _reduce and RViz's DepthCloud both
        already skip.
        """
        if depth.dtype == np.uint16:
            return depth
        if depth.dtype.kind != 'f':
            return None
        # A new array: `depth` may be a view onto the incoming message buffer.
        mm = np.multiply(depth, 1000.0, dtype=np.float32)
        np.nan_to_num(mm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(mm, 0.0, 65535.0, out=mm)
        return mm.astype(np.uint16)

    def _gate_open(self, now):
        """Rate limiter that lands on the requested rate, not below it.

        A naive "has min_period elapsed?" test cannot hit 5 Hz from a 15 Hz
        camera: frames arrive at multiples of 66.7 ms, and the first one at or
        past 200 ms is the one at 200.0 ms exactly. Any jitter puts it a hair
        early, it gets rejected, and the next candidate is 266.7 ms -- so the
        stream silently settles at 3.75 Hz, a quarter under target, for every
        rate that is not an exact divisor of the input.

        Instead, keep a deadline and admit a frame that is within half an
        inter-arrival gap of it, since no closer frame is coming. The input
        period is measured rather than assumed: it is a launch argument on the
        driver, and it changes between simulation and hardware.
        """
        if self._min_period <= 0.0:
            return True

        if self._last_arrival is not None:
            dt = now - self._last_arrival
            if 0.0 < dt < 1.0:  # ignore restarts and long stalls
                self._arrival_dt = dt if self._arrival_dt is None \
                    else 0.8 * self._arrival_dt + 0.2 * dt
        self._last_arrival = now

        tolerance = 0.5 * self._arrival_dt if self._arrival_dt else 0.0
        if self._next_due is not None and now + tolerance < self._next_due:
            return False
        # Advance from the deadline, not from now, so rounding does not
        # accumulate into a slow drift below the requested rate. Falling behind
        # (a stalled camera) resyncs to now instead of firing a catch-up burst.
        base = now if self._next_due is None else max(now, self._next_due)
        self._next_due = base + self._min_period
        return True

    def _reduce_color(self, color):
        """Decimate one colour image. The colour half of _reduce, on its own.

        Averaging over the block rather than sampling one pixel: colour has no
        silhouette problem (see _reduce) and the mean is the better downsample.
        """
        d = self.decimation
        h = color.shape[0] - color.shape[0] % d
        w = color.shape[1] - color.shape[1] % d
        color = color[:h, :w]
        if d > 1:
            color = color.reshape(h // d, d, w // d, d, 3).mean(
                axis=(1, 3), dtype=np.float32).astype(np.uint8)
        return color

    def _reduce(self, color, depth):
        d = self.decimation
        h = min(color.shape[0], depth.shape[0])
        w = min(color.shape[1], depth.shape[1])
        h -= h % d
        w -= w % d
        color = color[:h, :w]
        depth = depth[:h, :w]

        if d > 1:
            # Colour averages over the block; depth must not. Averaging depth
            # across a silhouette edge invents a surface at the mean of
            # foreground and background, hanging points in empty space. Take a
            # representative pixel instead. (The two sample points differ by
            # (d-1)/2 px, i.e. half a pixel at d=2 -- invisible in a view stream.)
            color = color.reshape(h // d, d, w // d, d, 3).mean(
                axis=(1, 3), dtype=np.float32).astype(np.uint8)
            depth = np.ascontiguousarray(depth[::d, ::d])

        # 0 is the RealSense "no reading" value and DepthCloud already skips it,
        # so clamping out-of-range to 0 both drops points the operator cannot use
        # and removes the noisiest, least compressible part of the image.
        depth = depth.copy()
        depth[(depth < self.depth_min_mm) | (depth > self.depth_max_mm)] = 0

        if self.depth_step_mm > 1:
            # The low bits of RealSense depth are sensor noise: incompressible,
            # and they dominate the PNG. Rounding them off is most of the size
            # win. Integer division keeps 0 at 0.
            np.floor_divide(depth, self.depth_step_mm, out=depth)
            depth *= self.depth_step_mm

        return color, depth

    def _scaled_camera_info(self, width, height):
        """Intrinsics for the reduced image. DepthCloud unprojects with these,
        so they must describe the image actually sent, not the sensor's."""
        if self.info is None:
            return None
        if self._scaled_info is not None and \
                self._scaled_info.width == width and self._scaled_info.height == height:
            return self._scaled_info

        src = self.info
        sx = width / float(src.width) if src.width else 1.0
        sy = height / float(src.height) if src.height else 1.0
        info = CameraInfo()
        info.height = height
        info.width = width
        info.distortion_model = src.distortion_model
        info.d = list(src.d)
        k = list(src.k)
        k[0] *= sx  # fx
        k[2] *= sx  # cx
        k[4] *= sy  # fy
        k[5] *= sy  # cy
        info.k = k
        info.r = list(src.r)
        p = list(src.p)
        p[0] *= sx  # fx
        p[2] *= sx  # cx
        p[3] *= sx  # Tx, carried in pixel units
        p[5] *= sy  # fy
        p[6] *= sy  # cy
        info.p = p
        info.binning_x = src.binning_x
        info.binning_y = src.binning_y
        self._scaled_info = info
        return info

    def _report(self):
        if self._seen == 0:
            # align_depth is not the advice to give for a camera that has no
            # depth to align; the useful check there is whether the driver came
            # up on the device at all.
            self.get_logger().warn(
                'no colour frames yet -- check the camera is up'
                if self.color_only else
                'no synchronised colour+depth pairs yet -- check the camera is up '
                'and that align_depth.enable is true', throttle_duration_sec=30.0)
            return
        kind = 'frames' if self.color_only else 'pairs'
        self.get_logger().debug(f'downlink forwarded {self._sent}/{self._seen} {kind}')


def main():
    rclpy.init()
    node = CameraDownlink()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
