#!/usr/bin/env python3
"""Publish the onboard computer's own health on ``/compute/status``.

The mission control dashboard has had a Compute panel since it was written and
nothing has ever fed it, so it has been drawing generated numbers that move like
real ones. This is the publisher that makes it real.

WIRE FORMAT: a ``std_msgs/String`` carrying JSON, because that is what the
dashboard's ``rosTopics.ts`` parses for this topic. Not a nicety -- the keys
below are the field names in its ``ComputeData`` interface and a rename here
silently empties the panel:

    {"cpuUsage": 23.4, "cpuTemp": 51.2, "gpuUsage": 8.0,
     "gpuTemp": 44.0, "ramUsage": 11.7, "storageUsage": 0.42}

    cpuUsage      percent, all cores           cpuTemp   degrees C, package
    gpuUsage      percent                      gpuTemp   degrees C
    ramUsage      GIGABYTES USED, not percent  storageUsage  TERABYTES USED

The two "usage" figures are absolute, not fractions, because the panel prints
them against fixed labels ("SYS RAM (32GB)", "NVME (2TB)") and does its own
proportion.

A METRIC THIS MACHINE CANNOT MEASURE IS PUBLISHED AS ``null``, NEVER AS 0.
The dashboard merges each message over the state it already holds, so a field
left OUT of the JSON keeps whatever was there before -- which on first connect
is the mock. Omitting the GPU on a machine with no NVIDIA card would therefore
leave a plausible fabricated GPU load on screen and nothing to say so. An
explicit null is read back as NaN and rendered as "--".

Started by full_hardware.launch.py. By hand:

    ros2 run aries_bringup compute_publisher.py
    ros2 run aries_bringup compute_publisher.py --ros-args -p publish_rate_hz:=0.5
"""

from __future__ import annotations

import json
import shutil
import subprocess

import psutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Order matters: the first label present wins. "Package id 0" is the die
# temperature on Intel; k10temp/Tctl is AMD's equivalent; acpitz is a
# motherboard sensor and a poor last resort, but a real reading either way.
CPU_TEMP_SOURCES = (
    ("coretemp", "Package id 0"),
    ("k10temp", "Tctl"),
    ("zenpower", "Tdie"),
    ("acpitz", None),
)

BYTES_PER_GB = 1024 ** 3
BYTES_PER_TB = 1024 ** 4

# nvidia-smi is a process launch, not a register read: it costs 50-200 ms. That
# is affordable once a second in a node that does nothing else, and would not be
# in one that did.
NVIDIA_SMI_TIMEOUT_S = 2.0

# Consecutive nvidia-smi failures after which it stops being called at all.
# A machine with no NVIDIA GPU fails every time, and launching a doomed
# subprocess once a second for a whole mission is a waste worth avoiding. The
# node keeps publishing nulls for the GPU, so the panel still says so.
GPU_FAILURE_LIMIT = 3


class ComputePublisher(Node):
    """Sample this machine and publish it as the dashboard's JSON."""

    def __init__(self) -> None:
        super().__init__("compute_publisher")

        self.declare_parameter("status_topic", "/compute/status")
        self.declare_parameter("publish_rate_hz", 1.0)
        # Which filesystem "storage" means. The root filesystem by default,
        # which on this rover is the NVMe the panel is labelled for.
        self.declare_parameter("storage_path", "/")
        self.declare_parameter("enable_gpu", True)

        topic = str(self.get_parameter("status_topic").value)
        rate = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self.storage_path = str(self.get_parameter("storage_path").value)
        self.gpu_enabled = bool(self.get_parameter("enable_gpu").value) and (
            shutil.which("nvidia-smi") is not None
        )
        self.gpu_failures = 0

        self.publisher = self.create_publisher(String, topic, 10)

        # psutil.cpu_percent(interval=None) reports the load since the PREVIOUS
        # call, so the first one has no window to measure and always returns
        # 0.0. Priming it here means the first published sample is a real
        # measurement rather than a zero that looks like an idle machine.
        psutil.cpu_percent(interval=None)

        self.create_timer(1.0 / rate, self.publish_status)

        if not self.gpu_enabled:
            self.get_logger().info(
                "no nvidia-smi on PATH (or enable_gpu:=false) -- GPU fields "
                "will be published as null"
            )
        self.get_logger().info(
            f"compute status -> {topic} at {rate:g} Hz, storage={self.storage_path}"
        )

    # --- individual metrics ------------------------------------------------
    #
    # Each returns None rather than a guess. See the module docstring for why
    # that distinction has to survive all the way to the dashboard.

    def cpu_temperature(self) -> float | None:
        try:
            sensors = psutil.sensors_temperatures()
        except (AttributeError, OSError):
            # sensors_temperatures is Linux-only and can raise if there is no
            # hwmon at all.
            return None

        for chip, label in CPU_TEMP_SOURCES:
            readings = sensors.get(chip)
            if not readings:
                continue
            if label is None:
                return float(readings[0].current)
            for reading in readings:
                if reading.label == label:
                    return float(reading.current)
            # The chip is there but not the label this kernel names it by --
            # take its first sensor rather than skipping the chip entirely.
            return float(readings[0].current)
        return None

    def gpu_stats(self) -> tuple[float | None, float | None]:
        if not self.gpu_enabled:
            return None, None

        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=NVIDIA_SMI_TIMEOUT_S,
                check=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            self.gpu_failures += 1
            if self.gpu_failures >= GPU_FAILURE_LIMIT:
                self.gpu_enabled = False
                self.get_logger().warn(
                    f"nvidia-smi failed {self.gpu_failures} times ({exc}) -- "
                    "giving up on GPU metrics, publishing null from here on"
                )
            return None, None

        # The first line only: a multi-GPU machine is not this rover, and
        # averaging two cards would report a number neither of them is doing.
        first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        parts = [part.strip() for part in first.split(",")]
        if len(parts) < 2:
            return None, None

        try:
            usage, temp = float(parts[0]), float(parts[1])
        except ValueError:
            # "[N/A]" is what nvidia-smi prints for a metric the card does not
            # expose -- a laptop GPU in a power-saved state, most often.
            return None, None

        self.gpu_failures = 0
        return usage, temp

    def storage_used_tb(self) -> float | None:
        try:
            return psutil.disk_usage(self.storage_path).used / BYTES_PER_TB
        except OSError as exc:
            self.get_logger().warn(
                f"cannot stat {self.storage_path}: {exc}",
                throttle_duration_sec=30.0,
            )
            return None

    # --- publish -----------------------------------------------------------

    def publish_status(self) -> None:
        gpu_usage, gpu_temp = self.gpu_stats()
        memory = psutil.virtual_memory()

        payload = {
            "cpuUsage": round(psutil.cpu_percent(interval=None), 1),
            "cpuTemp": self._round(self.cpu_temperature()),
            "gpuUsage": self._round(gpu_usage),
            "gpuTemp": self._round(gpu_temp),
            "ramUsage": round(memory.used / BYTES_PER_GB, 2),
            "storageUsage": self._round(self.storage_used_tb(), 3),
        }

        self.publisher.publish(String(data=json.dumps(payload)))

    @staticmethod
    def _round(value: float | None, digits: int = 1) -> float | None:
        return None if value is None else round(value, digits)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ComputePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
