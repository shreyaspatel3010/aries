"""Small helpers shared by the maintenance-panel ROS node and its tests."""

import math
import pathlib
import threading

import yaml
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def load_enabled_controls(config_file, node_name, control_order):
    """Read and validate the enabled-control snapshot directly from YAML.

    ROS parameter files are normally consumed only while a node starts. The
    panel trigger intentionally calls this helper again so operators can edit
    the sequence without restarting the perception and motion node.
    """
    path = pathlib.Path(config_file).expanduser().resolve(strict=True)
    document = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(document, dict):
        raise ValueError("YAML root must be a mapping")
    node_config = document.get(node_name)
    if node_config is None:
        node_config = document.get(f"/{node_name}")
    if not isinstance(node_config, dict):
        raise ValueError(f"YAML has no {node_name!r} node mapping")
    parameters = node_config.get("ros__parameters")
    if not isinstance(parameters, dict):
        raise ValueError("YAML node has no ros__parameters mapping")
    controls = parameters.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("YAML ros__parameters has no controls mapping")

    expected = list(control_order)
    unknown = sorted(set(controls) - set(expected))
    missing = sorted(set(expected) - set(controls))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        if missing:
            details.append("missing: " + ", ".join(missing))
        raise ValueError("control names do not match task table (" +
                         "; ".join(details) + ")")
    non_boolean = sorted(name for name, enabled in controls.items()
                         if type(enabled) is not bool)
    if non_boolean:
        raise ValueError("control values must be true/false: " +
                         ", ".join(non_boolean))
    enabled = [name for name in expected if controls[name]]
    return enabled, path, path.stat().st_mtime


def make_joint_trajectory(joint_name, position, duration_sec):
    """Build the command accepted by a JointTrajectoryController topic."""
    trajectory = JointTrajectory()
    trajectory.joint_names = [str(joint_name)]
    point = JointTrajectoryPoint()
    point.positions = [float(position)]
    duration = max(0.05, float(duration_sec))
    point.time_from_start.sec = int(math.floor(duration))
    point.time_from_start.nanosec = int(round(
        (duration - point.time_from_start.sec) * 1e9))
    if point.time_from_start.nanosec >= 1_000_000_000:
        point.time_from_start.sec += 1
        point.time_from_start.nanosec -= 1_000_000_000
    trajectory.points.append(point)
    return trajectory


def wait_future(future, timeout_sec):
    """Wait while another MultiThreadedExecutor worker services callbacks."""
    completed = threading.Event()
    future.add_done_callback(lambda _: completed.set())
    if not completed.wait(timeout=float(timeout_sec)):
        return None, "timed out"
    if future.exception() is not None:
        return None, str(future.exception())
    return future.result(), ""


def run_action(client, goal, label, result_timeout_sec):
    """Return an action GetResult response after validating its goal handle."""
    goal_handle, why = wait_future(client.send_goal_async(goal), 10.0)
    if goal_handle is None:
        return None, f"{label} goal response {why}"
    if not goal_handle.accepted:
        return None, f"{label} goal rejected"
    result, why = wait_future(
        goal_handle.get_result_async(), result_timeout_sec)
    if result is None:
        # A timed-out client must not abandon a live controller/MoveGroup goal:
        # MoveIt's trajectory manager otherwise remains busy for every later
        # request. Cancellation is best effort because a disconnected action
        # server may make the cancel future fail too.
        try:
            wait_future(goal_handle.cancel_goal_async(), 5.0)
        except Exception:
            pass
        return None, f"{label} result {why}"
    return result, ""


def log_operation_result(logger, name, ok, detail):
    """Log success/error from different call sites, as rclpy requires.

    Jazzy binds a logger call site to its first severity. Selecting ``info`` or
    ``error`` dynamically and invoking it from one source line therefore raises
    ``ValueError`` when a later command has a different outcome.
    """
    message = f"[{name}] {detail}"
    if ok:
        logger.info(message)
    else:
        logger.error(message)
