#!/usr/bin/env python3
"""
TEB & Pure Pursuit Smooth Global Trajectory Planner & Obstacle-Aware Local Follower
===================================================================================

Runs on the physical Aries rover by default; pass --sim for the Gazebo stack.

HARDWARE INTERFACE (default profile)
------------------------------------
  /odometry/filtered   (nav_msgs/Odometry)  robot_localization EKF, odom -> base_footprint
  /cmd_vel             (geometry_msgs/Twist) consumed by cmd_vel_odrive_bridge
  /aries_drive/enabled (std_msgs/Bool, latched)  true only when all six axes are CLOSED_LOOP
  /aries_drive/enable  (std_srvs/SetBool)    arms/disarms the six ODrive axes
  /cmd_vel/teleop      (geometry_msgs/Twist) LB-gated joystick, used as manual override
  /planner/start,/planner/stop (std_srvs/Trigger)  run control

Because cmd_vel_teleop_relay yields /cmd_vel to any other publisher, this node
becomes the sole motor-command owner the moment it advertises /cmd_vel. It
therefore forwards the LB-gated joystick itself (manual override always wins),
and --dry-run never advertises the topic at all so the relay keeps ownership.

Hardware safety layer:
  - motion is blocked until the drive reports armed and the run is started
  - stale/absent odometry (> odom_timeout_s) forces a hard stop
  - linear/angular acceleration limiting and stall-deadband compensation
  - optional geofence radius around the start pose (the Gazebo arena box is
    --sim only)

1. Global Path Planning:
   - Pre-generates a smooth, continuous-curvature (spline / TEB-smoothed) global
     trajectory through waypoints before starting navigation.
   - Calculates continuous curvature κ(s) and dynamic curvature-dependent velocity
     profiles to prevent aggressive turns, overshoot, and wheel-slip drift.
   - Supports open paths as well as closed circuits (--closed-loop).

2. Local Planner & Follower:
   - Tracks the smooth trajectory using curvature-adaptive Pure Pursuit with
     lookahead scaling and heading alignment.
   - Halts at every reached waypoint for a configurable duration (halt_time)
     before resuming toward the next target.
   - Accuracy Pointer System: 3D downward arrows, bullseye concentric tolerance rings,
     and arrival error metrics for every waypoint to check landing accuracy in RViz.
   - Lookahead & Tracking Visualizer: Publishes Lookahead Target Sphere, Lookahead
     Steering Ray, Cross-Track Error Vector, and real-time HUD telemetry.
   - Listens to obstacle notification topic (/obstacle_detected) and obstacle
     positions (/obstacles/markers or /obstacles).
   - Generates smooth local turnaround/detour splines maintaining at least
     a parameterized minimum clearance distance around obstacles.
   - Once cleared, smoothly rejoins and re-engages the global path.
"""

import math
import time
import signal
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.interpolate as interpolate

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, TwistStamped, PointStamped, Point, PoseStamped
from std_msgs.msg import Bool, Header
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401


# ═══════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

DEFAULT_BASE_VELOCITY      = 0.45   # Normal cruising speed (m/s) — Gazebo profile
DEFAULT_MIN_VELOCITY       = 0.15   # Minimum speed on sharp curves / approach (m/s)
DEFAULT_MAX_VELOCITY       = 0.60   # Maximum speed on straightaways (m/s)
DEFAULT_MAX_ANGULAR_Z      = 1.0    # Maximum yaw rate (rad/s)
DEFAULT_MAX_CURVATURE      = 2.0    # 1/m maximum allowable path curvature
DEFAULT_MAX_LAT_ACCEL      = 0.45   # m/s^2 max lateral acceleration on curves (v^2 * k <= a_lat)

DEFAULT_LOOKAHEAD_MIN      = 0.45   # Minimum lookahead distance (m)
DEFAULT_LOOKAHEAD_GAIN     = 0.60   # Lookahead scaling factor: Ld = max(Lmin, gain * v)
ALIGN_ONLY_ANGLE           = 0.85   # If heading error exceeds this (rad), align in place before cruising
ALIGN_EXIT_ANGLE           = 0.30   # Hysteresis: stay in align-in-place until error drops below this (rad)

# ═══════════════════════════════════════════════════════════════
# HARDWARE PROFILE (physical Aries rover)
# ═══════════════════════════════════════════════════════════════

# cmd_vel_odrive_bridge consumes an unstamped Twist here and fans it out to the
# six ODrive axes. It ignores commands older than command_timeout_s (0.25 s by
# default), so the control loop must keep publishing while it wants motion.
HW_CMD_VEL_TOPIC           = '/cmd_vel'
HW_CMD_VEL_STAMPED         = False
HW_ODOM_TOPIC              = '/odometry/filtered'
HW_DRIVE_ENABLED_TOPIC     = '/aries_drive/enabled'   # latched (TRANSIENT_LOCAL) Bool
HW_DRIVE_ENABLE_SERVICE    = '/aries_drive/enable'    # std_srvs/SetBool
HW_TELEOP_TOPIC            = '/cmd_vel/teleop'

# Gazebo profile (--sim): ros2_control rover_controller wants a TwistStamped.
SIM_CMD_VEL_TOPIC          = '/rover_controller/cmd_vel'
SIM_CMD_VEL_STAMPED        = True
SIM_ODOM_TOPIC             = '/odometry/filtered'

# Physical caps. The bridge clamps to max_linear_mps=0.45 / max_angular_rps=2.10
# and ramps the wheels at 3.0 rev/s^2 regardless; these keep the rover well
# inside that envelope so the commanded and executed twist stay the same thing.
HW_MAX_LINEAR_CAP          = 0.45   # cmd_vel_odrive_bridge max_linear_mps
HW_MAX_ANGULAR_CAP         = 2.10   # cmd_vel_odrive_bridge max_angular_rps
HW_BASE_VELOCITY           = 0.30   # m/s — conservative default for field runs
HW_MAX_ANGULAR_Z           = 1.20   # rad/s

# Rover-side ramping. Skid-steer scrub means a velocity step is a traction loss
# and an odometry error, so the command itself is rate limited.
DEFAULT_LINEAR_ACCEL       = 0.40   # m/s^2 while speeding up
DEFAULT_LINEAR_DECEL       = 0.90   # m/s^2 while slowing down / reversing sign
DEFAULT_ANGULAR_ACCEL      = 1.60   # rad/s^2

# Stall deadband: below these the six wheels scrub without moving the chassis.
DEFAULT_MIN_MOVE_LINEAR    = 0.10   # m/s applied to any non-zero drive command
DEFAULT_MIN_MOVE_ANGULAR   = 0.30   # rad/s applied only to turn-in-place commands

# The EKF is dead-reckoning (wheel odometry + MicroStrain yaw) and measured
# ~7 Hz on the rover, so this is a liveness check, not a rate check.
DEFAULT_ODOM_TIMEOUT_S     = 0.5

DEFAULT_TELEOP_HOLD_S      = 1.0    # Autonomy stays suspended this long after the last manual command
DEFAULT_TELEOP_DEADBAND    = 0.02   # A joystick Twist below this counts as "not driving"
DEFAULT_GEOFENCE_RADIUS    = 0.0    # m from the start pose; 0 disables
DEFAULT_ARM_RETRY_S        = 2.0

# Obstacle avoidance parameters
DEFAULT_MIN_OBS_TRIGGER_DIST = 1.8  # Distance (m) ahead to trigger turnaround detour
DEFAULT_MIN_OBS_CLEARANCE    = 0.80 # Minimum safe radius around obstacle (m)
DEFAULT_DETOUR_REJOIN_DIST   = 2.2  # Distance (m) downstream where detour rejoins global path
PATH_CORRIDOR_WIDTH          = 0.45 # Half-width of path corridor monitored for obstacles (m)
AVOID_VELOCITY               = 0.35 # Cruising speed while executing avoidance detour (m/s)

# Trajectory generation & waypoint halting
PATH_RESOLUTION              = 0.05 # Desired distance between interpolated points (m)
WAYPOINT_GOAL_TOLERANCE      = 0.35 # Distance to reach a waypoint and trigger halt (m)
DEFAULT_HALT_TIME            = 1.5  # Halt / pause duration (s) at each reached waypoint
FINE_APPROACH_RADIUS         = 0.40 # Radius to engage proportional fine landing (m)

# Arena bounding box safety limits (matching the Gazebo arena only — there is no
# such box on the physical rover, which uses the geofence radius instead).
ARENA_X_MIN   = -2.0
ARENA_X_MAX   = 12.0
ARENA_Y_MIN   = -2.0
ARENA_Y_MAX   = 15.0
ARENA_MARGIN  = 0.35
ARENA_LOOKAHEAD_T = 0.4


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw: float
    curvature: float
    target_speed: float
    s: float  # Cumulative arc length from start (m)


@dataclass
class ObstacleEntity:
    x: float          # Odom frame X
    y: float          # Odom frame Y
    radius: float     # Estimated radius (m)
    timestamp: float


@dataclass
class RunConfig:
    """Everything the CLI resolves before the node is built."""
    waypoints: Optional[List[Tuple[float, float]]] = None
    closed_loop: bool = False
    relative_coords: bool = False
    base_speed: float = HW_BASE_VELOCITY
    halt_time: float = DEFAULT_HALT_TIME
    min_obs_dist: float = DEFAULT_MIN_OBS_TRIGGER_DIST
    obs_clearance: float = DEFAULT_MIN_OBS_CLEARANCE

    # Hardware interface
    sim: bool = False
    cmd_vel_topic: str = HW_CMD_VEL_TOPIC
    cmd_vel_stamped: bool = HW_CMD_VEL_STAMPED
    odom_topic: str = HW_ODOM_TOPIC
    max_angular: float = HW_MAX_ANGULAR_Z

    # Safety / run control
    dry_run: bool = False
    autostart: bool = False
    arm: bool = False
    require_drive_enabled: bool = True
    teleop_override: bool = True
    odom_timeout_s: float = DEFAULT_ODOM_TIMEOUT_S
    geofence_radius: float = DEFAULT_GEOFENCE_RADIUS
    use_arena_bounds: bool = False


# ═══════════════════════════════════════════════════════════════
# GEOMETRIC & TRAJECTORY UTILITIES
# ═══════════════════════════════════════════════════════════════

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def global_to_local(point: Tuple[float, float], yaw: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    lx =  dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return lx, ly


def local_to_global(lx: float, ly: float, yaw: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    dx = lx * math.cos(yaw) - ly * math.sin(yaw)
    dy = lx * math.sin(yaw) + ly * math.cos(yaw)
    return origin[0] + dx, origin[1] + dy


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (ROS ENU) straight from the quaternion, with no Euler-order ambiguity."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def ramp(current: float, target: float, max_delta: float) -> float:
    """Move current toward target by at most max_delta."""
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


def apply_stall_deadband(value: float, minimum: float) -> float:
    """Raise a non-zero command to the smallest value that actually moves the rover."""
    if minimum <= 0.0 or value == 0.0:
        return value
    if abs(value) >= minimum:
        return value
    return math.copysign(minimum, value)


# ═══════════════════════════════════════════════════════════════
# GLOBAL TRAJECTORY GENERATOR (SPLINE & CURVATURE SMOOTHING)
# ═══════════════════════════════════════════════════════════════

class GlobalTrajectoryGenerator:
    """
    Constructs a smooth, continuous-curvature trajectory through waypoints.
    Eliminates aggressive cornering and piecewise sharp turns by:
    1. Fitting smooth parametric B-Splines / Cubic splines through waypoints.
    2. Enforcing curvature-bounded velocity profiling (slowing down on sharp turns).
    3. Uniform arc-length reparameterization (fixed spatial delta s).
    """

    @staticmethod
    def generate_trajectory(
        waypoints: List[Tuple[float, float]],
        base_velocity: float = DEFAULT_BASE_VELOCITY,
        max_velocity: float = DEFAULT_MAX_VELOCITY,
        min_velocity: float = DEFAULT_MIN_VELOCITY,
        max_curvature: float = DEFAULT_MAX_CURVATURE,
        max_lat_accel: float = DEFAULT_MAX_LAT_ACCEL,
        resolution: float = PATH_RESOLUTION,
        closed_loop: bool = False
    ) -> List[TrajectoryPoint]:
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required to generate a trajectory.")

        # Filter duplicates or extremely close waypoints
        filtered_wps = [waypoints[0]]
        for wp in waypoints[1:]:
            if euclidean_distance(wp, filtered_wps[-1]) > 0.05:
                filtered_wps.append(wp)

        if closed_loop and euclidean_distance(filtered_wps[0], filtered_wps[-1]) > 0.1:
            filtered_wps.append(filtered_wps[0])

        wps_arr = np.array(filtered_wps)
        num_pts = len(wps_arr)

        if num_pts == 2:
            # Straight line interpolation
            dist = euclidean_distance(tuple(wps_arr[0]), tuple(wps_arr[1]))
            n_samples = max(int(np.ceil(dist / resolution)), 10)
            t = np.linspace(0.0, 1.0, n_samples)
            x_dense = wps_arr[0, 0] + t * (wps_arr[1, 0] - wps_arr[0, 0])
            y_dense = wps_arr[0, 1] + t * (wps_arr[1, 1] - wps_arr[0, 1])
        else:
            # Parametric B-Spline interpolation
            k_spline = min(3, num_pts - 1)
            try:
                # Parametric spline fitting
                tck, u = interpolate.splprep(
                    [wps_arr[:, 0], wps_arr[:, 1]],
                    s=0.0,
                    k=k_spline,
                    per=closed_loop
                )
                # Estimate total path length
                diffs = np.diff(wps_arr, axis=0)
                chord_len = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
                n_samples = max(int(np.ceil(chord_len / resolution)), 20)
                u_dense = np.linspace(0.0, 1.0, n_samples)
                x_dense, y_dense = interpolate.splev(u_dense, tck)
            except Exception:
                # Fallback to linear piecewise interpolation
                diffs = np.diff(wps_arr, axis=0)
                seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
                cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
                total_len = cum_lens[-1]
                s_dense = np.arange(0.0, total_len, resolution)
                x_dense = np.interp(s_dense, cum_lens, wps_arr[:, 0])
                y_dense = np.interp(s_dense, cum_lens, wps_arr[:, 1])

        # Enforce exact endpoint pinning
        if len(x_dense) > 0:
            x_dense[0] = wps_arr[0, 0]
            y_dense[0] = wps_arr[0, 1]
            if not closed_loop:
                x_dense[-1] = wps_arr[-1, 0]
                y_dense[-1] = wps_arr[-1, 1]

        # Compute first and second derivatives along spline
        dx = np.gradient(x_dense)
        dy = np.gradient(y_dense)
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.maximum(ds, 1e-6)

        dx_ds = dx / ds
        dy_ds = dy / ds
        d2x_ds2 = np.gradient(dx_ds) / ds
        d2y_ds2 = np.gradient(dy_ds) / ds

        # Signed Curvature kappa = (dx * d2y - dy * d2x) / ds^3
        curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
        curvature = np.clip(curvature, -max_curvature, max_curvature)

        # Cumulative distance
        cum_dist = np.zeros(len(x_dense))
        cum_dist[1:] = np.cumsum(np.sqrt(np.diff(x_dense)**2 + np.diff(y_dense)**2))

        # Continuous headings
        yaws = np.arctan2(dy_ds, dx_ds)

        # Curvature-adaptive Velocity Profiling
        target_speeds = np.zeros(len(x_dense))
        for i in range(len(x_dense)):
            k_abs = abs(curvature[i])
            if k_abs > 0.05:
                v_limit = math.sqrt(max_lat_accel / (k_abs + 1e-4))
                target_speeds[i] = float(np.clip(v_limit, min_velocity, base_velocity))
            else:
                target_speeds[i] = base_velocity

        # Smooth deceleration / acceleration
        for i in range(1, len(target_speeds)):
            target_speeds[i] = min(target_speeds[i], target_speeds[i - 1] + 0.05)
        for i in range(len(target_speeds) - 2, -1, -1):
            target_speeds[i] = min(target_speeds[i], target_speeds[i + 1] + 0.05)

        # De-ramp speed near goal for open paths
        if not closed_loop and len(target_speeds) > 10:
            deramp_distance = 1.0
            total_len = cum_dist[-1]
            for i in range(len(target_speeds)):
                dist_to_end = total_len - cum_dist[i]
                if dist_to_end < deramp_distance:
                    scale = max(0.0, dist_to_end / deramp_distance)
                    deramp_v = min_velocity + (target_speeds[i] - min_velocity) * scale
                    target_speeds[i] = min(target_speeds[i], deramp_v)

        trajectory: List[TrajectoryPoint] = []
        for i in range(len(x_dense)):
            trajectory.append(TrajectoryPoint(
                x=float(x_dense[i]),
                y=float(y_dense[i]),
                yaw=float(yaws[i]),
                curvature=float(curvature[i]),
                target_speed=float(target_speeds[i]),
                s=float(cum_dist[i])
            ))

        return trajectory


# ═══════════════════════════════════════════════════════════════
# LOCAL DETOUR & TURNAROUND GENERATOR
# ═══════════════════════════════════════════════════════════════

class LocalDetourPlanner:
    """
    Generates a smooth avoidance detour spline around an obstacle, maintaining
    at least a specified minimum clearance distance, and rejoins the global path.
    """

    @staticmethod
    def generate_detour(
        start_pose: Tuple[float, float, float],
        obstacle: ObstacleEntity,
        global_traj: List[TrajectoryPoint],
        closest_idx: int,
        min_clearance: float = DEFAULT_MIN_OBS_CLEARANCE,
        rejoin_distance: float = DEFAULT_DETOUR_REJOIN_DIST,
        avoid_velocity: float = AVOID_VELOCITY,
        resolution: float = PATH_RESOLUTION
    ) -> Optional[List[TrajectoryPoint]]:
        x_r, y_r, yaw_r = start_pose
        obs_x, obs_y = obstacle.x, obstacle.y
        obs_radius = max(obstacle.radius, 0.2)
        required_clearance = obs_radius + min_clearance

        # Find where obstacle projects onto global path
        dists_to_obs = [euclidean_distance((tp.x, tp.y), (obs_x, obs_y)) for tp in global_traj]
        obs_nearest_idx = int(np.argmin(dists_to_obs))
        obs_path_point = global_traj[obs_nearest_idx]

        # Determine avoidance side (turn right or turn left depending on relative angle)
        lx, ly = global_to_local((obs_x, obs_y), yaw_r, (x_r, y_r))
        side = 1.0 if ly <= 0.0 else -1.0

        # Normal vector to path at obstacle
        path_yaw = obs_path_point.yaw
        normal_x = -math.sin(path_yaw) * side
        normal_y =  math.cos(path_yaw) * side

        envelope_clearance = required_clearance * 1.25
        tangent_x = math.cos(path_yaw)
        tangent_y = math.sin(path_yaw)

        entry_x = obs_x - 0.6 * required_clearance * tangent_x + normal_x * (required_clearance * 0.95)
        entry_y = obs_y - 0.6 * required_clearance * tangent_y + normal_y * (required_clearance * 0.95)

        apex_x = obs_x + normal_x * envelope_clearance
        apex_y = obs_y + normal_y * envelope_clearance

        exit_x = obs_x + 0.6 * required_clearance * tangent_x + normal_x * (required_clearance * 0.95)
        exit_y = obs_y + 0.6 * required_clearance * tangent_y + normal_y * (required_clearance * 0.95)

        rejoin_s = obs_path_point.s + rejoin_distance
        rejoin_idx = obs_nearest_idx
        for i in range(obs_nearest_idx, len(global_traj)):
            if global_traj[i].s >= rejoin_s:
                rejoin_idx = i
                break
        else:
            rejoin_idx = len(global_traj) - 1

        rejoin_point = global_traj[rejoin_idx]

        control_points = [
            (x_r, y_r),
            (entry_x, entry_y),
            (apex_x, apex_y),
            (exit_x, exit_y),
            (rejoin_point.x, rejoin_point.y)
        ]

        try:
            detour_traj = GlobalTrajectoryGenerator.generate_trajectory(
                control_points,
                base_velocity=avoid_velocity,
                max_velocity=avoid_velocity,
                min_velocity=DEFAULT_MIN_VELOCITY,
                resolution=resolution,
                closed_loop=False
            )
            return detour_traj
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
# CLI PARSER
# ═══════════════════════════════════════════════════════════════

def parse_cli_args(argv) -> RunConfig:
    parser = argparse.ArgumentParser(
        description='TEB Pure Pursuit Global & Local Planner (physical rover by default)'
    )
    parser.add_argument(
        '--waypoints', nargs='+', metavar='X,Y', default=None,
        help='List of X,Y waypoints, e.g. --waypoints 3.0,2.0 5.0,7.0 8.0,2.0 0.0,0.0'
    )
    parser.add_argument(
        '--closed-loop', action='store_true', default=False,
        help='If set, generates a closed circuit trajectory looping back to start'
    )
    parser.add_argument(
        '--relative-coords', action='store_true', default=False,
        help='Treat waypoints as offsets from the pose the rover is in when the run starts. '
             'Recommended on hardware: the odom frame origin is wherever the EKF came up.'
    )
    parser.add_argument(
        '--speed', type=float, default=None,
        help=f'Base cruising speed in m/s (default: {HW_BASE_VELOCITY} on hardware, '
             f'{DEFAULT_BASE_VELOCITY} with --sim)'
    )
    parser.add_argument(
        '--max-angular', type=float, default=None,
        help=f'Yaw rate cap in rad/s (default: {HW_MAX_ANGULAR_Z} on hardware, {DEFAULT_MAX_ANGULAR_Z} with --sim)'
    )
    parser.add_argument(
        '--halt-time', type=float, default=DEFAULT_HALT_TIME,
        help=f'Halt duration in seconds at every reached waypoint (default: {DEFAULT_HALT_TIME}s)'
    )
    parser.add_argument(
        '--min-obs-dist', type=float, default=DEFAULT_MIN_OBS_TRIGGER_DIST,
        help=f'Minimum distance to obstacle before triggering detour (default: {DEFAULT_MIN_OBS_TRIGGER_DIST} m)'
    )
    parser.add_argument(
        '--obs-clearance', type=float, default=DEFAULT_MIN_OBS_CLEARANCE,
        help=f'Minimum safe clearance around obstacle during detour (default: {DEFAULT_MIN_OBS_CLEARANCE} m)'
    )

    hw = parser.add_argument_group('hardware interface & safety')
    hw.add_argument(
        '--sim', action='store_true', default=False,
        help=f'Gazebo profile: TwistStamped on {SIM_CMD_VEL_TOPIC}, arena box on, '
             'no arm check, starts immediately'
    )
    hw.add_argument('--cmd-vel-topic', default=None, help='Override the command topic')
    hw.add_argument('--odom-topic', default=None, help='Override the odometry topic')
    hw.add_argument(
        '--dry-run', action='store_true', default=False,
        help='Plan, track and visualize but never advertise cmd_vel — the joystick keeps the rover'
    )
    hw.add_argument(
        '--autostart', action='store_true', default=False,
        help='Start driving as soon as the rover is armed and localized, without waiting for /planner/start'
    )
    hw.add_argument(
        '--arm', action='store_true', default=False,
        help='Request /aries_drive/enable at startup instead of expecting the operator to arm the rover'
    )
    hw.add_argument(
        '--ignore-drive-state', action='store_true', default=False,
        help='Do not require /aries_drive/enabled before commanding motion'
    )
    hw.add_argument(
        '--no-teleop-override', action='store_true', default=False,
        help='Do not forward LB-gated /cmd_vel/teleop (manual override off)'
    )
    hw.add_argument(
        '--odom-timeout', type=float, default=DEFAULT_ODOM_TIMEOUT_S,
        help=f'Hard stop when odometry is older than this (default: {DEFAULT_ODOM_TIMEOUT_S}s)'
    )
    hw.add_argument(
        '--geofence', type=float, default=DEFAULT_GEOFENCE_RADIUS,
        help='Latch a stop if the rover gets further than this many metres from its start pose (0 = off)'
    )
    parsed, _ = parser.parse_known_args(argv)

    waypoints = None
    if parsed.waypoints:
        waypoints = []
        for wp_str in parsed.waypoints:
            try:
                x_str, y_str = wp_str.split(',')
                waypoints.append((float(x_str), float(y_str)))
            except ValueError:
                raise ValueError(f"Invalid waypoint format '{wp_str}'. Expected 'X,Y' (e.g. 3.0,2.0)")

    sim = parsed.sim
    speed = parsed.speed if parsed.speed is not None else (DEFAULT_BASE_VELOCITY if sim else HW_BASE_VELOCITY)
    max_ang = parsed.max_angular if parsed.max_angular is not None else (
        DEFAULT_MAX_ANGULAR_Z if sim else HW_MAX_ANGULAR_Z
    )
    if not sim:
        # Never ask the bridge for more than it will pass through, so the twist
        # the controller reasons about is the twist the wheels receive.
        speed = min(speed, HW_MAX_LINEAR_CAP)
        max_ang = min(max_ang, HW_MAX_ANGULAR_CAP)

    return RunConfig(
        waypoints=waypoints,
        closed_loop=parsed.closed_loop,
        relative_coords=parsed.relative_coords,
        base_speed=speed,
        halt_time=parsed.halt_time,
        min_obs_dist=parsed.min_obs_dist,
        obs_clearance=parsed.obs_clearance,
        sim=sim,
        cmd_vel_topic=parsed.cmd_vel_topic or (SIM_CMD_VEL_TOPIC if sim else HW_CMD_VEL_TOPIC),
        cmd_vel_stamped=SIM_CMD_VEL_STAMPED if sim else HW_CMD_VEL_STAMPED,
        odom_topic=parsed.odom_topic or (SIM_ODOM_TOPIC if sim else HW_ODOM_TOPIC),
        max_angular=max_ang,
        dry_run=parsed.dry_run,
        autostart=parsed.autostart or sim,
        arm=parsed.arm and not sim,
        require_drive_enabled=(not parsed.ignore_drive_state) and not sim,
        teleop_override=(not parsed.no_teleop_override) and not sim,
        odom_timeout_s=parsed.odom_timeout,
        geofence_radius=max(0.0, parsed.geofence),
        use_arena_bounds=sim,
    )


# ═══════════════════════════════════════════════════════════════
# MAIN ROS 2 NODE: TEB_Pure_Pursuit_Planner
# ═══════════════════════════════════════════════════════════════

class TEBPurePursuitPlannerNode(Node):
    """
    ROS 2 Node executing smooth global trajectory planning and obstacle-aware
    pure pursuit local following with safe detour turnaround, accuracy pointers,
    and waypoint halting.
    """

    STATE_INIT               = "INITIALIZING"
    STATE_FOLLOW_GLOBAL      = "FOLLOWING_GLOBAL_PATH"
    STATE_AVOID_OBSTACLE     = "AVOIDING_OBSTACLE"
    STATE_REENGAGE_GLOBAL    = "REENGAGING_GLOBAL_PATH"
    STATE_PAUSE_WAYPOINT     = "PAUSED_AT_WAYPOINT"
    STATE_GOAL_REACHED       = "GOAL_REACHED"

    def __init__(self, cfg: Optional[RunConfig] = None):
        super().__init__('teb_pure_pursuit_planner')

        self.cfg = cfg or RunConfig()

        # Declare parameters for dynamic reconfig
        self.declare_parameter('base_velocity', self.cfg.base_speed)
        self.declare_parameter('max_angular_z', self.cfg.max_angular)
        self.declare_parameter('max_curvature', DEFAULT_MAX_CURVATURE)
        self.declare_parameter('halt_time', self.cfg.halt_time)
        self.declare_parameter('min_obstacle_distance', self.cfg.min_obs_dist)
        self.declare_parameter('obstacle_clearance', self.cfg.obs_clearance)
        self.declare_parameter('goal_tolerance', WAYPOINT_GOAL_TOLERANCE)
        self.declare_parameter('closed_loop', self.cfg.closed_loop)
        self.declare_parameter('use_relative_coords', self.cfg.relative_coords)
        self.declare_parameter('obstacle_detected_topic', '/obstacle_detected')
        self.declare_parameter('obstacle_markers_topic', '/obstacles/markers')

        # Hardware interface & safety limits
        self.declare_parameter('cmd_vel_topic', self.cfg.cmd_vel_topic)
        self.declare_parameter('cmd_vel_stamped', self.cfg.cmd_vel_stamped)
        self.declare_parameter('odom_topic', self.cfg.odom_topic)
        self.declare_parameter('drive_enabled_topic', HW_DRIVE_ENABLED_TOPIC)
        self.declare_parameter('drive_enable_service', HW_DRIVE_ENABLE_SERVICE)
        self.declare_parameter('teleop_topic', HW_TELEOP_TOPIC)
        self.declare_parameter('require_drive_enabled', self.cfg.require_drive_enabled)
        self.declare_parameter('teleop_override', self.cfg.teleop_override)
        self.declare_parameter('odom_timeout_s', self.cfg.odom_timeout_s)
        self.declare_parameter('geofence_radius', self.cfg.geofence_radius)
        self.declare_parameter('use_arena_bounds', self.cfg.use_arena_bounds)
        self.declare_parameter('linear_accel', DEFAULT_LINEAR_ACCEL)
        self.declare_parameter('linear_decel', DEFAULT_LINEAR_DECEL)
        self.declare_parameter('angular_accel', DEFAULT_ANGULAR_ACCEL)
        self.declare_parameter('min_move_linear', DEFAULT_MIN_MOVE_LINEAR)
        self.declare_parameter('min_move_angular', DEFAULT_MIN_MOVE_ANGULAR)

        # Raw user waypoints
        if self.cfg.waypoints:
            self.raw_waypoints = list(self.cfg.waypoints)
        else:
            self.raw_waypoints = [(3.0, 2.0), (5.0, 7.0), (8.0, 2.0), (0.0, 0.0)]

        self.closed_loop = self.cfg.closed_loop

        # State variables
        self.state = self.STATE_INIT
        self.car_pos: Optional[Tuple[float, float]] = None
        self.car_yaw: Optional[float] = None
        self.start_pos: Optional[Tuple[float, float]] = None

        # Trajectories & Discrete Waypoint Tracking
        self.global_trajectory: List[TrajectoryPoint] = []
        self.local_detour_trajectory: Optional[List[TrajectoryPoint]] = None
        self.traversed_points: List[Tuple[float, float]] = []
        self.abs_waypoints: List[Tuple[float, float]] = []
        self.waypoint_reached: List[bool] = []
        self.waypoint_arrival_errors: Dict[int, float] = {}
        self.current_wp_idx: int = 1

        # Obstacles
        self.obstacle_flag_active = False
        self.detected_obstacles: List[ObstacleEntity] = []
        self.active_avoidance_obstacle: Optional[ObstacleEntity] = None

        # Progress tracking
        self.closest_global_idx = 0
        self.closest_detour_idx = 0
        self.pause_until_time = 0.0
        self.prev_angular_cmd = 0.0
        self.force_reacquire = False
        self.aligning_in_place = False

        # ── Hardware run control & safety state ──────────────────────────────
        self.control_period = 0.05
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.cmd_vel_stamped = bool(self.get_parameter('cmd_vel_stamped').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.require_drive_enabled = bool(self.get_parameter('require_drive_enabled').value)
        self.use_teleop_override = bool(self.get_parameter('teleop_override').value)
        self.odom_timeout_s = float(self.get_parameter('odom_timeout_s').value)
        self.geofence_radius = float(self.get_parameter('geofence_radius').value)
        self.use_arena_bounds = bool(self.get_parameter('use_arena_bounds').value)
        self.linear_accel = float(self.get_parameter('linear_accel').value)
        self.linear_decel = float(self.get_parameter('linear_decel').value)
        self.angular_accel = float(self.get_parameter('angular_accel').value)
        self.min_move_linear = float(self.get_parameter('min_move_linear').value)
        self.min_move_angular = float(self.get_parameter('min_move_angular').value)

        self.run_started = bool(self.cfg.autostart)
        self.estopped = False
        self.drive_enabled = not self.require_drive_enabled
        self.drive_enabled_seen = False
        self.we_armed_the_drive = False
        self.last_odom_time: Optional[float] = None
        self.last_teleop_cmd: Optional[Twist] = None
        self.last_teleop_active_time = 0.0
        self.cur_linear_cmd = 0.0
        self.cur_angular_cmd = 0.0
        self.gate_reason = "waiting for first odometry"

        # TF Buffer for obstacle transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS 2 Subscriptions
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)

        # Obstacle topic subscriptions
        flag_topic = self.get_parameter('obstacle_detected_topic').value
        markers_topic = self.get_parameter('obstacle_markers_topic').value
        self.create_subscription(Bool, flag_topic, self._obstacle_flag_cb, 10)
        self.create_subscription(MarkerArray, markers_topic, self._obstacle_markers_cb, 10)

        # The bridge latches /aries_drive/enabled, so match its durability or the
        # current arm state is missed until the next transition.
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool, str(self.get_parameter('drive_enabled_topic').value),
            self._drive_enabled_cb, latched_qos
        )

        if self.use_teleop_override:
            self.create_subscription(
                Twist, str(self.get_parameter('teleop_topic').value),
                self._teleop_cb, 10
            )

        self.arm_client = self.create_client(
            SetBool, str(self.get_parameter('drive_enable_service').value)
        )

        # ROS 2 Publishers.  In dry-run the cmd_vel publisher is never created:
        # cmd_vel_teleop_relay yields on the presence of another publisher, so
        # merely advertising would kill manual driving.
        if self.cfg.dry_run:
            self.vel_pub = None
        elif self.cmd_vel_stamped:
            self.vel_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        else:
            self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Run control services
        self.create_service(Trigger, '/planner/start', self._on_start_srv)
        self.create_service(Trigger, '/planner/stop', self._on_stop_srv)

        # Path and Marker Publishers for RViz
        self.global_path_pub = self.create_publisher(Path, '/planner/global_path', 10)
        self.detour_path_pub = self.create_publisher(Path, '/planner/local_detour', 10)
        self.planned_marker_pub = self.create_publisher(Marker, '/pure_pursuit/planned_path', 10)
        self.traversed_marker_pub = self.create_publisher(Marker, '/pure_pursuit/traversed_path', 10)
        self.target_marker_pub = self.create_publisher(MarkerArray, '/planner/lookahead_target', 10)
        self.waypoint_marker_pub = self.create_publisher(MarkerArray, '/planner/waypoint_markers', 10)
        self.obstacle_bubble_pub = self.create_publisher(MarkerArray, '/planner/obstacle_bubbles', 10)

        # Control timer (20 Hz = 50 ms loop; also keeps the bridge's 0.25 s
        # command-freshness window satisfied with a five-fold margin)
        self.control_timer = self.create_timer(self.control_period, self._control_loop)
        self.arm_timer = self.create_timer(DEFAULT_ARM_RETRY_S, self._arm_tick)

        self._log_startup_banner()

    def _log_startup_banner(self):
        profile = "SIMULATION" if self.cfg.sim else "HARDWARE"
        msg_kind = "TwistStamped" if self.cmd_vel_stamped else "Twist"
        sink = "DRY RUN (no cmd_vel published)" if self.cfg.dry_run else \
               f"{self.cmd_vel_topic} ({msg_kind})"
        self.get_logger().info(
            f"TEB Pure Pursuit Planner [{profile}] — {len(self.raw_waypoints)} waypoints, "
            f"closed_loop={self.closed_loop}, base={self.cfg.base_speed:.2f} m/s, "
            f"max_yaw={self.cfg.max_angular:.2f} rad/s, halt={self.cfg.halt_time:.1f}s"
        )
        self.get_logger().info(
            f"  odometry: {self.odom_topic} | commands: {sink} | "
            f"waypoints are {'RELATIVE to start pose' if self.cfg.relative_coords else 'ABSOLUTE in the odom frame'}"
        )
        self.get_logger().info(
            f"  safety: require_armed={self.require_drive_enabled} "
            f"odom_timeout={self.odom_timeout_s:.2f}s "
            f"geofence={'off' if self.geofence_radius <= 0 else f'{self.geofence_radius:.1f} m'} "
            f"arena_box={self.use_arena_bounds} teleop_override={self.use_teleop_override}"
        )
        if self.run_started:
            self.get_logger().warn("  autostart enabled — the rover will drive as soon as it is armed and localized")
        else:
            self.get_logger().info("  run is HELD. Start it with:  ros2 service call /planner/start std_srvs/srv/Trigger")

        # Obstacle avoidance has no data source on the rover since the LiDAR was
        # removed; say so once rather than silently never detouring.
        flag_topic = str(self.get_parameter('obstacle_detected_topic').value)
        markers_topic = str(self.get_parameter('obstacle_markers_topic').value)
        publishers = (self.count_publishers(flag_topic) + self.count_publishers(markers_topic))
        if publishers == 0:
            self.get_logger().warn(
                f"  no publisher on {flag_topic} or {markers_topic} — obstacle detour is INACTIVE; "
                "the rover will drive the planned path regardless of what is in front of it"
            )

    # ── Odometry & State Callbacks ────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.car_pos = (x, y)

        q = msg.pose.pose.orientation
        self.car_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.last_odom_time = self._now()

        # Log traversed trajectory
        if not self.traversed_points:
            self.traversed_points.append((x, y))
        else:
            if euclidean_distance(self.traversed_points[-1], (x, y)) >= 0.05:
                self.traversed_points.append((x, y))

    def _drive_enabled_cb(self, msg: Bool):
        enabled = bool(msg.data)
        if self.drive_enabled_seen and enabled != self.drive_enabled:
            if enabled:
                self.get_logger().info("🔌 Drive armed (all six ODrive axes in CLOSED_LOOP_CONTROL)")
            else:
                self.get_logger().warn("🔌 Drive DISARMED — autonomous motion blocked")
        self.drive_enabled = enabled
        self.drive_enabled_seen = True

    def _teleop_cb(self, msg: Twist):
        """LB-gated joystick. Any real stick input suspends autonomy and passes through."""
        self.last_teleop_cmd = msg
        if (abs(msg.linear.x) > DEFAULT_TELEOP_DEADBAND
                or abs(msg.angular.z) > DEFAULT_TELEOP_DEADBAND):
            if not self._teleop_override_active():
                self.get_logger().warn("🎮 Manual joystick input — autonomy suspended, forwarding teleop")
            self.last_teleop_active_time = self._now()

    def _teleop_override_active(self) -> bool:
        return (self.use_teleop_override
                and (self._now() - self.last_teleop_active_time) < DEFAULT_TELEOP_HOLD_S)

    # ── Run control services ──────────────────────────────────────────────────

    def _on_start_srv(self, request, response):
        self.estopped = False
        if not self.run_started:
            self.run_started = True
            # Re-plan from wherever the rover actually is when the operator says go.
            self.start_pos = None
            self.global_trajectory = []
            response.message = "run started"
            self.get_logger().info("▶️  Run STARTED — planning from the current pose")
        else:
            response.message = "run already active (e-stop cleared)"
        response.success = True
        return response

    def _on_stop_srv(self, request, response):
        self.estopped = True
        self.run_started = False
        self._hard_stop("stop service called")
        self.get_logger().warn("⏹️  Run STOPPED by service call — zero twist latched")
        response.success = True
        response.message = "stopped"
        return response

    def _now(self) -> float:
        """Node clock in seconds (honours use_sim_time)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _obstacle_flag_cb(self, msg: Bool):
        self.obstacle_flag_active = msg.data

    def _obstacle_markers_cb(self, msg: MarkerArray):
        """Extracts obstacle positions and extents in odom frame."""
        now = time.time()
        obstacles: List[ObstacleEntity] = []

        for marker in msg.markers:
            if marker.action == Marker.DELETE or not marker.points:
                continue

            pts = np.array([[p.x, p.y, p.z] for p in marker.points])
            cx, cy, cz = pts.mean(axis=0)

            rad = float(np.max(np.linalg.norm(pts[:, :2] - np.array([cx, cy]), axis=1)))
            rad = max(rad, 0.25)

            pt = PointStamped()
            pt.header = marker.header
            pt.point.x, pt.point.y, pt.point.z = float(cx), float(cy), float(cz)

            try:
                pt_odom = self.tf_buffer.transform(pt, 'odom', timeout=rclpy.duration.Duration(seconds=0.05))
                obstacles.append(ObstacleEntity(x=pt_odom.point.x, y=pt_odom.point.y, radius=rad, timestamp=now))
            except Exception:
                obstacles.append(ObstacleEntity(x=float(cx), y=float(cy), radius=rad, timestamp=now))

        self.detected_obstacles = obstacles

    # ── Global Trajectory Initialization ──────────────────────────────────────

    def _initialize_global_trajectory(self):
        """Builds smooth global trajectory starting from rover start position."""
        if self.car_pos is None:
            return
        self.start_pos = self.car_pos

        use_relative = self.get_parameter('use_relative_coords').value

        self.abs_waypoints = [self.start_pos]
        if use_relative:
            # Relative mode: offset from rover start position
            for wp in self.raw_waypoints:
                self.abs_waypoints.append((self.start_pos[0] + wp[0], self.start_pos[1] + wp[1]))
        else:
            # Absolute mode: waypoints are exact world/odom coordinates
            for wp in self.raw_waypoints:
                self.abs_waypoints.append(wp)

        self.waypoint_reached = [True] + [False] * (len(self.abs_waypoints) - 1)
        self.waypoint_arrival_errors = {0: 0.0}
        self.current_wp_idx = 1
        self.closest_global_idx = 0
        self.closest_detour_idx = 0
        self.prev_angular_cmd = 0.0

        base_speed = self.get_parameter('base_velocity').value
        max_curv = self.get_parameter('max_curvature').value

        self.global_trajectory = GlobalTrajectoryGenerator.generate_trajectory(
            self.abs_waypoints,
            base_velocity=base_speed,
            max_velocity=DEFAULT_MAX_VELOCITY,
            min_velocity=DEFAULT_MIN_VELOCITY,
            max_curvature=max_curv,
            max_lat_accel=DEFAULT_MAX_LAT_ACCEL,
            resolution=PATH_RESOLUTION,
            closed_loop=self.closed_loop
        )

        self.state = self.STATE_FOLLOW_GLOBAL
        self.traversed_points = [self.start_pos]
        self.force_reacquire = True
        self.get_logger().info(
            f"✅ Smooth global trajectory generated: {len(self.global_trajectory)} points, "
            f"total length {self.global_trajectory[-1].s:.2f} m, {len(self.abs_waypoints)} reach points."
        )
        self.get_logger().info(
            "   start (%.2f, %.2f) -> " % self.start_pos
            + " -> ".join(f"({wp[0]:.2f}, {wp[1]:.2f})" for wp in self.abs_waypoints[1:])
        )
        self._publish_global_path_visualization()
        self._publish_waypoint_markers()

    # ── Hardware gating, arming and command shaping ───────────────────────────

    def _arm_tick(self):
        """Ask the ODrive bridge to arm, retrying until it reports enabled."""
        if not self.cfg.arm or self.drive_enabled or self.estopped:
            return
        if not self.arm_client.service_is_ready():
            self.get_logger().warn(
                "waiting for the drive enable service — is the rover drive stack up?",
                throttle_duration_sec=5.0
            )
            return
        self.get_logger().info("requesting ODrive arm (/aries_drive/enable true)")
        self.arm_client.call_async(SetBool.Request(data=True))
        self.we_armed_the_drive = True

    def _safety_gate(self, now: float) -> Tuple[bool, str]:
        """Decide whether autonomous motion is permitted this tick."""
        if self.estopped:
            return False, "e-stopped"
        if not self.run_started:
            return False, "run not started (call /planner/start)"
        if self.car_pos is None or self.car_yaw is None or self.last_odom_time is None:
            return False, f"no odometry on {self.odom_topic}"
        odom_age = now - self.last_odom_time
        if odom_age > self.odom_timeout_s:
            return False, f"odometry stale by {odom_age:.2f}s"
        if self.require_drive_enabled and not self.drive_enabled:
            reason = "drive not armed" if self.drive_enabled_seen else "drive arm state unknown"
            return False, reason
        if self.geofence_radius > 0.0 and self.start_pos is not None:
            drift = euclidean_distance(self.car_pos, self.start_pos)
            if drift > self.geofence_radius:
                if not self.estopped:
                    self.estopped = True
                    self.get_logger().error(
                        f"🚧 GEOFENCE BREACH: {drift:.2f} m from start exceeds "
                        f"{self.geofence_radius:.2f} m — latching stop"
                    )
                return False, "geofence breach"
        return True, ""

    def _hard_stop(self, reason: str):
        """Zero the twist immediately, bypassing the ramp, and log why."""
        self.cur_linear_cmd = 0.0
        self.cur_angular_cmd = 0.0
        self.prev_angular_cmd = 0.0
        if reason != self.gate_reason:
            self.gate_reason = reason
            self.get_logger().warn(f"⏸️  Holding: {reason}")
        else:
            self.get_logger().info(f"holding: {reason}", throttle_duration_sec=10.0)
        self._publish_raw(0.0, 0.0)

    # ── Main Control Loop ─────────────────────────────────────────────────────

    def _control_loop(self):
        now = self._now()

        # Manual joystick always wins: forward it and let autonomy re-acquire
        # the path afterwards from wherever the operator left the rover.
        if self._teleop_override_active():
            self.cur_linear_cmd = 0.0
            self.cur_angular_cmd = 0.0
            self.prev_angular_cmd = 0.0
            self.force_reacquire = True
            self.gate_reason = "manual teleop override"
            if self.last_teleop_cmd is not None:
                self._publish_raw(self.last_teleop_cmd.linear.x, self.last_teleop_cmd.angular.z)
            self._publish_traversed_marker()
            self._publish_waypoint_markers()
            return

        allowed, reason = self._safety_gate(now)
        if not allowed:
            self._hard_stop(reason)
            if self.car_pos is not None:
                self._publish_traversed_marker()
                self._publish_waypoint_markers()
            return

        if self.gate_reason:
            self.get_logger().info("▶️  Cleared to drive")
            self.gate_reason = ""

        # First cleared tick: plan from the pose the rover is actually in.
        if not self.global_trajectory:
            self._initialize_global_trajectory()
            if not self.global_trajectory:
                return

        # Parked at the goal: keep a fresh zero twist on the wire so the drive
        # holds a commanded stop instead of timing out into one.
        if self.state == self.STATE_GOAL_REACHED:
            self._publish_raw(0.0, 0.0)
            self._publish_traversed_marker()
            self._publish_waypoint_markers()
            return

        # Handle pause / halt at waypoint
        if self.state == self.STATE_PAUSE_WAYPOINT:
            if now >= self.pause_until_time:
                if self.current_wp_idx >= len(self.abs_waypoints):
                    if self.closed_loop:
                        self.current_wp_idx = 1
                        self.closest_global_idx = 0
                        self.waypoint_reached = [True] + [False] * (len(self.abs_waypoints) - 1)
                        self.waypoint_arrival_errors = {0: 0.0}
                        self.state = self.STATE_FOLLOW_GLOBAL
                        self.get_logger().info("🔄 Starting next closed-loop circuit lap!")
                    else:
                        self.state = self.STATE_GOAL_REACHED
                        self.get_logger().info("🎯 All waypoints completed! Rover is parked at goal.")
                        self._publish_cmd_vel(0.0, 0.0)
                        self._publish_waypoint_markers()
                        return
                else:
                    self.state = self.STATE_FOLLOW_GLOBAL
                    self.get_logger().info(f"🚀 Resuming navigation toward Waypoint {self.current_wp_idx}...")
            else:
                self._publish_cmd_vel(0.0, 0.0)
                self._publish_waypoint_markers()
                return

        # Check for obstacles in corridor
        blocking_obstacle = self._find_blocking_obstacle()

        # State Machine Transitions
        if self.state == self.STATE_FOLLOW_GLOBAL:
            if blocking_obstacle is not None:
                self.get_logger().warn(
                    f"⚠️ Obstacle detected at ({blocking_obstacle.x:.2f}, {blocking_obstacle.y:.2f}) "
                    f"— initiating turnaround detour with safe clearance!"
                )
                self._initiate_detour(blocking_obstacle)

        elif self.state == self.STATE_AVOID_OBSTACLE:
            if self.local_detour_trajectory and self.closest_detour_idx >= len(self.local_detour_trajectory) - 3:
                self.get_logger().info("✅ Detour completed — resuming global trajectory follower.")
                self.local_detour_trajectory = None
                self.active_avoidance_obstacle = None
                self.state = self.STATE_FOLLOW_GLOBAL

        # Execute active controller
        if self.state == self.STATE_AVOID_OBSTACLE and self.local_detour_trajectory:
            self._execute_trajectory_tracking(self.local_detour_trajectory, is_detour=True)
        elif self.state == self.STATE_FOLLOW_GLOBAL:
            self._check_waypoint_reach(now)
            if self.state != self.STATE_PAUSE_WAYPOINT:
                self._execute_trajectory_tracking(self.global_trajectory, is_detour=False)

        # Publish visualizations
        self._publish_traversed_marker()
        self._publish_obstacle_bubbles()
        self._publish_waypoint_markers()

    # ── Waypoint Reach & Halt Detection ───────────────────────────────────────

    def _check_waypoint_reach(self, current_time: float):
        """Checks if the rover has arrived at the current target waypoint to trigger halt."""
        if self.current_wp_idx >= len(self.abs_waypoints):
            return

        target_wp = self.abs_waypoints[self.current_wp_idx]
        dist_to_wp = euclidean_distance(self.car_pos, target_wp)
        goal_tol = self.get_parameter('goal_tolerance').value

        if dist_to_wp <= goal_tol and not self.waypoint_reached[self.current_wp_idx]:
            self.waypoint_reached[self.current_wp_idx] = True
            self.waypoint_arrival_errors[self.current_wp_idx] = dist_to_wp
            halt_duration = self.get_parameter('halt_time').value
            self.pause_until_time = current_time + halt_duration
            self.state = self.STATE_PAUSE_WAYPOINT

            wp_label = f"Waypoint {self.current_wp_idx}"
            if self.current_wp_idx == len(self.abs_waypoints) - 1 and not self.closed_loop:
                wp_label = "Final Goal"

            self.get_logger().info(
                f"🛑 Reached {wp_label} at ({target_wp[0]:.2f}, {target_wp[1]:.2f}) "
                f"with accuracy landing error: {dist_to_wp*100:.1f} cm — Halting for {halt_duration:.1f}s!"
            )
            self._publish_cmd_vel(0.0, 0.0)
            self.current_wp_idx += 1
            self._publish_waypoint_markers()

    # ── Obstacle Detection & Assessment ───────────────────────────────────────

    def _find_blocking_obstacle(self) -> Optional[ObstacleEntity]:
        min_trigger_dist = self.get_parameter('min_obstacle_distance').value
        obs_clearance = self.get_parameter('obstacle_clearance').value

        for obs in self.detected_obstacles:
            lx, ly = global_to_local((obs.x, obs.y), self.car_yaw, self.car_pos)
            if 0.1 < lx <= min_trigger_dist:
                corridor_bound = PATH_CORRIDOR_WIDTH + obs.radius + obs_clearance * 0.5
                if abs(ly) <= corridor_bound:
                    return obs

        if self.obstacle_flag_active:
            target_idx = min(self.closest_global_idx + int(min_trigger_dist / PATH_RESOLUTION), len(self.global_trajectory) - 1)
            target_tp = self.global_trajectory[target_idx]
            virtual_obs = ObstacleEntity(
                x=target_tp.x,
                y=target_tp.y,
                radius=0.35,
                timestamp=time.time()
            )
            return virtual_obs

        return None

    def _initiate_detour(self, obstacle: ObstacleEntity):
        obs_clearance = self.get_parameter('obstacle_clearance').value
        detour = LocalDetourPlanner.generate_detour(
            start_pose=(self.car_pos[0], self.car_pos[1], self.car_yaw),
            obstacle=obstacle,
            global_traj=self.global_trajectory,
            closest_idx=self.closest_global_idx,
            min_clearance=obs_clearance,
            rejoin_distance=DEFAULT_DETOUR_REJOIN_DIST,
            avoid_velocity=AVOID_VELOCITY,
            resolution=PATH_RESOLUTION
        )

        if detour is not None and len(detour) > 2:
            self.local_detour_trajectory = detour
            self.closest_detour_idx = 0
            self.active_avoidance_obstacle = obstacle
            self.state = self.STATE_AVOID_OBSTACLE
            self._publish_detour_path_visualization(detour)
            self.get_logger().info(f"✨ Safe detour curve planned with {len(detour)} points.")
        else:
            self.get_logger().error("Failed to generate feasible detour spline! Pausing.")
            self._publish_cmd_vel(0.0, 0.0)

    # ── Trajectory Tracking & Pure Pursuit Controller ─────────────────────────

    def _execute_trajectory_tracking(self, trajectory: List[TrajectoryPoint], is_detour: bool):
        # Progress-constrained search window to prevent jumping to endpoints on loops or self-crossings.
        # After a manual override or a re-plan the rover may be anywhere relative
        # to the path, so re-acquire once over the whole trajectory.
        last_idx = self.closest_detour_idx if is_detour else self.closest_global_idx
        if self.force_reacquire:
            self.force_reacquire = False
            search_start = 0
            search_end = len(trajectory)
        else:
            search_start = max(0, last_idx - 5)
            search_end = min(len(trajectory), last_idx + 80)

        window_dists = [euclidean_distance((trajectory[i].x, trajectory[i].y), self.car_pos) for i in range(search_start, search_end)]
        best_window_idx = int(np.argmin(window_dists))
        closest_idx = search_start + best_window_idx
        cross_track_err = window_dists[best_window_idx]
        closest_tp = trajectory[closest_idx]

        if is_detour:
            self.closest_detour_idx = closest_idx
        else:
            self.closest_global_idx = closest_idx

        # Check goal arrival
        goal_point = trajectory[-1]
        dist_to_goal = euclidean_distance(self.car_pos, (goal_point.x, goal_point.y))
        goal_tol = self.get_parameter('goal_tolerance').value

        if not self.closed_loop and not is_detour and dist_to_goal < goal_tol and self.current_wp_idx >= len(self.abs_waypoints):
            self.state = self.STATE_GOAL_REACHED
            self.get_logger().info("🎯 Reached final destination! Stopping rover.")
            self._publish_cmd_vel(0.0, 0.0)
            return

        if not is_detour and dist_to_goal < FINE_APPROACH_RADIUS and not self.closed_loop and self.current_wp_idx >= len(self.abs_waypoints):
            linear_v, angular_w = self._fine_proportional_approach((goal_point.x, goal_point.y))
            self._publish_cmd_vel(linear_v, angular_w)
            self._log_telemetry(cross_track_err, 0.0, linear_v, angular_w, "fine approach")
            return

        # Adaptive Lookahead Distance
        ref_speed = trajectory[closest_idx].target_speed
        lookahead_dist = max(DEFAULT_LOOKAHEAD_MIN, DEFAULT_LOOKAHEAD_GAIN * ref_speed)

        # Search forward along trajectory for lookahead point
        target_idx = closest_idx
        target_tp = trajectory[closest_idx]
        current_s = trajectory[closest_idx].s

        for i in range(closest_idx, len(trajectory)):
            if (trajectory[i].s - current_s) >= lookahead_dist:
                target_idx = i
                target_tp = trajectory[i]
                break
        else:
            target_tp = trajectory[-1]

        # Publish lookahead target marker with vector ray and CTE line
        self._publish_target_marker(
            target_pos=(target_tp.x, target_tp.y),
            closest_pos=(closest_tp.x, closest_tp.y),
            lookahead_dist=lookahead_dist,
            cross_track_err=cross_track_err
        )

        # Pure Pursuit Curvature Calculation
        lx, ly = global_to_local((target_tp.x, target_tp.y), self.car_yaw, self.car_pos)
        ld = math.hypot(lx, ly)
        heading_error = math.atan2(ly, lx)

        if ld < 0.01:
            curvature_cmd = 0.0
        else:
            curvature_cmd = float(2.0 * ly / (ld**2))

        max_curv = self.get_parameter('max_curvature').value
        curvature_cmd = float(np.clip(curvature_cmd, -max_curv, max_curv))
        max_ang = self.get_parameter('max_angular_z').value

        # Turn in place when badly misaligned. Hysteresis keeps a skid-steer from
        # chattering between spinning and driving at the threshold.
        align_threshold = ALIGN_EXIT_ANGLE if self.aligning_in_place else ALIGN_ONLY_ANGLE
        if abs(heading_error) > align_threshold:
            self.aligning_in_place = True
            linear_cmd = 0.0
            angular_cmd = float(np.clip(1.1 * heading_error, -max_ang, max_ang))
        else:
            self.aligning_in_place = False
            heading_scale = max(0.25, math.cos(heading_error)**2)
            linear_cmd = ref_speed * heading_scale
            raw_angular = curvature_cmd * linear_cmd

            # Smooth exponential filtering for gradual steering adjustments
            alpha = 0.45
            angular_cmd = alpha * raw_angular + (1.0 - alpha) * self.prev_angular_cmd
            angular_cmd = float(np.clip(angular_cmd, -max_ang, max_ang))

        self.prev_angular_cmd = angular_cmd
        linear_cmd, angular_cmd = self._clamp_to_arena(linear_cmd, angular_cmd)
        self._publish_cmd_vel(linear_cmd, angular_cmd)
        self._log_telemetry(cross_track_err, heading_error, linear_cmd, angular_cmd,
                            "detour" if is_detour else self.state)

    def _log_telemetry(self, cross_track_err: float, heading_error: float,
                       linear_cmd: float, angular_cmd: float, phase: str):
        """One throttled line per second — in the field the terminal is the HUD."""
        if self.current_wp_idx < len(self.abs_waypoints):
            wp = self.abs_waypoints[self.current_wp_idx]
            target = f"wp{self.current_wp_idx}/{len(self.abs_waypoints) - 1} " \
                     f"d={euclidean_distance(self.car_pos, wp):.2f}m"
        else:
            target = "final approach"
        self.get_logger().info(
            f"{phase} | {target} | cmd v={self.cur_linear_cmd:+.2f} w={self.cur_angular_cmd:+.2f} "
            f"(want {linear_cmd:+.2f}/{angular_cmd:+.2f}) | cte={cross_track_err * 100:.0f}cm "
            f"| heading err={math.degrees(heading_error):+.0f}deg"
            + (" | DRY RUN" if self.vel_pub is None else ""),
            throttle_duration_sec=1.0
        )

    def _fine_proportional_approach(self, goal_pos: Tuple[float, float]) -> Tuple[float, float]:
        lx, ly = global_to_local(goal_pos, self.car_yaw, self.car_pos)
        dist = math.hypot(lx, ly)
        heading_error = math.atan2(ly, lx)
        max_ang = self.get_parameter('max_angular_z').value

        if abs(heading_error) > 0.3:
            linear_x = 0.0
        else:
            linear_x = float(np.clip(0.6 * dist, 0.0, 0.15))

        angular_z = float(np.clip(1.8 * heading_error, -max_ang * 0.8, max_ang * 0.8))
        return linear_x, angular_z

    def _clamp_to_arena(self, linear: float, angular: float) -> Tuple[float, float]:
        # Simulation-only: the box matches the Gazebo arena and means nothing on
        # the physical rover, which is fenced by geofence_radius instead.
        if not self.use_arena_bounds:
            return linear, angular
        if self.car_pos is None or self.car_yaw is None:
            return linear, angular

        x, y = self.car_pos
        pred_x = x + linear * math.cos(self.car_yaw) * ARENA_LOOKAHEAD_T
        pred_y = y + linear * math.sin(self.car_yaw) * ARENA_LOOKAHEAD_T

        out_of_bounds = (
            pred_x < ARENA_X_MIN + ARENA_MARGIN or
            pred_x > ARENA_X_MAX - ARENA_MARGIN or
            pred_y < ARENA_Y_MIN + ARENA_MARGIN or
            pred_y > ARENA_Y_MAX - ARENA_MARGIN
        )

        if not out_of_bounds:
            return linear, angular

        center = ((ARENA_X_MIN + ARENA_X_MAX) / 2.0, (ARENA_Y_MIN + ARENA_Y_MAX) / 2.0)
        lx, ly = global_to_local(center, self.car_yaw, self.car_pos)
        ld = math.hypot(lx, ly)
        curv = (2.0 * ly / ld**2) if ld > 0.01 else 0.0
        max_ang = self.get_parameter('max_angular_z').value
        correction_angular = float(np.clip(curv * linear, -max_ang, max_ang))
        safe_linear = min(linear, 0.15)

        self.get_logger().warn(
            "⚠️ Approaching arena boundary — steering back toward arena center",
            throttle_duration_sec=1.0
        )
        return safe_linear, correction_angular

    # ── Command & Visualization Publishers ────────────────────────────────────

    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        """Shape an autonomy command for the drivetrain, then publish it.

        Everything the controller produces goes through here: hardware caps,
        stall-deadband compensation, then acceleration limiting.  The deadband
        is applied to the target before the ramp so a small steady command is
        lifted to something the wheels can actually execute, rather than the
        ramp being stepped up on every tick.
        """
        max_lin = float(self.get_parameter('base_velocity').value)
        max_ang = float(self.get_parameter('max_angular_z').value)

        target_lin = float(np.clip(linear_x, -max_lin, max_lin))
        target_ang = float(np.clip(angular_z, -max_ang, max_ang))

        target_lin = apply_stall_deadband(target_lin, self.min_move_linear)
        # Only turn-in-place needs the yaw deadband; adding it to steering
        # corrections while driving would make the rover snake down the path.
        if abs(target_lin) < 1e-6 and abs(target_ang) > 1e-6:
            target_ang = apply_stall_deadband(target_ang, self.min_move_angular)

        target_lin = float(np.clip(target_lin, -max_lin, max_lin))
        target_ang = float(np.clip(target_ang, -max_ang, max_ang))

        dt = self.control_period
        speeding_up = abs(target_lin) > abs(self.cur_linear_cmd) and target_lin * self.cur_linear_cmd >= 0.0
        lin_limit = self.linear_accel if speeding_up else self.linear_decel
        self.cur_linear_cmd = ramp(self.cur_linear_cmd, target_lin, lin_limit * dt)
        self.cur_angular_cmd = ramp(self.cur_angular_cmd, target_ang, self.angular_accel * dt)

        self._publish_raw(self.cur_linear_cmd, self.cur_angular_cmd)

    def shutdown_drive(self):
        """Disarm on exit, but only if this node is what armed the rover."""
        if not (self.we_armed_the_drive and self.arm_client.service_is_ready()):
            return
        self.get_logger().info("disarming the drive this node armed (/aries_drive/enable false)")
        future = self.arm_client.call_async(SetBool.Request(data=False))
        deadline = time.monotonic() + 1.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _publish_raw(self, linear_x: float, angular_z: float):
        """Put a twist on the wire unchanged (stops, and forwarded teleop)."""
        if self.vel_pub is None:   # --dry-run
            return
        if self.cmd_vel_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_footprint'
            msg.twist.linear.x = float(linear_x)
            msg.twist.angular.z = float(angular_z)
        else:
            msg = Twist()
            msg.linear.x = float(linear_x)
            msg.angular.z = float(angular_z)
        self.vel_pub.publish(msg)

    def _publish_global_path_visualization(self):
        if not self.global_trajectory:
            return

        now = self.get_clock().now().to_msg()

        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = now
        for tp in self.global_trajectory:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = now
            pose.pose.position.x = float(tp.x)
            pose.pose.position.y = float(tp.y)
            pose.pose.position.z = 0.05
            path_msg.poses.append(pose)
        self.global_path_pub.publish(path_msg)

        planned_marker = Marker()
        planned_marker.header.frame_id = 'odom'
        planned_marker.header.stamp = now
        planned_marker.ns = 'pure_pursuit'
        planned_marker.id = 1
        planned_marker.type = Marker.LINE_STRIP
        planned_marker.action = Marker.ADD
        planned_marker.scale.x = 0.06
        planned_marker.color.r = 0.1
        planned_marker.color.g = 0.75
        planned_marker.color.b = 1.0
        planned_marker.color.a = 1.0
        planned_marker.points = [Point(x=float(tp.x), y=float(tp.y), z=0.05) for tp in self.global_trajectory]
        self.planned_marker_pub.publish(planned_marker)

    def _publish_detour_path_visualization(self, detour: List[TrajectoryPoint]):
        now = self.get_clock().now().to_msg()
        detour_path = Path()
        detour_path.header.frame_id = 'odom'
        detour_path.header.stamp = now
        for tp in detour:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = now
            pose.pose.position.x = float(tp.x)
            pose.pose.position.y = float(tp.y)
            pose.pose.position.z = 0.08
            detour_path.poses.append(pose)
        self.detour_path_pub.publish(detour_path)

    def _publish_traversed_marker(self):
        if not self.traversed_points:
            return
        now = self.get_clock().now().to_msg()
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = now
        m.ns = 'pure_pursuit'
        m.id = 2
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r = 0.2
        m.color.g = 0.95
        m.color.b = 0.3
        m.color.a = 1.0
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.04) for p in self.traversed_points]
        self.traversed_marker_pub.publish(m)

    def _publish_target_marker(
        self,
        target_pos: Tuple[float, float],
        closest_pos: Optional[Tuple[float, float]] = None,
        lookahead_dist: float = 0.0,
        cross_track_err: float = 0.0
    ):
        """
        Publishes:
        1. Lookahead Target Sphere (Red)
        2. Lookahead Vector Ray connecting rover center -> target point
        3. Cross-Track Error Normal Line connecting rover center -> closest spline point
        4. Real-time Telemetry Text HUD above the rover
        """
        if self.car_pos is None:
            return

        now = self.get_clock().now().to_msg()
        m_array = MarkerArray()

        # 1. Lookahead Target Sphere (Red)
        sphere = Marker()
        sphere.header.frame_id = 'odom'
        sphere.header.stamp = now
        sphere.ns = 'lookahead_target'
        sphere.id = 10
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(target_pos[0])
        sphere.pose.position.y = float(target_pos[1])
        sphere.pose.position.z = 0.15
        sphere.scale.x = 0.22
        sphere.scale.y = 0.22
        sphere.scale.z = 0.22
        sphere.color.r = 1.0
        sphere.color.g = 0.15
        sphere.color.b = 0.1
        sphere.color.a = 0.95
        m_array.markers.append(sphere)

        # 2. Lookahead Steering Vector Ray (Rover Pose -> Lookahead Target)
        ray = Marker()
        ray.header.frame_id = 'odom'
        ray.header.stamp = now
        ray.ns = 'lookahead_ray'
        ray.id = 11
        ray.type = Marker.LINE_STRIP
        ray.action = Marker.ADD
        ray.scale.x = 0.03
        ray.color.r = 1.0
        ray.color.g = 0.35
        ray.color.b = 0.0
        ray.color.a = 0.85
        ray.points = [
            Point(x=float(self.car_pos[0]), y=float(self.car_pos[1]), z=0.08),
            Point(x=float(target_pos[0]), y=float(target_pos[1]), z=0.15)
        ]
        m_array.markers.append(ray)

        # 3. Cross-Track Error Normal Line (Rover Pose -> Closest Path Point)
        if closest_pos is not None:
            cte_line = Marker()
            cte_line.header.frame_id = 'odom'
            cte_line.header.stamp = now
            cte_line.ns = 'crosstrack_error'
            cte_line.id = 12
            cte_line.type = Marker.LINE_STRIP
            cte_line.action = Marker.ADD
            cte_line.scale.x = 0.03
            cte_line.color.r = 0.9
            cte_line.color.g = 0.1
            cte_line.color.b = 0.9
            cte_line.color.a = 0.9
            cte_line.points = [
                Point(x=float(self.car_pos[0]), y=float(self.car_pos[1]), z=0.06),
                Point(x=float(closest_pos[0]), y=float(closest_pos[1]), z=0.06)
            ]
            m_array.markers.append(cte_line)

        # 4. Live Rover Tracking HUD Text
        hud = Marker()
        hud.header.frame_id = 'odom'
        hud.header.stamp = now
        hud.ns = 'tracking_hud'
        hud.id = 13
        hud.type = Marker.TEXT_VIEW_FACING
        hud.action = Marker.ADD
        hud.pose.position.x = float(self.car_pos[0])
        hud.pose.position.y = float(self.car_pos[1])
        hud.pose.position.z = 0.90
        hud.scale.z = 0.18
        hud.text = f"Lookahead Ld: {lookahead_dist:.2f}m | CTE: {cross_track_err*100:.1f}cm"
        hud.color.r, hud.color.g, hud.color.b, hud.color.a = 1.0, 1.0, 0.3, 0.95
        m_array.markers.append(hud)

        self.target_marker_pub.publish(m_array)

    def _publish_waypoint_markers(self):
        """
        Publishes comprehensive waypoint visualization matching reference sketch:
        1. Crossed-Circle Target Marker (⨂): Outer circular boundary ring + 'X' & '+' crosshairs.
        2. Inner Precision Bullseye Ring & Center Landing Pin.
        3. Straight Dashed Chord Lines connecting consecutive waypoints.
        4. 3D Accuracy Downward Beacon Arrows.
        5. Floating 3D Telemetry Text Badges (with explicit "Starting", "Reached", and "Target" status).
        """
        wps_to_draw = self.abs_waypoints if self.abs_waypoints else (
            [self.start_pos] + [wp for wp in self.raw_waypoints] if self.start_pos else [(0.0, 0.0)] + [wp for wp in self.raw_waypoints]
        )
        if not wps_to_draw:
            return

        now = self.get_clock().now().to_msg()
        m_array = MarkerArray()
        total_wps = len(wps_to_draw)
        goal_tol = self.get_parameter('goal_tolerance').value
        target_radius = max(goal_tol, 0.35)

        # ── 1. Straight Dashed Chord Lines (Connecting Consecutive Waypoints) ─
        dashed_chords = Marker()
        dashed_chords.header.frame_id = 'odom'
        dashed_chords.header.stamp = now
        dashed_chords.ns = 'waypoints_chord_lines'
        dashed_chords.id = 50
        dashed_chords.type = Marker.LINE_LIST
        dashed_chords.action = Marker.ADD
        dashed_chords.scale.x = 0.035
        dashed_chords.color.r = 0.95
        dashed_chords.color.g = 0.95
        dashed_chords.color.b = 0.95
        dashed_chords.color.a = 0.80

        chord_pairs = []
        for i in range(total_wps - 1):
            chord_pairs.append((wps_to_draw[i], wps_to_draw[i + 1]))
        if self.closed_loop and total_wps > 2:
            chord_pairs.append((wps_to_draw[-1], wps_to_draw[0]))

        dash_len = 0.18
        gap_len = 0.12
        for p1, p2 in chord_pairs:
            seg_dist = euclidean_distance(p1, p2)
            if seg_dist < 1e-3:
                continue
            dx = (p2[0] - p1[0]) / seg_dist
            dy = (p2[1] - p1[1]) / seg_dist
            curr_dist = 0.0
            while curr_dist < seg_dist:
                d_start = curr_dist
                d_end = min(curr_dist + dash_len, seg_dist)
                dashed_chords.points.append(Point(x=p1[0] + dx * d_start, y=p1[1] + dy * d_start, z=0.035))
                dashed_chords.points.append(Point(x=p1[0] + dx * d_end, y=p1[1] + dy * d_end, z=0.035))
                curr_dist += dash_len + gap_len

        m_array.markers.append(dashed_chords)

        # ── 2. Individual Waypoint Target Markers (Circle + X Crosshair + Text) ─
        for i, wp in enumerate(wps_to_draw):
            # Label designation
            if i == 0:
                base_label = "Starting"
            elif i == total_wps - 1 and not self.closed_loop:
                base_label = "Goal"
            else:
                base_label = f"WP {i}"

            # Status determination
            is_reached = i < self.current_wp_idx or (i < len(self.waypoint_reached) and self.waypoint_reached[i])
            is_target = (i == self.current_wp_idx)

            if is_reached:
                # Vibrant Reached Green
                r, g, b, a = 0.10, 0.95, 0.25, 1.0
                err_cm = self.waypoint_arrival_errors.get(i, 0.0) * 100.0
                if i == 0:
                    state_text = f"Starting [ORIGIN]"
                else:
                    state_text = f"{base_label} [REACHED (Err: {err_cm:.1f}cm)]"
            elif is_target:
                # Glowing Active Target Amber/Gold
                r, g, b, a = 1.0, 0.78, 0.0, 1.0
                dist_to_wp = euclidean_distance(self.car_pos, wp) if self.car_pos else 0.0
                state_text = f"{base_label} [TARGET (Dist: {dist_to_wp:.2f}m)]"
            else:
                # Crisp Upcoming Cyan / Sky Blue
                r, g, b, a = 0.0, 0.85, 1.0, 0.85
                state_text = f"{base_label} ({wp[0]:.2f}, {wp[1]:.2f})"

            # (A) Outer Circular Boundary Ring (LINE_STRIP)
            ring = Marker()
            ring.header.frame_id = 'odom'
            ring.header.stamp = now
            ring.ns = 'waypoints_target_circle'
            ring.id = 100 + i
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.scale.x = 0.045
            ring.color.r, ring.color.g, ring.color.b, ring.color.a = r, g, b, a
            n_segments = 36
            ring.points = [
                Point(
                    x=float(wp[0] + target_radius * math.cos(2.0 * math.pi * k / n_segments)),
                    y=float(wp[1] + target_radius * math.sin(2.0 * math.pi * k / n_segments)),
                    z=0.045
                )
                for k in range(n_segments + 1)
            ]
            m_array.markers.append(ring)

            # (B) 'X' Diagonal Crosshairs + Cardinal Cross Inside Circle (LINE_LIST -> ⨂)
            cross = Marker()
            cross.header.frame_id = 'odom'
            cross.header.stamp = now
            cross.ns = 'waypoints_target_cross'
            cross.id = 200 + i
            cross.type = Marker.LINE_LIST
            cross.action = Marker.ADD
            cross.scale.x = 0.038
            cross.color.r, cross.color.g, cross.color.b, cross.color.a = r, g, b, a

            diag_r = target_radius * 0.92
            cos45 = math.cos(math.pi / 4.0) * diag_r
            sin45 = math.sin(math.pi / 4.0) * diag_r
            z_cross = 0.050

            # Diagonal 1 (\)
            cross.points.append(Point(x=float(wp[0] - cos45), y=float(wp[1] - sin45), z=z_cross))
            cross.points.append(Point(x=float(wp[0] + cos45), y=float(wp[1] + sin45), z=z_cross))
            # Diagonal 2 (/)
            cross.points.append(Point(x=float(wp[0] - cos45), y=float(wp[1] + sin45), z=z_cross))
            cross.points.append(Point(x=float(wp[0] + cos45), y=float(wp[1] - sin45), z=z_cross))
            # Cardinal Horizontal (-)
            cross.points.append(Point(x=float(wp[0] - diag_r), y=float(wp[1]), z=z_cross))
            cross.points.append(Point(x=float(wp[0] + diag_r), y=float(wp[1]), z=z_cross))
            # Cardinal Vertical (|)
            cross.points.append(Point(x=float(wp[0]), y=float(wp[1] - diag_r), z=z_cross))
            cross.points.append(Point(x=float(wp[0]), y=float(wp[1] + diag_r), z=z_cross))
            m_array.markers.append(cross)

            # (C) Inner Precision Bullseye Ring
            inner_ring = Marker()
            inner_ring.header.frame_id = 'odom'
            inner_ring.header.stamp = now
            inner_ring.ns = 'waypoints_bullseye_inner'
            inner_ring.id = 300 + i
            inner_ring.type = Marker.LINE_STRIP
            inner_ring.action = Marker.ADD
            inner_ring.scale.x = 0.035
            inner_ring.color.r, inner_ring.color.g, inner_ring.color.b, inner_ring.color.a = r, g, b, 0.90
            r_inner = target_radius * 0.45
            inner_ring.points = [
                Point(
                    x=float(wp[0] + r_inner * math.cos(2.0 * math.pi * k / 24)),
                    y=float(wp[1] + r_inner * math.sin(2.0 * math.pi * k / 24)),
                    z=0.052
                )
                for k in range(25)
            ]
            m_array.markers.append(inner_ring)

            # (D) Center Landing Pin (White Dot)
            pin = Marker()
            pin.header.frame_id = 'odom'
            pin.header.stamp = now
            pin.ns = 'waypoints_pin_center'
            pin.id = 400 + i
            pin.type = Marker.CYLINDER
            pin.action = Marker.ADD
            pin.pose.position.x = float(wp[0])
            pin.pose.position.y = float(wp[1])
            pin.pose.position.z = 0.060
            pin.scale.x = 0.08
            pin.scale.y = 0.08
            pin.scale.z = 0.04
            pin.color.r, pin.color.g, pin.color.b, pin.color.a = 1.0, 1.0, 1.0, 1.0
            m_array.markers.append(pin)

            # (E) 3D Downward Accuracy Beacon Pointer
            ptr = Marker()
            ptr.header.frame_id = 'odom'
            ptr.header.stamp = now
            ptr.ns = 'waypoints_accuracy_pointer'
            ptr.id = 500 + i
            ptr.type = Marker.ARROW
            ptr.action = Marker.ADD
            ptr.scale.x = 0.06   # Shaft diameter
            ptr.scale.y = 0.18   # Head diameter
            ptr.scale.z = 0.24   # Head length
            ptr.color.r, ptr.color.g, ptr.color.b, ptr.color.a = r, g, b, 0.95
            ptr.points = [
                Point(x=float(wp[0]), y=float(wp[1]), z=1.10),
                Point(x=float(wp[0]), y=float(wp[1]), z=0.08)
            ]
            m_array.markers.append(ptr)

            # (F) Floating 3D Text Label
            txt = Marker()
            txt.header.frame_id = 'odom'
            txt.header.stamp = now
            txt.ns = 'waypoints_text'
            txt.id = 600 + i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(wp[0])
            txt.pose.position.y = float(wp[1])
            txt.pose.position.z = 1.30
            txt.scale.z = 0.22
            txt.text = state_text
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = 1.0, 1.0, 1.0, 0.95
            m_array.markers.append(txt)

        self.waypoint_marker_pub.publish(m_array)

    def _publish_obstacle_bubbles(self):
        """Publishes safety boundary bubble markers around detected obstacles."""
        now = self.get_clock().now().to_msg()
        m_array = MarkerArray()
        obs_clearance = self.get_parameter('obstacle_clearance').value

        for i, obs in enumerate(self.detected_obstacles):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = now
            m.ns = 'obstacle_clearance'
            m.id = 100 + i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(obs.x)
            m.pose.position.y = float(obs.y)
            m.pose.position.z = 0.1
            diam = 2.0 * (obs.radius + obs_clearance)
            m.scale.x = float(diam)
            m.scale.y = float(diam)
            m.scale.z = 0.05
            m.color.r = 1.0
            m.color.g = 0.6
            m.color.b = 0.0
            m.color.a = 0.35
            m_array.markers.append(m)

        self.obstacle_bubble_pub.publish(m_array)


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def main(args=None):
    # Keep the context alive through Ctrl-C so the final zero twist is actually
    # delivered before DDS is torn down, the same contract the ODrive bridge uses.
    rclpy.init(args=args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)

    raw_argv = sys.argv if args is None else args
    non_ros_argv = rclpy.utilities.remove_ros_args(args=raw_argv)[1:]
    cfg = parse_cli_args(non_ros_argv)

    node = TEBPurePursuitPlannerNode(cfg)

    shutdown_requested = False

    def _sigint_handler(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while rclpy.ok() and not shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.get_logger().info("Shutting down planner — sending stop command")
        node.estopped = True
        for _ in range(10):
            node._publish_raw(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.02)
        node.shutdown_drive()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
