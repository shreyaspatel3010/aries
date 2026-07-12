#!/usr/bin/env python3
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
from utils import distance, transform_to_local, yaw_from_quaternion
from rover_nav.msg import Obstacle, ObstacleArray


# ═══════════════════════════════════════════════════════════════
# GLOBAL VARIABLES
# ═══════════════════════════════════════════════════════════════

# Rover state (from odom)
car_yaw = None
car_global_axis = None

# Goal
goal_point = [7, 11]

# Obstacles (updated by obstacle callback)
current_obstacles = []

# Publishers
vel_pub = None

# ═══════════════════════════════════════════════════════════════
# PARAMETERS (TUNE THESE)
# ═══════════════════════════════════════════════════════════════

LA = 1.0                    # Lookahead distance
WHEEL_BASE = 0.6            # Wheel base

DETECTION_RANGE = 2.0       # How far ahead to look for obstacles (m)
DETECTION_ANGLE = 45.0      # Cone half-angle (degrees)
SAFETY_MARGIN = 0.3         # Buffer around obstacles (m)
SHIFT_ANGLE = 30.0          # How much to shift target (degrees)

BASE_VELOCITY = 0.5         # Normal speed (m/s)
AVOIDANCE_VELOCITY = 0.25   # Speed while avoiding (m/s)
GOAL_TOLERANCE = 0.3        # Stop when this close to goal (m)


# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════

def odom_callback(odom):
    """Update rover position and heading from odometry."""
    global car_yaw
    global car_global_axis

    qx = odom.pose.pose.orientation.x
    qy = odom.pose.pose.orientation.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w

    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y

    car_global_axis = [x, y]

    r = R.from_quat([qx, qy, qz, qw])
    _, _, car_yaw = r.as_euler('xyz')


def obstacles_callback(obstacles_msg):
    """Update current obstacle list from obstacle detector."""
    global current_obstacles
    
    # TO DO: Parse obstacles_msg and update current_obstacles
    # Format: list of dicts [{'center': (x, y), 'radius': r}, ...]
    pass


# ═══════════════════════════════════════════════════════════════
# AVOIDANCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_blocking_obstacle():
    """
    Check if any obstacle is blocking our path.
    
    Returns:
        dict: blocking obstacle {'center': (x,y), 'radius': r, 'local_y': ly}
        None: if path is clear
    """
    global current_obstacles, car_global_axis, car_yaw
    
    # TODO: Implement
    # For each obstacle:
    #   1. Transform to local frame
    #   2. Check if in front (local_x > 0)
    #   3. Check if within DETECTION_RANGE
    #   4. Check if within DETECTION_ANGLE
    #   5. Return closest blocking obstacle
    

    # QUESTIONS:
    # Are the coordinates in the dictionary relative to local position?

    # 1) call the function to get the array with the coordenades and the radius
    # I'm expecting something like this: ex:[(2, 3, 1), (4, 5, 6), (7, 8, 10)]

    rover_x, rover_y = car_global_axis
    rover_yaw = car_yaw

    resultado = []

    for (x, y, r) in lista_objetos:  
        local = transform_to_local(x, y, rover_x, rover_y, rover_yaw)
        resultado.append(local)

    return resultado



    pass


def calculate_target(blocking_obstacle):
    """
    Calculate target point (shifted if obstacle, else goal).
    
    Args:
        blocking_obstacle: obstacle dict or None
    
    Returns:
        tuple: (target_x, target_y, velocity)
    """
    global goal_point, car_global_axis, car_yaw
    
    # TODO: Implement
    # If no obstacle:
    #   return (goal_x, goal_y, BASE_VELOCITY)
    # If obstacle:
    #   1. Get direction to goal
    #   2. Shift direction away from obstacle
    #   3. Calculate shifted target point
    #   return (shifted_x, shifted_y, AVOIDANCE_VELOCITY)
    
    pass


# ═══════════════════════════════════════════════════════════════
# PURE PURSUIT FUNCTIONS (YOUR EXISTING CODE)
# ═══════════════════════════════════════════════════════════════

def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def point_global_to_local(point_global_axis, car_yaw, car_axis):
    """Transform point from global to rover's local frame."""
    dx = point_global_axis[0] - car_axis[0]
    dy = point_global_axis[1] - car_axis[1]

    point_local_x = dy * np.sin(car_yaw) + dx * np.cos(car_yaw)
    point_local_y = dy * np.cos(car_yaw) - dx * np.sin(car_yaw)

    return point_local_x, point_local_y


def calc_curv(point_local_y, look_ahead=LA):
    """Calculate curvature for pure pursuit."""
    curvature = (2 * point_local_y) / (look_ahead * look_ahead)
    return curvature


def generate_command(curvature, linear_speed):
    """Generate Twist command from curvature and speed."""
    angular_speed = curvature * linear_speed

    cmd = Twist()
    cmd.linear.x = float(linear_speed)
    cmd.angular.z = float(angular_speed)

    return cmd


# ═══════════════════════════════════════════════════════════════
# MAIN CONTROL LOOP
# ═══════════════════════════════════════════════════════════════

def control_loop(node):
    """Main control loop - called by timer."""
    global car_yaw, car_global_axis, goal_point, vel_pub

    # Wait for odometry
    if car_yaw is None or car_global_axis is None:
        node.get_logger().warn("Waiting for odometry...")
        return

    # Check if reached goal
    dist_to_goal = distance(car_global_axis, goal_point)
    if dist_to_goal < GOAL_TOLERANCE:
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        vel_pub.publish(cmd)
        node.get_logger().info("Goal reached!")
        return

    # Step 1: Find blocking obstacle
    blocking_obstacle = find_blocking_obstacle()

    # Step 2: Calculate target (shifted or goal)
    target_x, target_y, velocity = calculate_target(blocking_obstacle)

    # Step 3: Transform target to local frame
    local_x, local_y = point_global_to_local(
        [target_x, target_y], car_yaw, car_global_axis
    )

    # Step 4: Calculate curvature and command
    curv = calc_curv(local_y)
    cmd = generate_command(curv, velocity)

    # Step 5: Publish
    vel_pub.publish(cmd)

    # Logging
    if blocking_obstacle:
        node.get_logger().info(f"AVOIDING - vel: {cmd.linear.x:.2f}, ang: {cmd.angular.z:.2f}")
    else:
        node.get_logger().info(f"NORMAL - vel: {cmd.linear.x:.2f}, ang: {cmd.angular.z:.2f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(args=None):
    global vel_pub

    rclpy.init(args=args)
    node = rclpy.create_node("pure_pursuit_avoidance")

    # Subscribers
    node.create_subscription(Odometry, "/odom", odom_callback, 10)
    
    # TO DO: Subscribe to obstacles topic
    node.create_subscription(ObstacleArray, '/obstacles', obstacles_callback, 10)

    # Publisher
    vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)

    # Timer (10 Hz control loop)
    timer = node.create_timer(0.1, lambda: control_loop(node))

    node.get_logger().info("Pure Pursuit with Avoidance started!")

    rclpy.spin(node)
    node.destroy_timer(timer)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()