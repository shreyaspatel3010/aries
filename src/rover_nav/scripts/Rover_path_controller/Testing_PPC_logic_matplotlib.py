#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Constants ---
LOOKAHEAD = 1.0  # Lookahead distance
DT = 0.1  # Time step
LINEAR_SPEED = 1.0  # Constant forward speed [m/s]
ROBOT_RADIUS = 0.3  # For visualization
MAX_ANGULAR_SPEED = np.pi / 4  # Max turning rate (rad/s) ~45 deg/s

# --- Waypoints in a circular arc, more spaced ---
angle_start = np.pi / 4  # 45 degrees
angle_end = 3 * np.pi / 4  # 135 degrees
num_points = 5  # fewer points, more spaced out
radius = 12.0  # bigger radius, points further away

waypoints = [
    np.array([
        radius * np.cos(angle_start + i * (angle_end - angle_start) / (num_points - 1)),
        radius * np.sin(angle_start + i * (angle_end - angle_start) / (num_points - 1))
    ])
    for i in range(num_points)
]

# --- Robot initial state ---
x, y, yaw = 0.0, 0.0, 0.0

trajectory_x = [x]
trajectory_y = [y]

current_goal_index = 0

def point_global_to_local(point_global, yaw, position):
    dx = point_global[0] - position[0]
    dy = point_global[1] - position[1]
    local_x = dx * np.cos(yaw) + dy * np.sin(yaw)
    local_y = dy * np.cos(yaw) - dx * np.sin(yaw)
    return local_x, local_y

def calc_curvature(point_local_y, lookahead=LOOKAHEAD):
    return (2 * point_local_y) / (lookahead ** 2)

def generate_command(curvature, linear_speed=LINEAR_SPEED):
    angular_speed = curvature * linear_speed
    # Limit angular speed to max turn rate
    if angular_speed > MAX_ANGULAR_SPEED:
        angular_speed = MAX_ANGULAR_SPEED
    elif angular_speed < -MAX_ANGULAR_SPEED:
        angular_speed = -MAX_ANGULAR_SPEED
    return linear_speed, angular_speed

def draw_robot(ax, x, y, yaw, radius=ROBOT_RADIUS):
    pts = np.array([
        [radius, 0],
        [-radius * 0.5, radius * 0.7],
        [-radius * 0.5, -radius * 0.7],
        [radius, 0],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R_mat = np.array([[c, -s], [s, c]])
    pts_rot = pts @ R_mat.T
    pts_rot[:, 0] += x
    pts_rot[:, 1] += y
    ax.plot(pts_rot[:, 0], pts_rot[:, 1], 'k-')

def update(frame):
    global x, y, yaw, current_goal_index

    if current_goal_index >= len(waypoints):
        ax.set_title("All goals reached!")
        return

    goal = waypoints[current_goal_index]
    dist_to_goal = np.linalg.norm([x - goal[0], y - goal[1]])

    if dist_to_goal < 0.3:
        current_goal_index += 1
        if current_goal_index >= len(waypoints):
            ax.set_title("All goals reached!")
            return
        goal = waypoints[current_goal_index]

    local_x, local_y = point_global_to_local(goal, yaw, [x, y])
    curvature = calc_curvature(local_y)
    v, w = generate_command(curvature)

    # Differential-drive kinematics with angular speed limit
    x += v * np.cos(yaw) * DT
    y += v * np.sin(yaw) * DT
    yaw += w * DT

    # Normalize yaw angle between -pi and pi
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    trajectory_x.append(x)
    trajectory_y.append(y)

    ax.clear()
    ax.plot(trajectory_x, trajectory_y, 'b-', label="Robot Path")

    for i, wp in enumerate(waypoints):
        ax.plot(wp[0], wp[1], 'go' if i != current_goal_index else 'ro')

    draw_robot(ax, x, y, yaw)

    # Dynamic plot limits based on trajectory and waypoints
    all_x = trajectory_x + [wp[0] for wp in waypoints]
    all_y = trajectory_y + [wp[1] for wp in waypoints]

    margin = 2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Pure Pursuit: Goal {current_goal_index+1}/{len(waypoints)} - Distance: {dist_to_goal:.2f} m")

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=range(1500), interval=50)

plt.show()
