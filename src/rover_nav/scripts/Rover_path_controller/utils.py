#!/usr/bin/env python3

import math
from scipy.spatial.transform import Rotation as R


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def yaw_from_quaternion(qx, qy, qz, qw):
    r = R.from_quat([qx, qy, qz, qw])
    _, _, yaw = r.as_euler('xyz')
    return yaw


def transform_to_local(point_x, point_y, rover_x, rover_y, rover_yaw):
    dx = point_x - rover_x
    dy = point_y - rover_y
    local_x = dy * math.sin(rover_yaw) + dx * math.cos(rover_yaw)
    local_y = dy * math.cos(rover_yaw) - dx * math.sin(rover_yaw)
    return local_x, local_y



'''

OBSTACLE DETECTOR:
──────────────────
scan_callback()
    │
    ├── scan_to_xy()
    │
    ├── cluster_points()
    │
    ├── extract_obstacles()
    │
    └── publish_obstacles()


PURE PURSUIT + AVOIDANCE:
─────────────────────────
control_loop()  ← timer calls this
    │
    ├── find_blocking_obstacle()
    │
    ├── calculate_target()
    │
    ├── pure_pursuit()
    │
    └── publish_cmd_vel()

'''