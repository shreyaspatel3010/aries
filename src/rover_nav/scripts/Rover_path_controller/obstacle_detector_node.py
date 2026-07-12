#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
from sklearn.cluster import DBSCAN
import math
from utils import distance, transform_to_local, yaw_from_quaternion
from rover_nav.msg import Obstacle, ObstacleArray


obstacle_pub = None



# ═══════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════
def scan_callback(scan_msg):
    # Called when new LiDAR data arrives
    # Calls: scan_to_xy → cluster_points → extract_obstacles → publish_obstacles

    points  = scan_to_xy(scan_msg)





# ═══════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════
def scan_to_xy(scan):
    # Convert LaserScan → XY points
    # Returns: numpy array (N, 2)

    ranges = scan.ranges
    
    angle = scan.angle_min

    points = np.zeros((len(ranges)), 2)
    
    for r in ranges:
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        points[(angle/(scan.angle_increment))] = [x,y]
            
        angle+= scan.angle_increment 

    return points

    


def cluster_points(points):
    # Run DBSCAN on points

    # Returns: cluster labels
    pass


def extract_obstacles(points, labels):
    # Convert clusters → obstacles (center + radius)
    # Returns: list of obstacles
    pass


def publish_obstacles(obstacles):
    # Publish to /obstacles topic
    pass



def main(args = None):
    # Setup ROS2 node, subscriber, publisher
    
    rclpy.init(args=args)
    node = rclpy.create_node("detectObstacles")

    node.create_subscription(LaserScan, '/scan', 10, scan_callback )

    obstacle_pub = node.create_publisher(ObstacleArray, 'obstacles', 10)

if __name__ == '__main__':
    main()