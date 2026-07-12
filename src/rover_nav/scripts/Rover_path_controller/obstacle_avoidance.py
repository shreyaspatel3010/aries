#!/usr/bin/env python3

import rclpy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math



#for only having lidar points between -90 to 90

def filter_lidar_readings(): #tested
    node = rclpy.create_node('lidar_front_180')
    node.get_logger().info('Node /scan_180 starting')
    pub = node.create_publisher(LaserScan, '/scan_180', 10)
    def scan_callback(msg: LaserScan):
        total_points = len(msg.ranges)
        if total_points < 180:
            node.get_logger().warn(' Few points available, try again :( )')
            return
        # lidar points between -90 to 90
        front_ranges = msg.ranges[-90:] + msg.ranges[:90]
        front_intensities = msg.intensities[-90:] + msg.intensities[:90]
        # create a new message
        filtered = LaserScan()
        filtered.header = msg.header
        filtered.ranges = front_ranges
        filtered.intensities = front_intensities
        filtered.range_min = msg.range_min
        filtered.range_max = msg.range_max
        filtered.scan_time = msg.scan_time
        filtered.time_increment = msg.time_increment
        filtered.angle_increment = msg.angle_increment
        filtered.angle_min = -math.pi / 2   # -90°
        filtered.angle_max = math.pi / 2    # +90°
        pub.publish(filtered)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Finishing node /scan_180...')
    finally:
        node.destroy_node()



def define_nearest_obstacle():
    pass



#detect where the dispaities are
def find_disparity():
    pass



#dor each disparity extend it half the width of the car 
def extend_disparity():
    pass



#after 
def choose_path():
    pass




def main(args =  None):
    rclpy.init(args=args)
    filter_lidar_readings()
    rclpy.shutdown()
    


if __name__ == '__main__':
    main()