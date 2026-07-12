
#!/usr/bin/env python3

import rclpy 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String , Float32
from geometry_msgs.msg import Point 
from sensor_msgs.msg import Imu
import math
import os
from pathlib import Path

''' 1 > the initial position of car known from ips sensor data  
    2 > the target here is one goal point to verfiy the pure pursuit control
    start with a one goal point (selected based on a lookahead assumed ) located on the y axis of the car 
    3 > impelement the pure pursuit formula to get the steering angle value
'''

#-----------------------Global variables-------------------------------- 

## car current position & orientation 
x_postition =  0.748
y_postition =  3.16
postition = np.array([x_postition , y_postition ])

car_yaw = 0.0

# Centerline Path of CDC Practice
default_path_csv = (
    Path(__file__).resolve().parent
    / "csv_paths_practice_cdc"
    / "Centerline_less_points.csv"
)
path_data = pd.read_csv(os.environ.get("PURE_PURSUIT_PATH_CSV", default_path_csv))

#x_offset = 4.71
#y_offset = -5.03

goal_list = list(zip(
    (path_data['positions_X']  ) , # - x_offset + x_postition
    (path_data['positions_y']  )   # - y_offset + y_postition
    ))
goal = np.array(goal_list)

# pure pursuit parameter 
velocity = 0.08  
look_ahead = 1 ## with thr 0.08  >> 1 
change_point = look_ahead + 0.5
wheelbase = 0.3240 

count = 150


# ---------- Live Plotting Setup -------------

car_path_x = []
car_path_y = []

plt.ion()
fig, ax = plt.subplots()

# Red dot for current position
sc, = ax.plot([], [], 'ro')

# Green X for goal point
# Plot all waypoints as green Xs
goal_markers = ax.plot(goal[:, 1],goal[:, 0],  'm.')[0]

# Line to draw full path
path_line, = ax.plot([], [], 'c--', linewidth=1.5)

ax.set_title("Live Car Trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)



#-----------------------call back functions-------------------------------- 
def pose_callback (ips_msg):
    global postition

    postition[0] =  ips_msg.x  
    postition[1] =  ips_msg.y 
    
    
def yaw_callback (imu_msg) : 
    global car_yaw 

    ## Quaternion from imu data 
    qx = imu_msg.orientation.x
    qy = imu_msg.orientation.y
    qz = imu_msg.orientation.z
    qw = imu_msg.orientation.w
    ## convert to euler to get yaw 
    r = R.from_quat ([qx,qy,qz,qw])
    roll , pitch , car_yaw = r.as_euler('xyz')
    #print(f"Euler angles: roll={roll}, pitch={pitch}, yaw={car_yaw}")


#-------------------------Pure pursuit functions--------------------------------

def transformation ( xy_world_arr , point_world_arr , yaw  ) : 

    R_T = np.array([ [np.cos(yaw) , np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]   ])
    
    point_car_frame = R_T @ (point_world_arr-xy_world_arr)
    return  point_car_frame

def curvature_calc (xy_car_frame) :
    x = xy_car_frame[0]
    y = xy_car_frame[1]
    curvature = (2* y) / (look_ahead*look_ahead)  ## without abs()
    return curvature

def steering_func (wh_base , gamma) :

    steering_angle = np.arctan(wh_base *gamma) 
    
    """Avoid steering more than maximum"""
    #if (steering_angle >=  0.5236):
    #    steering_angle = 1
    #elif (steering_angle <= -0.5236):
    #    steering_angle = -1

    return steering_angle




#-------------------- ROS 2 Timer_function ------------------------
def timer_func( node, st_pub , thr_pub ) : 
    global  postition , car_yaw  , count
    st = Float32()
    thr = Float32()

    node.get_logger().info("Publishing : >_<" )
    #if np.isclose(10.748, postition[0], atol=0.1) and np.isclose(1.580, postition[1], atol=0.1):
    #    print(f"Goal Reached at x={postition[0]}, y={postition[1]}")

    #    # Stop motion
    #    st.data = 0.0
    #    thr.data = 0.0
    #    st_pub.publish(st)
    #    thr_pub.publish(thr)
    #    return  # stop this timer iteration
        
 

    dist = np.linalg.norm( goal[count]-postition ) 
    if dist <= change_point :  ### with 0.08 1.5 
        count += 1 
    ##### >> interpolation 

    if ( count >= len(goal) ):
        count = 0

    xy_cf = transformation( postition , goal[count] ,car_yaw )
    curve = curvature_calc ( xy_cf )
    steer = steering_func( wheelbase, curve )
    st.data =  float(steer)


    # #math.sqrt(3.906*9.81*1/abs(curve)) 
    
    thr_msg = velocity
    thr.data = thr_msg
    st_pub.publish(st) 
    thr_pub.publish(thr) 
        
    node.get_logger().info(f" yaw angle  : {round(car_yaw,3)} " )
    node.get_logger().info(f" steeing command value : {round(steer,3)} >_<" )
    node.get_logger().info(f" throttle command value : {thr_msg} >_<" )
    node.get_logger().info(f" Lookahead  : {look_ahead} >_<" )
    node.get_logger().info(f" change point  : {change_point} >_<" )

        # --- Live plot update ---
    car_path_x.append(postition[0])
    car_path_y.append(postition[1])

    sc.set_data(postition[1] , postition[0] )  # red dot
    path_line.set_data( car_path_y , car_path_x)  # trajectory line

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


#------------------------- Main ------------------------
def main (args=None):    
    

    #   Node creation

    rclpy.init(args=args)    
    my_node = rclpy.create_node('control_node')

    #  subscribers

    car_pose = my_node.create_subscription(Point , '/autodrive/f1tenth_1/ips' , pose_callback , 10)
    ips_msg = Point()    
    imu_sub = my_node.create_subscription(Imu , '/autodrive/f1tenth_1/imu' , yaw_callback , 10)
    imu_msg = Imu()
    steer_pub = my_node.create_publisher( Float32 , "/autodrive/f1tenth_1/steering_command" , 10 )  
    st_msg = Float32()
    throttle_pub = my_node.create_publisher( Float32 , "/autodrive/f1tenth_1/throttle_command" , 10 )
    th_msg = Float32()


    timer = my_node.create_timer ( 0.01 , lambda:timer_func(my_node,steer_pub,throttle_pub) )   #  100 hz  
    rclpy.spin(my_node)
    my_node.destroy_timer(timer)
    my_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__" :    # the entry of the code     


    main()      
