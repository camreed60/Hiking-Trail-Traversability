#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry

class PoseListener:
    def __init__(self):
        self.vehicleX = 0.0
        self.vehicleY = 0.0
        self.vehicleZ = 0.0

        rospy.Subscriber('/state_estimation', Odometry, self.pose_callback)

    def pose_callback(self, msg):
        self.vehicleX = msg.pose.pose.position.x
        self.vehicleY = msg.pose.pose.position.y
        self.vehicleZ = msg.pose.pose.position.z

    def get_vehicle_position(self):
        return self.vehicleX, self.vehicleY, self.vehicleZ

    def spin(self):
        rospy.spin()  # Keeps the node running until terminated