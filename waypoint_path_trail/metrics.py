#!/usr/bin/env python3

import rospy
import time
from pose_listener import PoseListener
from std_msgs.msg import Float32 
import threading

class MetricCollection:
    def __init__(self, travs_quad1, travs_quad2, travs_quad3, travs_quad4, scale):
        self.start_time = None
        self.total_time = None
        self.traversability_map_values = [travs_quad1, travs_quad2, travs_quad3, travs_quad4, scale]
        self.pose_object = PoseListener()
        self.trail_counter = 0
        self.start_dist = None
        self.total_distance_traversed = None
        self.sim_total_distance_traversed = 0.0
        self.stop_thread = False

        # ROS subscriber
        rospy.Subscriber("/traveling_distance", Float32, self.travel_distance_callback)

    def travel_distance_callback(self, msg):
        self.sim_total_distance_traversed = msg.data

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        self.total_time = time.time() - self.start_time

    def compare_position_with_travs(self, x_value, y_value):
        # Scale x and y
        scaled_x = x_value * self.traversability_map_values[4]
        scaled_y = y_value * self.traversability_map_values[4]
        scaled_x = int(scaled_x)
        scaled_y = int(scaled_y)
        # Check which quadrant the x and y value belong to
        # If X and Y are positive, set traversability map to quadrant 1
        if scaled_x >= 0 and scaled_y >= 0:
            travs_map = self.traversability_map_values[0]
        # If X is negative and Y is positive, set traversability map to quadrant 2
        elif scaled_x < 0 and scaled_y >= 0:
            travs_map = self.traversability_map_values[1]
        # If X and Y are negative, set traversability map to quadrant 3
        elif scaled_x < 0 and scaled_y < 0:
            travs_map = self.traversability_map_values[2]
        # If X is positive and Y is negative, set traversability map to quadrant 4
        else:
            travs_map = self.traversability_map_values[3]
        # Find the position's traversability value
        value = travs_map[abs(scaled_y)][abs(scaled_x)]
        # If the currently position is in a traversable region, return true
        if value == 1:
            return True
        else:
            return False
    
    def time_on_trail_controller(self):
        # In a thread, continuously get the current position of the agent
        # and compare with traversability
        while not self.stop_thread:
            vehicleX, vehicleY, vehicleZ = self.pose_object.get_vehicle_position()
            if self.compare_position_with_travs(vehicleX, vehicleY):
                self.trail_counter += 1
            time.sleep(0.1)

    def start_time_on_trail(self):
        self.trail_thread = threading.Thread(target=self.time_on_trail_controller)
        self.trail_thread.daemon = True
        self.trail_thread.start()

    def end_time_on_trail(self):
        self.stop_thread = True
        if self.trail_thread.is_alive():
            self.trail_thread.join()
        self.percent_time_on_trail = (self.trail_counter * 0.1) / self.total_time * 100
    
    def start_distance_traversed(self):
        self.start_dist = self.sim_total_distance_traversed
    
    def end_distance_traversed(self):
        self.total_distance_traversed = self.sim_total_distance_traversed - self.start_dist
        return self.total_distance_traversed