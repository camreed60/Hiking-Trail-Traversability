#!/usr/bin/env python
# Metric Collection Script for ROS

import rospy
import time
import threading
import numpy as np
from pose_listener import PoseListener  # Custom module to listen to vehicle poses
from std_msgs.msg import Float32, String  # ROS standard message types
from sensor_msgs.msg import PointCloud2  # ROS sensor message for point clouds
import sensor_msgs.point_cloud2 as pc2  # Helper functions to work with PointCloud2 messages

class MetricCollection:
    def __init__(self, start_position, end_position, position_threshold=1):
        # Convert start and end positions to NumPy arrays for easier computation.
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        self.position_threshold = position_threshold

        # Timer and distance attributes.
        self.start_time = None
        self.total_time = None
        self.start_dist = None
        self.total_distance_traversed = None
        self.sim_total_distance_traversed = None

        # Initialize the PoseListener to obtain the vehicle's position.
        self.pose_object = PoseListener()

        # Trail detection attributes.
        self.trail_counter = 0  # Counts the number of samples where the vehicle is on the trail.
        self.stop_thread = False  # Flag to stop the background thread.
        self.collection_started = False  # Flag to indicate if collection has started.
        self.collection_ended = False  # Flag to indicate if collection has ended.

        # Point cloud related attributes.
        self.latest_cloud = None  # Will store the latest received point cloud message.
        self.lock = threading.Lock()  # Lock for thread-safe access to point cloud data.

        # Sampling rate (in Hz) for checking if the vehicle is on the trail.
        self.sampling_rate = 10

        # Subscribe to ROS topics for traveling distance and point cloud data.
        rospy.Subscriber("/traveling_distance", Float32, self.travel_distance_callback)
        rospy.Subscriber("/terrain_truth", PointCloud2, self.point_cloud_callback)

        # Publisher to output debug information related to trail detection.
        self.debug_pub = rospy.Publisher('/trail_detection_debug', String, queue_size=10)

    # Callback for the /traveling_distance topic.
    # Stores the simulated total distance traversed.
    def travel_distance_callback(self, msg):
        self.sim_total_distance_traversed = msg.data

    # Callback for the /terrain_truth point cloud topic.
    # Saves the latest point cloud and calls the debug function.
    def point_cloud_callback(self, cloud_msg):
        with self.lock:
            self.latest_cloud = cloud_msg
        self.debug_point_cloud()

    # Computes and publishes basic statistics about the current point cloud.
    def debug_point_cloud(self):
        if self.latest_cloud:
            point_count = 0
            # Initialize min and max values for x, y, and z coordinates.
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            min_z, max_z = float('inf'), float('-inf')
            # Iterate through each point in the point cloud.
            for point in pc2.read_points(self.latest_cloud, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True):
                point_count += 1
                x, y, z = point[:3]
                # Update the minimum and maximum values for each axis.
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
            # Construct a debug message with the point cloud statistics.
            debug_msg = f"Point cloud stats: {point_count} points, X: [{min_x:.2f}, {max_x:.2f}], Y: [{min_y:.2f}, {max_y:.2f}], Z: [{min_z:.2f}, {max_z:.2f}]"
            self.debug_pub.publish(String(debug_msg))
        else:
            print("No point cloud data available")

    # Determines whether a given position is on the trail based on the point cloud data.
    def is_on_trail(self, position):
        with self.lock:
            if self.latest_cloud is None:
                print("No point cloud data available")
                self.debug_pub.publish(String("No point cloud data available"))
                return False
            
            # Extract position coordinates.
            x, y, z = position
            search_radius = 0.05  # Tolerance in meters for finding nearby points.
            on_trail_points = 0  # Count of points that indicate being on the trail.
            total_points = 0  # Total points within the search radius.
            closest_point_distance = float('inf')
            closest_point_color = None

            # Iterate over the points in the point cloud.
            for point in pc2.read_points(self.latest_cloud, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True):
                px, py, pz, r, g, b = point
                # Compute Euclidean distance from the current position to the point.
                distance = ((px - x)**2 + (py - y)**2 + (pz - z)**2)**0.5
                
                # Keep track of the closest point for debugging purposes.
                if distance < closest_point_distance:
                    closest_point_distance = distance
                    closest_point_color = (r, g, b)
                
                # If the point is within the search radius, check its color.
                if distance <= search_radius:
                    total_points += 1
                    # Determine if the point's color suggests it is part of the trail.
                    if r > 0.5 and g > 0.5 and b > 0.5:
                        on_trail_points += 1

            # If no points were found within the search radius, output debug info and return False.
            if total_points == 0:
                debug_msg = f"No points found in search radius. Closest point: dist={closest_point_distance:.2f}, color={closest_point_color}"
                print(debug_msg)
                self.debug_pub.publish(String(debug_msg))
                return False

            # Calculate the ratio of on-trail points within the search radius.
            trail_ratio = on_trail_points / total_points
            # The position is considered on the trail if the ratio exceeds a threshold (90%).
            is_on_trail = trail_ratio > 0.9

            # Publish detailed debug information.
            debug_msg = (f"On trail: {is_on_trail}, ratio: {trail_ratio:.2f}, "
                         f"points: {on_trail_points}/{total_points}, "
                         f"closest point: dist={closest_point_distance:.2f}, color={closest_point_color}")
            self.debug_pub.publish(String(debug_msg))
            return is_on_trail

    # Records the current time as the start time of metric collection.
    def start_timer(self):
        self.start_time = time.time()

    # Calculates the total elapsed time since metric collection started.
    def end_timer(self):
        self.total_time = time.time() - self.start_time

    # Background controller that periodically checks whether the vehicle is on the trail.
    # This function runs in a separate thread.
    def time_on_trail_controller(self):
        rate = rospy.Rate(self.sampling_rate)
        while not self.stop_thread:
            # Get the current vehicle position from the PoseListener.
            vehicleX, vehicleY, vehicleZ = self.pose_object.get_vehicle_position()
            # Check if the current position is on the trail using point cloud data.
            if self.is_on_trail([vehicleX, vehicleY, vehicleZ]):
                self.trail_counter += 1
            rate.sleep()

    # Starts the background thread to monitor the time the vehicle stays on the trail.
    def start_time_on_trail(self):
        self.trail_thread = threading.Thread(target=self.time_on_trail_controller)
        self.trail_thread.daemon = True
        self.trail_thread.start()

    # Stops the background thread for trail monitoring and computes the percentage
    # of total time that the vehicle was on the trail.
    def end_time_on_trail(self):
        self.stop_thread = True
        if self.trail_thread.is_alive():
            self.trail_thread.join()
        self.percent_time_on_trail = (self.trail_counter / self.sampling_rate) / self.total_time * 100 if self.total_time else 0
    
    # Waits for the simulated distance data to be available and records the starting distance.
    def start_distance_traversed(self):
        while self.sim_total_distance_traversed is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        self.start_dist = self.sim_total_distance_traversed
    
    # Computes the total distance traversed by subtracting the starting distance
    # from the final simulated distance.
    def end_distance_traversed(self):
        self.total_distance_traversed = self.sim_total_distance_traversed - self.start_dist
        return self.total_distance_traversed

    # Checks the vehicle's current position against the start and end positions.
    def check_position(self):
        current_position = np.array(self.pose_object.get_vehicle_position())
        
        # Check if collection has not started yet.
        if not self.collection_started:
            distance_to_start = np.linalg.norm(current_position - self.start_position)
            rospy.loginfo(f"Distance to start: {distance_to_start:.2f} m")
            if distance_to_start <= self.position_threshold:
                self.start_collection()
        # If collection is ongoing, check for the end position.
        elif not self.collection_ended:
            distance_to_end = np.linalg.norm(current_position - self.end_position)
            rospy.loginfo(f"Distance to end: {distance_to_end:.2f} m")
            if distance_to_end <= self.position_threshold:
                self.end_collection()

    # Begins metric collection by initializing timers, distance tracking, and the
    # background thread for trail detection.
    def start_collection(self):
        self.collection_started = True
        self.start_timer()              # Start the overall timer.
        self.start_time_on_trail()      # Start the thread that monitors time on trail.
        self.start_distance_traversed() # Record the starting distance.
        rospy.loginfo("Metric collection started!")

    # Ends metric collection by stopping timers and threads, then computing and printing results.
    def end_collection(self):
        self.collection_ended = True
        self.end_timer()            # Calculate total elapsed time.
        self.end_time_on_trail()    # Stop trail monitoring and compute trail time percentage.
        self.end_distance_traversed()  # Calculate total distance traversed.
        self.stop_thread = True     # Ensure the background thread is stopped.
        rospy.loginfo("Metric collection ended!")
        self.print_results()        # Output the collected metrics.

    # Logs the results of the metric collection including:
    #     - Total time elapsed
    #     - Time on trail
    #     - Percentage time on trail
    #     - Distance traversed
    #     - Final vehicle position
    def print_results(self):
        rospy.loginfo("Results:")
        rospy.loginfo(f"Total time: {self.total_time:.2f} seconds")
        rospy.loginfo(f"Time on trail: {self.trail_counter / self.sampling_rate:.2f} seconds")
        rospy.loginfo(f"Percent time on trail: {self.percent_time_on_trail:.2f}%")
        rospy.loginfo(f"Distance traversed: {self.total_distance_traversed:.2f} meters")

        # Retrieve and log the final vehicle position.
        x, y, z = self.pose_object.get_vehicle_position()
        rospy.loginfo(f"Final position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

# Main function to initialize the ROS node, set up the MetricCollection object,
# and manage the overall metric collection process.
def main():
    # Initialize the ROS node.
    rospy.init_node('metric_collection')
    
    # Define the start and end positions for metric collection.
    start_position = [-55, 21, 0]  # Replace with your desired start position.
    end_position = [56, -23, 0]    # Replace with your desired end position.
    
    # Create an instance of MetricCollection with the specified positions.
    mc = MetricCollection(start_position, end_position)
    
    # Wait until the first point cloud message and distance data are received.
    rate = rospy.Rate(10)  # 10 Hz loop rate.
    while (not mc.latest_cloud or mc.sim_total_distance_traversed is None) and not rospy.is_shutdown():
        if not mc.latest_cloud:
            rospy.loginfo("Waiting for point cloud data...")
        if mc.sim_total_distance_traversed is None:
            rospy.loginfo("Waiting for distance data...")
        rate.sleep()
    
    # Exit if ROS is shutdown before receiving necessary data.
    if rospy.is_shutdown():
        rospy.loginfo("ROS shutdown before receiving necessary data.")
        return

    rospy.loginfo("Received initial point cloud and distance data. Waiting for start position...")

    # Continuously check the vehicle's position to start and end metric collection.
    while not rospy.is_shutdown() and not mc.collection_ended:
        current_position = mc.pose_object.get_vehicle_position()
        rospy.loginfo(f"Current position: x={current_position[0]:.2f}, y={current_position[1]:.2f}, z={current_position[2]:.2f}")
        mc.check_position()  # Check if start or end positions have been reached.
        rate.sleep()

    rospy.loginfo("Metric collection complete. You can Ctrl+C to exit.")
    rospy.spin()  # Keep the node alive until shutdown.

if __name__ == "__main__":
    main()
