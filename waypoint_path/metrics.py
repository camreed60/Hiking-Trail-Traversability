import rospy
import time
import threading
import numpy as np
from pose_listener import PoseListener
from std_msgs.msg import Float32, String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class MetricCollection:
    def __init__(self):
        self.start_time = None
        self.total_time = None
        self.pose_object = PoseListener()
        self.trail_counter = 0
        self.start_dist = None
        self.total_distance_traversed = None
        self.sim_total_distance_traversed = None
        self.stop_thread = False

        # Point cloud related attributes
        self.latest_cloud = None
        self.lock = threading.Lock()

        # Sampling rate (Hz)
        self.sampling_rate = 10

        # ROS subscribers
        rospy.Subscriber("/traveling_distance", Float32, self.travel_distance_callback)
        rospy.Subscriber("/terrain_truth", PointCloud2, self.point_cloud_callback)

        # Debug publisher
        self.debug_pub = rospy.Publisher('/trail_detection_debug', String, queue_size=10)

    def travel_distance_callback(self, msg):
        self.sim_total_distance_traversed = msg.data

    def point_cloud_callback(self, cloud_msg):
        with self.lock:
            self.latest_cloud = cloud_msg
        self.debug_point_cloud()

    def debug_point_cloud(self):
        if self.latest_cloud:
            point_count = 0
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            min_z, max_z = float('inf'), float('-inf')
            for point in pc2.read_points(self.latest_cloud, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True):
                point_count += 1
                x, y, z = point[:3]
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
            debug_msg = f"Point cloud stats: {point_count} points, X: [{min_x:.2f}, {max_x:.2f}], Y: [{min_y:.2f}, {max_y:.2f}], Z: [{min_z:.2f}, {max_z:.2f}]"
            #print(debug_msg)
            self.debug_pub.publish(String(debug_msg))
        else:
            print("No point cloud data available")

    def is_on_trail(self, position):
        with self.lock:
            if self.latest_cloud is None:
                print("No point cloud data available")
                self.debug_pub.publish(String("No point cloud data available"))
                return False
            
            x, y, z = position
            search_radius = 2  # meters
            on_trail_points = 0
            total_points = 0
            closest_point_distance = float('inf')
            closest_point_color = None

            #print(f"Searching for points near position: ({x:.2f}, {y:.2f}, {z:.2f})")

            for point in pc2.read_points(self.latest_cloud, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True):
                px, py, pz, r, g, b = point
                distance = ((px - x)**2 + (py - y)**2 + (pz - z)**2)**0.5
                
                if distance < closest_point_distance:
                    closest_point_distance = distance
                    closest_point_color = (r, g, b)
                
                if distance <= search_radius:
                    total_points += 1
                    if r > 0.5 and g > 0.5 and b > 0.5:
                        on_trail_points += 1

            if total_points == 0:
                debug_msg = f"No points found in search radius. Closest point: dist={closest_point_distance:.2f}, color={closest_point_color}"
                print(debug_msg)
                self.debug_pub.publish(String(debug_msg))
                return False

            trail_ratio = on_trail_points / total_points
            is_on_trail = trail_ratio > 0.1  # Lowered threshold to 30%

            debug_msg = (f"On trail: {is_on_trail}, ratio: {trail_ratio:.2f}, "
                         f"points: {on_trail_points}/{total_points}, "
                         f"closest point: dist={closest_point_distance:.2f}, color={closest_point_color}")
            #print(debug_msg)
            self.debug_pub.publish(String(debug_msg))
            return is_on_trail

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        self.total_time = time.time() - self.start_time

    def time_on_trail_controller(self):
        rate = rospy.Rate(self.sampling_rate)
        while not self.stop_thread:
            vehicleX, vehicleY, vehicleZ = self.pose_object.get_vehicle_position()
            #print(f"Vehicle position: ({vehicleX:.2f}, {vehicleY:.2f}, {vehicleZ:.2f})")
            if self.is_on_trail([vehicleX, vehicleY, vehicleZ]):
                self.trail_counter += 1
            rate.sleep()

    def start_time_on_trail(self):
        self.trail_thread = threading.Thread(target=self.time_on_trail_controller)
        self.trail_thread.daemon = True
        self.trail_thread.start()

    def end_time_on_trail(self):
        self.stop_thread = True
        if self.trail_thread.is_alive():
            self.trail_thread.join()
        self.percent_time_on_trail = (self.trail_counter / self.sampling_rate) / self.total_time * 100 if self.total_time else 0
    
    def start_distance_traversed(self):
        while self.sim_total_distance_traversed is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        self.start_dist = self.sim_total_distance_traversed
    
    def end_distance_traversed(self):
        self.total_distance_traversed = self.sim_total_distance_traversed - self.start_dist
        return self.total_distance_traversed

if __name__ == "__main__":
    rospy.init_node('metric_collection_test')
    
    mc = MetricCollection()
    
    # Wait for the first point cloud message and distance message
    rate = rospy.Rate(10)  # 10 Hz
    while (not mc.latest_cloud or mc.sim_total_distance_traversed is None) and not rospy.is_shutdown():
        if not mc.latest_cloud:
            rospy.loginfo("Waiting for point cloud data...")
        if mc.sim_total_distance_traversed is None:
            rospy.loginfo("Waiting for distance data...")
        rate.sleep()
    
    if rospy.is_shutdown():
        rospy.loginfo("ROS shutdown before receiving necessary data.")
        exit()
    
    rospy.loginfo("Received initial point cloud and distance data. Starting 10-second test...")
    
    mc.start_timer()
    mc.start_time_on_trail()
    mc.start_distance_traversed()
    
    # Run for 10 seconds with countdown
    test_duration = 10
    for i in range(test_duration, 0, -1):
        rospy.loginfo(f"Test running. {i} seconds remaining...")
        rospy.sleep(1)
    
    mc.end_timer()
    mc.end_time_on_trail()
    end_distance = mc.end_distance_traversed()
    
    rospy.loginfo("Test completed. Results:")
    rospy.loginfo(f"Total time: {mc.total_time:.2f} seconds")
    rospy.loginfo(f"Time on trail: {mc.trail_counter / mc.sampling_rate:.2f} seconds")
    rospy.loginfo(f"Percent time on trail: {mc.percent_time_on_trail:.2f}%")
    rospy.loginfo(f"Distance traversed: {end_distance:.2f} meters")

    x, y, z = mc.pose_object.get_vehicle_position()
    rospy.loginfo(f"Final position: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    
    
    rospy.loginfo("Test complete. You can Ctrl+C to exit.")
    rospy.spin()