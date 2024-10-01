#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from scipy.spatial import cKDTree

class PointCloudCostMerger:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('point_cloud_cost_merger', anonymous=True)
        
        # Initialize point cloud storage
        self.semantic_cloud = None
        self.geometric_cloud = None
        
        # Set up ROS subscribers
        self.semantic_sub = rospy.Subscriber("/segmented_pointcloud", PointCloud2, self.semantic_callback)
        self.geometric_sub = rospy.Subscriber("/terrain_map", PointCloud2, self.geometric_callback)
        
        # Set up ROS publisher for the merged cost cloud
        self.cost_cloud_pub = rospy.Publisher("/trav_map", PointCloud2, queue_size=1)
        
        # Set the rate for the main loop
        self.rate = rospy.Rate(10)  # 10 Hz
        
        # Set the cost weight
        self.weight = 0.5

        # Define color ranges for cost values
        self.color_ranges = [
            (0.0, 0.1, (0, 255, 0)),     # Bright Green (lowest cost)
            (0.1, 0.2, (128, 255, 0)),   # Light Green
            (0.2, 0.3, (255, 255, 0)),   # Yellow
            (0.3, 0.4, (255, 192, 0)),   # Light Orange
            (0.4, 0.5, (255, 128, 0)),   # Orange
            (0.5, 0.6, (255, 64, 0)),    # Dark Orange
            (0.6, 0.7, (255, 0, 0)),     # Red
            (0.7, 0.8, (255, 0, 128)),   # Pink
            (0.8, 0.9, (255, 0, 255)),   # Magenta
            (0.9, 1.0, (128, 0, 255))    # Purple (highest cost)
        ]

    def semantic_callback(self, msg):
        # Store the received semantic point cloud
        self.semantic_cloud = msg

    def geometric_callback(self, msg):
        # Store the received geometric point cloud
        self.geometric_cloud = msg

    def cost_function(self, semantic_rgb, geometric_intensity):
        # Calculate cost based on semantic RGB and geometric intensity
        # Use the semantic_weight to balance between semantic and geometric information
        r, g, b = semantic_rgb
        normalized_rgb = (r + g + b) / (3 * 255)  # Normalize RGB to 0-1 range
        semantic_cost = normalized_rgb
        geometric_cost = max(0, min(1, geometric_intensity))  # Clamp to [0, 1]
        
        # Weighted sum of semantic and geometric costs
        cost = self.weight * semantic_cost + (1 - self.weight) * geometric_cost
        
        # Ensure the final cost is between 0 and 1
        return max(0, min(1, cost)) # Combine semantic and geometric info

    def color_from_cost(self, cost):
        # Map cost to a color using predefined ranges
        for low, high, color in self.color_ranges:
            if low <= cost < high:
                return color
        return self.color_ranges[-1][2]  # Return the highest cost color if out of range

    def process_clouds(self):
        # Check if both point clouds have been received
        if self.semantic_cloud is None or self.geometric_cloud is None:
            rospy.logwarn("Waiting for both semantic and geometric point clouds...")
            return

        try:
            # Extract points from both clouds
            semantic_points = list(pc2.read_points(self.semantic_cloud, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True))
            geometric_points = list(pc2.read_points(self.geometric_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True))

            if not semantic_points or not geometric_points:
                rospy.logwarn("One or both point clouds are empty.")
                return

            semantic_xyz = np.array([[p[0], p[1], p[2]] for p in semantic_points])
            geometric_xyz = np.array([[p[0], p[1], p[2]] for p in geometric_points])

            # Build KDTree for efficient nearest neighbor search
            tree = cKDTree(geometric_xyz)

            cost_points = []

            # Find nearest neighbors for all semantic points at once
            distances, indices = tree.query(semantic_xyz, k=1)

            for i, (sem_point, distance, index) in enumerate(zip(semantic_points, distances, indices)):
                if distance > 0.1:  # 10cm threshold, adjust as needed
                    continue  # Skip if no close geometric point found

                x, y, z, r, g, b = sem_point
                geo_intensity = geometric_points[index][3]

                # Calculate cost
                cost = self.cost_function((r, g, b), geo_intensity)
                print(cost)

                # Determine color based on cost
                color = self.color_from_cost(cost)

                cost_points.append([x, y, z, color[0], color[1], color[2]])

            if not cost_points:
                rospy.logwarn("No matching points found within the distance threshold.")
                return

            # Create and publish the cost point cloud
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.semantic_cloud.header.frame_id

            fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
            cost_cloud = pc2.create_cloud(header, fields, cost_points)

            self.cost_cloud_pub.publish(cost_cloud)
            rospy.loginfo(f"Published cost cloud with {len(cost_points)} points.")

        except Exception as e:
            rospy.logerr(f"Error in process_clouds: {str(e)}")

    def run(self):
        while not rospy.is_shutdown():
            self.process_clouds()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        merger = PointCloudCostMerger()
        merger.run()
    except rospy.ROSInterruptException:
        pass