#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import tf.transformations as tf_trans
from sklearn.neighbors import KDTree
from tf.transformations import quaternion_matrix

class PointCloudStitcher:
    def __init__(self):
        # Subscribe to semantic pointcloud, geometric pointcloud, and odometry
        self.semantic_pointcloud_sub = rospy.Subscriber('/segmented_pointcloud', PointCloud2, self.semantic_pointcloud_callback, queue_size=1)
        self.geometric_pointcloud_sub = rospy.Subscriber('/terrain_map', PointCloud2, self.geometric_pointcloud_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.odometry_callback, queue_size=1)
        # Publisher for the stitched traversability map
        self.stitched_pointcloud_pub = rospy.Publisher('/trav_map', PointCloud2, queue_size=1)
        
        self.global_pointcloud = None
        self.previous_pointcloud = None
        self.latest_semantic_pointcloud = None
        self.latest_geometric_pointcloud = None
        self.latest_odometry = None
        self.previous_pose = None
        
        self.voxel_size = 0.25
        self.fitness_threshold = 0.55
        self.timer = rospy.Timer(rospy.Duration(.1), self.timer_callback) # can try .1
        self.weight_semantic = 1
        self.weight_geometric = 1

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Color map for cost calculation 
        self.color_map = {
            (1.0, 1.0, 0.0): 0.4,    # yellow : grass    
            (1.0, 0.5, 0.0): 0.5,    # Orange : rock 
            (0.0, 1.0, 0.0): 0.0,    # green : trail   
            (0.0, 0.0, 1.0): 0.45,    # blue : roots 
            (1.0, 0.0, 1.0): 0.75,    # magenta: rough-trail
            (0.0, 1.0, 1.0): 0.9,    # cyan : structure 
            (0.59, 0.29, 0.0): 0.8,  # brown : tree-trunk (150/255, 75/255, 0)
            (0.5, 0.0, 1.0): 0.4,    # Purple : vegetation 
        }
        self.color_array = np.array(list(self.color_map.keys()))

    def semantic_pointcloud_callback(self, semantic_msg):
        self.latest_semantic_pointcloud = semantic_msg
        rospy.logdebug("Received semantic pointcloud")

    def geometric_pointcloud_callback(self, geometric_msg):
        self.latest_geometric_pointcloud = geometric_msg
        rospy.logdebug("Received geometric pointcloud")

    def odometry_callback(self, odom_msg):
        self.latest_odometry = odom_msg
        rospy.logdebug("Received odometry")

    def timer_callback(self, event):
        self.process_pointclouds()

    def process_pointclouds(self):
        if self.latest_semantic_pointcloud is None or self.latest_geometric_pointcloud is None or self.latest_odometry is None:
            rospy.logwarn_throttle(10, "Semantic PointCloud, Geometric PointCloud, or Odometry not received yet. Skipping this point cloud.")
            return

        # Convert pointclouds to numpy arrays
        semantic_np = self.pointcloud2_to_array(self.latest_semantic_pointcloud)
        geometric_np = self.pointcloud2_to_array(self.latest_geometric_pointcloud, fields=("x", "y", "z", "intensity"))

        # Transform geometric pointcloud to map frame
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                self.latest_geometric_pointcloud.header.frame_id,
                self.latest_geometric_pointcloud.header.stamp,
                rospy.Duration(1.0)
            )
            geometric_np[:, :3] = self.transform_points(geometric_np[:, :3], transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to transform geometric pointcloud: {e}")
            return
        
        # Build k-d trees for both pointclouds
        semantic_tree = KDTree(semantic_np[:, :3])
        geometric_tree = KDTree(geometric_np[:, :3])

        # Find nearest neighbors in both directions
        distances_sem_to_geo, indices_sem_to_geo = geometric_tree.query(semantic_np[:, :3], k=1)
        distances_geo_to_sem, indices_geo_to_sem = semantic_tree.query(geometric_np[:, :3], k=1)

        # Set a distance threshold for valid correspondences
        distance_threshold = 0.15 #10cm 

        # Create masks for valid correspondences in both directions
        valid_mask_sem = distances_sem_to_geo[:, 0] < distance_threshold
        # Find mutual nearest neighbors
        mutual_nearest_sem = np.arange(len(semantic_np))[valid_mask_sem]
        mutual_nearest_geo = indices_sem_to_geo[mutual_nearest_sem, 0]
        mutual_mask = indices_geo_to_sem[mutual_nearest_geo, 0] == mutual_nearest_sem

        # Final valid indices
        valid_sem_indices = mutual_nearest_sem[mutual_mask]
        valid_geo_indices = mutual_nearest_geo[mutual_mask]

        # Check if we have any valid points
        if len(valid_sem_indices) == 0 or len(valid_geo_indices) == 0:
            rospy.logwarn("No valid point correspondences found. Skipping this point cloud.")
            return

        # Extract valid points and colors
        valid_semantic_points = semantic_np[valid_sem_indices]
        valid_semantic_colors = semantic_np[valid_sem_indices, 3:6]
        valid_geometric_intensities = geometric_np[valid_geo_indices, 3]

        costs = self.calculate_cost(valid_semantic_colors, valid_geometric_intensities)

        # Create cost pointcloud (x, y, z, intensity)
        cost_pointcloud = np.column_stack((valid_semantic_points[:, :3], costs))

        # Convert cost pointcloud to Open3D format
        cost_pcd = self.numpy_to_o3d_pointcloud(cost_pointcloud)

        if self.previous_pointcloud is not None:
            previous_pointcloud_np = self.pointcloud2_to_array(self.previous_pointcloud)
            previous_pcd = self.numpy_to_o3d_pointcloud(previous_pointcloud_np)

            initial_guess = self.get_initial_guess()
            icp_transform = self.icp_registration(cost_pcd, previous_pcd, initial_guess)
            if icp_transform is not None:
                cost_pcd.transform(icp_transform)
                cost_pointcloud = np.asarray(cost_pcd.points)

        # Ensure cost_pointcloud has 4 columns (x, y, z, intensity)
        if cost_pointcloud.shape[1] == 3:
            cost_pointcloud = np.column_stack((cost_pointcloud, np.zeros(cost_pointcloud.shape[0])))

        # Add filtered points to global pointcloud
        if self.global_pointcloud is None:
            self.global_pointcloud = cost_pointcloud
        else:
            self.global_pointcloud = np.vstack((self.global_pointcloud, cost_pointcloud))

        # Downsample the global pointcloud
        self.global_pointcloud = self.voxel_downsample(self.global_pointcloud)

        # Publish the stitched pointcloud
        self.publish_stitched_pointcloud(self.latest_semantic_pointcloud.header)

        # Update previous pointcloud and pose for the next iteration
        self.previous_pointcloud = self.latest_semantic_pointcloud
        self.previous_pose = self.latest_odometry.pose.pose
        self.latest_semantic_pointcloud = None
        self.latest_geometric_pointcloud = None
    
    def transform_points(self, points, transform):
        translation = np.array([transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z])
        rotation = np.array([transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w])
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix(rotation)[:3, :3]  # Use the imported function
        
        # Apply rotation and translation
        transformed_points = np.dot(points, rotation_matrix.T) + translation
        
        return transformed_points
    
    def classify_color(self, color):
        color = tuple(color)  
        if color in self.color_map:
            return color, self.color_map[color]
        else:
            distances = np.sum((self.color_array - color)**2, axis=1)
            nearest_index = np.argmin(distances)
            nearest_color = tuple(self.color_array[nearest_index])
            return nearest_color, self.color_map[nearest_color]

    def calculate_cost(self, colors, intensities):
        costs = np.zeros(len(colors))
        for i, (color, intensity) in enumerate(zip(colors, intensities)):
            nearest_color, color_cost = self.classify_color(color)
            costs[i] = (color_cost) * (self.weight_semantic) + (intensity) * (self.weight_geometric)
        
        return costs

    def voxel_downsample(self, pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack((pointcloud[:, 3], pointcloud[:, 3], pointcloud[:, 3])))  # Use intensity as color
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        return np.hstack([np.asarray(downsampled_pcd.points), np.asarray(downsampled_pcd.colors)[:, 0:1]])  # Only keep one channel as intensity

    def publish_stitched_pointcloud(self, header):
        header.frame_id = 'map'  # Ensure the stitched pointcloud is in the map frame
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('intensity', 12, pc2.PointField.FLOAT32, 1)]
        pointcloud_msg = pc2.create_cloud(header, fields, self.global_pointcloud)
        self.stitched_pointcloud_pub.publish(pointcloud_msg)

    def get_initial_guess(self):
        if self.previous_pose is None or self.latest_odometry is None:
            return np.eye(4)
    
        current_pose = self.latest_odometry.pose.pose
        prev_pos = self.previous_pose.position
        curr_pos = current_pose.position
        
        # Calculate the change in position
        delta_x = curr_pos.x - prev_pos.x
        delta_y = curr_pos.y - prev_pos.y
        delta_z = curr_pos.z - prev_pos.z

        # Get the rotation quaternions
        prev_quat = [self.previous_pose.orientation.x, self.previous_pose.orientation.y, 
                    self.previous_pose.orientation.z, self.previous_pose.orientation.w]
        curr_quat = [current_pose.orientation.x, current_pose.orientation.y, 
                    current_pose.orientation.z, current_pose.orientation.w]

        # Calculate the change in rotation
        delta_rot = tf_trans.quaternion_multiply(curr_quat, tf_trans.quaternion_inverse(prev_quat))
        
        # Convert to transformation matrix
        delta_trans = tf_trans.translation_matrix([delta_x, delta_y, delta_z])
        delta_rot_mat = tf_trans.quaternion_matrix(delta_rot)
        
        # Combine translation and rotation
        initial_guess = np.dot(delta_trans, delta_rot_mat)

        return initial_guess

    def icp_registration(self, source_pcd, target_pcd, initial_guess=np.eye(4)):
        threshold = 0.1  # Distance threshold for ICP
        if initial_guess is None:
            initial_guess = np.eye(4)
        
        # Add more iterations and a stricter relative_fitness
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100, relative_fitness=1e-8)
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria
        )
        
        if reg_p2p.fitness > self.fitness_threshold:  # Adjust this threshold as needed
            return reg_p2p.transformation
        else:
            rospy.logwarn(f"ICP registration failed. Fitness: {reg_p2p.fitness}")
            return None

    def numpy_to_o3d_pointcloud(self, numpy_pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(numpy_pc[:, :3])
        if numpy_pc.shape[1] > 3:
            # Use intensity as grayscale color
            intensity = numpy_pc[:, 3]
            if len(intensity) > 0:
                # Normalize intensity to [0, 1] range
                intensity_min = np.min(intensity)
                intensity_max = np.max(intensity)
                if intensity_max > intensity_min:
                    intensity_normalized = (intensity - intensity_min) / (intensity_max - intensity_min)
                else:
                    intensity_normalized = np.zeros_like(intensity)
                # Create RGB array where R=G=B=intensity
                colors = np.column_stack((intensity_normalized, intensity_normalized, intensity_normalized))
                pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd    
    
    @staticmethod
    def pointcloud2_to_array(cloud_msg, fields=("x", "y", "z", "r", "g", "b")):
        return np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=fields)), dtype=np.float32)

if __name__ == '__main__':
    rospy.init_node('pointcloud_stitcher_node', anonymous=True)
    stitcher = PointCloudStitcher()
    rospy.spin()