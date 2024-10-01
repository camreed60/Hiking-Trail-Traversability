#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import tf.transformations as tf_trans

class PointCloudStitcher:
    def __init__(self):
        # Subscribe to segmented pointcloud and odometry
        self.pointcloud_sub = rospy.Subscriber('/segmented_pointcloud', PointCloud2, self.segmented_pointcloud_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.odometry_callback, queue_size=1)
        # Publisher for the stitched traversability map
        self.stitched_pointcloud_pub = rospy.Publisher('/terrain_classification', PointCloud2, queue_size=1)
        
        self.global_pointcloud = None
        self.previous_pointcloud = None
        self.latest_segmented_pointcloud = None
        self.latest_odometry = None
        self.previous_pose = None  # Store the pose when the previous point cloud was captured
        
        # Parameters for point cloud processing
        self.voxel_size = 0.25  # 25cm voxel size for downsampling
        self.fitness_threshold = 0.55
        # Timer for periodic processing
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)  # 10Hz

    def timer_callback(self, event):
        self.process_pointclouds()
        
    def segmented_pointcloud_callback(self, segmented_msg):
        self.latest_segmented_pointcloud = segmented_msg
        rospy.logdebug("Received segmented pointcloud")

    def odometry_callback(self, odom_msg):
        self.latest_odometry = odom_msg
        rospy.logdebug("Received odometry")

    def process_pointclouds(self):
        if self.latest_segmented_pointcloud is None or self.latest_odometry is None:
            rospy.logwarn_throttle(10, "PointCloud or Odometry not received yet. Skipping this point cloud.")
            return

        # Convert the latest segmented pointcloud to a numpy array
        new_pointcloud_np = self.pointcloud2_to_array(self.latest_segmented_pointcloud)
        new_pcd = self.numpy_to_o3d_pointcloud(new_pointcloud_np)

        if self.previous_pointcloud is not None:
            previous_pointcloud_np = self.pointcloud2_to_array(self.previous_pointcloud)
            previous_pcd = self.numpy_to_o3d_pointcloud(previous_pointcloud_np)

            # Get the initial guess transformation between the two point clouds
            initial_guess = self.get_initial_guess()
            
            # Perform ICP registration using the initial guess
            icp_transform = self.icp_registration(new_pcd, previous_pcd, initial_guess)
            if icp_transform is not None:
                # Apply the ICP transformation to the new point cloud
                new_pcd.transform(icp_transform)
                # Extract the transformed points and keep the original colors
                new_pointcloud_np = np.hstack((np.asarray(new_pcd.points), new_pointcloud_np[:, 3:]))

        # Add filtered points to global pointcloud
        if self.global_pointcloud is None:
            self.global_pointcloud = new_pointcloud_np
        else:
            self.global_pointcloud = np.vstack((self.global_pointcloud, new_pointcloud_np))

        # Downsample the global pointcloud
        self.global_pointcloud = self.voxel_downsample(self.global_pointcloud)

        # Publish the stitched pointcloud
        self.publish_stitched_pointcloud(self.latest_segmented_pointcloud.header)

        # Update previous pointcloud and pose for the next iteration
        self.previous_pointcloud = self.latest_segmented_pointcloud
        self.previous_pose = self.latest_odometry.pose.pose
        self.latest_segmented_pointcloud = None

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
    
    def voxel_downsample(self, pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:])
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        return np.hstack([np.asarray(downsampled_pcd.points), np.asarray(downsampled_pcd.colors)])

    def publish_stitched_pointcloud(self, header):
        header.frame_id = 'map'  # Ensure the stitched pointcloud is in the map frame
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
        pointcloud_msg = pc2.create_cloud(header, fields, self.global_pointcloud)
        self.stitched_pointcloud_pub.publish(pointcloud_msg)

    def icp_registration(self, source_pcd, target_pcd, initial_guess=np.eye(4)):
        threshold = 0.05  # Distance threshold for ICP
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
            pcd.colors = o3d.utility.Vector3dVector(numpy_pc[:, 3:])
        return pcd

    @staticmethod
    def pointcloud2_to_array(cloud_msg):
        # Convert PointCloud2 message to numpy array
        return np.array(list(pc2.read_points(cloud_msg, skip_nans=True, 
                                             field_names=("x", "y", "z", "r", "g", "b"))), 
                        dtype=np.float32)

if __name__ == '__main__':
    rospy.init_node('pointcloud_stitcher_node', anonymous=True)
    stitcher = PointCloudStitcher()
    rospy.spin()
