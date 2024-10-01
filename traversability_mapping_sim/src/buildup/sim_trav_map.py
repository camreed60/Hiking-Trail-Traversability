#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import message_filters
import open3d as o3d
from scipy.spatial import cKDTree

class PointCloudMerger:
    def __init__(self):
        self.voxel_size = 0.25  # 25cm voxel size, adjust as needed
        self.distance_threshold = 0.25  # 10cm threshold for considering points as "similar", adjust as needed

        rospy.loginfo("Initializing PointCloudMerger...")

        # Subscribers
        self.cloud1_sub = message_filters.Subscriber('/segmented_pointcloud', PointCloud2)
        self.cloud2_sub = message_filters.Subscriber('/stitched_terrain_map', PointCloud2)

        rospy.loginfo(f"Subscribed to topics: {self.cloud1_sub.name} and {self.cloud2_sub.name}")

        # Synchronize the subscriptions
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cloud1_sub, self.cloud2_sub], 10, 0.1)
        self.ts.registerCallback(self.pointcloud_callback)

        rospy.loginfo("Registered synchronized callback")

        # Publisher
        self.merged_pointcloud_pub = rospy.Publisher('/traversability_map', PointCloud2, queue_size=10)

        rospy.loginfo(f"Publishing merged pointcloud on topic: {self.merged_pointcloud_pub.name}")

        rospy.loginfo("PointCloudMerger initialized. Waiting for messages...")

    def pointcloud_callback(self, cloud1_msg, cloud2_msg):
        rospy.loginfo("Received synchronized pointcloud messages")
        
        try:
            # Convert both pointclouds to numpy arrays
            cloud1 = self.pointcloud2_to_array(cloud1_msg)
            cloud2 = self.pointcloud2_to_array(cloud2_msg)

            rospy.loginfo(f"Pointcloud 1 shape: {cloud1.shape}, Pointcloud 2 shape: {cloud2.shape}")

            # Check if either pointcloud is empty
            if cloud1.size == 0 or cloud2.size == 0:
                rospy.logwarn("One or both pointclouds are empty. Skipping merge.")
                return

            # Merge pointclouds
            merged_cloud = self.merge_pointclouds(cloud1, cloud2)

            rospy.loginfo(f"Merged pointcloud shape: {merged_cloud.shape}")

            # Downsample the merged pointcloud
            merged_cloud = self.voxel_downsample(merged_cloud)

            rospy.loginfo(f"Downsampled merged pointcloud shape: {merged_cloud.shape}")

            # Publish the merged pointcloud
            self.publish_merged_pointcloud(merged_cloud, cloud1_msg.header)

            rospy.loginfo("Published merged pointcloud")

        except Exception as e:
            rospy.logerr(f"Error in pointcloud_callback: {str(e)}")

    def merge_pointclouds(self, cloud1, cloud2):
        rospy.loginfo("Merging pointclouds...")

        # Extract red points from cloud2
        red_points = cloud2[np.all(cloud2[:, 3:6] == [1, 0, 0], axis=1)]
        
        if len(red_points) == 0:
            rospy.loginfo("No red points found in cloud2. Returning cloud1 unchanged.")
            return cloud1

        # Create KD-Tree for efficient nearest neighbor search
        tree = cKDTree(red_points[:, :3])

        # Find all points in cloud1 that are within the distance threshold of any red point
        distances, _ = tree.query(cloud1[:, :3], distance_upper_bound=self.distance_threshold)
        
        # Create a mask for points that are within the threshold
        mask = np.isfinite(distances)
        
        # Change the color of these points to red
        cloud1[mask, 3:6] = [1, 0, 0]

        rospy.loginfo(f"Changed {np.sum(mask)} points to red")
        return cloud1

    def voxel_downsample(self, pointcloud):
        rospy.loginfo("Downsampling merged pointcloud...")
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3].astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:].astype(np.float64))

            downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
            
            # Extract downsampled points and colors
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_colors = np.asarray(downsampled_pcd.colors)

            # Combine downsampled points and colors
            result = np.hstack((downsampled_points, downsampled_colors)).astype(np.float32)
            rospy.loginfo("Pointcloud downsampled successfully")
            return result
        except Exception as e:
            rospy.logerr(f"Error in voxel_downsample: {str(e)}")
            return pointcloud

    def publish_merged_pointcloud(self, pointcloud, header):
        rospy.loginfo("Publishing merged pointcloud...")
        try:
            header.frame_id = 'map'  # Ensure the merged pointcloud is in the map frame
            fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
            
            pointcloud_msg = pc2.create_cloud(header, fields, pointcloud)
            self.merged_pointcloud_pub.publish(pointcloud_msg)
            rospy.loginfo("Merged pointcloud published successfully")
        except Exception as e:
            rospy.logerr(f"Error in publish_merged_pointcloud: {str(e)}")

    @staticmethod
    def pointcloud2_to_array(cloud_msg):
        rospy.loginfo("Converting PointCloud2 message to numpy array...")
        try:
            points_list = list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "r", "g", "b")))
            result = np.array(points_list, dtype=np.float32)
            rospy.loginfo(f"Converted PointCloud2 to numpy array with shape: {result.shape}")
            return result
        except Exception as e:
            rospy.logerr(f"Error in pointcloud2_to_array: {str(e)}")
            return np.array([])
        

if __name__ == '__main__':
    rospy.init_node('pointcloud_merger_node', anonymous=True)
    merger = PointCloudMerger()
    rospy.loginfo("PointCloudMerger node is running...")
    rospy.spin()