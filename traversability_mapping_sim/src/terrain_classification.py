#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

class GlobalMapMaintainer:
    def __init__(self):
        self.voxel_size = 0.25  # 25cm voxel size, adjust as needed
        self.global_map = None

        rospy.loginfo("Initializing GlobalMapMaintainer...")

        # Subscriber
        self.merged_cloud_sub = rospy.Subscriber('/segmented_terrain', PointCloud2, self.update_global_map)

        # Publisher
        self.global_map_pub = rospy.Publisher('/terrain_classification', PointCloud2, queue_size=10)

        rospy.loginfo("GlobalMapMaintainer initialized. Waiting for messages...")

    def update_global_map(self, cloud_msg):
        rospy.loginfo("Received merged pointcloud")
        
        try:
            new_cloud = self.pointcloud2_to_array(cloud_msg)

            if self.global_map is None:
                self.global_map = new_cloud
            else:
                self.global_map = np.vstack((self.global_map, new_cloud))

            rospy.loginfo(f"Updated global map. New shape: {self.global_map.shape}")

            # Downsample the global map
            downsampled_map = self.voxel_downsample(self.global_map)

            rospy.loginfo(f"Downsampled global map shape: {downsampled_map.shape}")

            # Publish the downsampled global map
            self.publish_global_map(downsampled_map, cloud_msg.header)

        except Exception as e:
            rospy.logerr(f"Error in update_global_map: {str(e)}")

    def voxel_downsample(self, pointcloud):
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3].astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:6].astype(np.float64))

            downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
            
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_colors = np.asarray(downsampled_pcd.colors)

            return np.hstack((downsampled_points, downsampled_colors)).astype(np.float32)
        except Exception as e:
            rospy.logerr(f"Error in voxel_downsample: {str(e)}")
            return pointcloud

    def publish_global_map(self, pointcloud, header):
        try:
            header.frame_id = 'map'
            fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
            
            pointcloud_msg = pc2.create_cloud(header, fields, pointcloud)
            self.global_map_pub.publish(pointcloud_msg)
            rospy.loginfo("Global map published successfully")
        except Exception as e:
            rospy.logerr(f"Error in publish_global_map: {str(e)}")

    @staticmethod
    def pointcloud2_to_array(cloud_msg):
        try:
            points_list = list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "r", "g", "b")))
            return np.array(points_list, dtype=np.float32)
        except Exception as e:
            rospy.logerr(f"Error in pointcloud2_to_array: {str(e)}")
            return np.array([])

if __name__ == '__main__':
    rospy.init_node('global_map_maintainer_node', anonymous=True)
    maintainer = GlobalMapMaintainer()
    rospy.spin()