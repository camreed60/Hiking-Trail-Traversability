#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import tf2_ros

class PointCloudStitcher:
    def __init__(self):
        self.pointcloud_sub = rospy.Subscriber('/terrain_map', PointCloud2, self.pointcloud_callback)
        self.stitched_pointcloud_pub = rospy.Publisher('/stitched_geometric_map', PointCloud2, queue_size=10)
        self.global_pointcloud = None
        self.voxel_size = 0.1  # 25cm voxel size, adjust as needed
        self.height_threshold = .1
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.min_intensity = float('inf')
        self.max_intensity = float('-inf')

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            return np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get robot position: {e}")
            return None

    def pointcloud_callback(self, pointcloud_msg):
        try:
            local_pointcloud = self.pointcloud2_to_array(pointcloud_msg)
            rospy.loginfo(f"Received pointcloud with shape: {local_pointcloud.shape}")

            if local_pointcloud.shape[1] != 4:  # x, y, z, intensity
                rospy.logerr(f"Expected 4 fields (x, y, z, intensity), but got {local_pointcloud.shape[1]}")
                return

            intensities = local_pointcloud[:, 3]
            rospy.loginfo(f"Intensity stats - min: {np.min(intensities)}, max: {np.max(intensities)}, mean: {np.mean(intensities)}")

            self.update_intensity_range(intensities)
            self.update_global_pointcloud(local_pointcloud)
            self.publish_stitched_pointcloud(pointcloud_msg.header)
        except Exception as e:
            rospy.logerr(f"Error in pointcloud_callback: {str(e)}")

    def update_intensity_range(self, intensities):
        self.min_intensity = min(self.min_intensity, np.min(intensities))
        self.max_intensity = max(self.max_intensity, np.max(intensities))
        rospy.loginfo(f"Intensity range: [{self.min_intensity}, {self.max_intensity}]")

    def update_global_pointcloud(self, new_pointcloud):
        try:
            robot_position = self.get_robot_position()
            if robot_position is not None:
                height_threshold = self.height_threshold
                height_mask = new_pointcloud[:, 2] <= (robot_position[2] + height_threshold)
                new_pointcloud = new_pointcloud[height_mask]

            if self.global_pointcloud is None:
                self.global_pointcloud = new_pointcloud
            else:
                # Combine with existing global pointcloud
                self.global_pointcloud = np.vstack((self.global_pointcloud, new_pointcloud))

            # Downsample the combined pointcloud
            self.global_pointcloud = self.voxel_downsample(self.global_pointcloud)
        except Exception as e:
            rospy.logerr(f"Error in update_global_pointcloud: {str(e)}")

    def voxel_downsample(self, pointcloud):
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            
            # Clip intensities to [0, 1] range
            intensities = np.clip(pointcloud[:, 3], 0, 1)
            
            # Convert to a 3-channel color (R=G=B=intensity)
            colors = np.column_stack([intensities] * 3)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

            # Extract downsampled points and intensities
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_intensities = np.asarray(downsampled_pcd.colors)[:, 0]  # Take first channel as intensity
            
            # Combine downsampled points and intensities
            return np.column_stack((downsampled_points, downsampled_intensities))
        except Exception as e:
            rospy.logerr(f"Error in voxel_downsample: {str(e)}")
            return pointcloud

    def publish_stitched_pointcloud(self, header):
        try:
            header.frame_id = 'map'  # Ensure the stitched pointcloud is in the map frame
            fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('intensity', 12, pc2.PointField.FLOAT32, 1)]

            pointcloud_msg = pc2.create_cloud(header, fields, self.global_pointcloud)
            self.stitched_pointcloud_pub.publish(pointcloud_msg)
        except Exception as e:
            rospy.logerr(f"Error in publish_stitched_pointcloud: {str(e)}")

    @staticmethod
    def pointcloud2_to_array(cloud_msg):
        try:
            points_list = list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")))
            return np.array(points_list, dtype=np.float32)
        except Exception as e:
            rospy.logerr(f"Error in pointcloud2_to_array: {str(e)}")
            return np.array([])

if __name__ == '__main__':
    rospy.init_node('pointcloud_stitcher_node', anonymous=True)
    stitcher = PointCloudStitcher()
    rospy.spin()