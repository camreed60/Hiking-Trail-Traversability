#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from scipy.spatial import cKDTree
import tf2_ros

class PointCloudStitcher:
    def __init__(self):
        # Subscribe to segmented pointcloud and LIO-SAM map
        self.pointcloud_sub = rospy.Subscriber('/segmented_pointcloud', PointCloud2, self.segmented_pointcloud_callback, queue_size=1)
        self.liosam_map_sub = rospy.Subscriber('/lio_sam/mapping/map_global', PointCloud2, self.liosam_map_callback, queue_size=1)
        # Publisher for the stitched traversability map
        self.stitched_pointcloud_pub = rospy.Publisher('/traversability_map', PointCloud2, queue_size=1)
        
        self.global_pointcloud = None
        self.liosam_map = None
        self.liosam_tree = None
        self.latest_segmented_pointcloud = None
        
        # Parameters for point cloud processing
        self.voxel_size = 0.25  # 25cm voxel size for downsampling
        self.distance_threshold = 0.5  # 50cm threshold for point matching
        self.height_threshold = 0.25  # 25cm height threshold
        
        # Timer for periodic processing
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)  # 10Hz

        # TF buffer for getting robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def timer_callback(self, event):
        self.process_pointclouds()
        
    def segmented_pointcloud_callback(self, segmented_msg):
        self.latest_segmented_pointcloud = segmented_msg
        rospy.logdebug("Received segmented pointcloud")

    def liosam_map_callback(self, map_msg):
        self.liosam_map = self.pointcloud2_to_array(map_msg)
        self.liosam_tree = cKDTree(self.liosam_map[:, :3])

    def get_robot_position(self):
        try:
            # Get the latest transform from map to base_link
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            return np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(10, f"Failed to get robot position: {e}")
            return None
        
    def process_pointclouds(self):
        if self.liosam_tree is None or self.latest_segmented_pointcloud is None:
            rospy.logwarn_throttle(10, "LIO-SAM map or segmented pointcloud not received yet. Skipping this point cloud.")
            return

        local_classes = self.pointcloud2_to_array(self.latest_segmented_pointcloud)
        self.update_global_pointcloud(local_classes)
        self.publish_stitched_pointcloud(self.latest_segmented_pointcloud.header)

        self.latest_segmented_pointcloud = None

    def update_global_pointcloud(self, new_pointcloud):
        # Filter points based on distance to LIO-SAM map
        distances, _ = self.liosam_tree.query(new_pointcloud[:, :3])
        mask = distances < self.distance_threshold
        filtered_pointcloud = new_pointcloud[mask]

        # Apply height threshold based on robot position
        robot_position = self.get_robot_position()
        if robot_position is not None:
            height_mask = filtered_pointcloud[:, 2] <= (robot_position[2] + self.height_threshold)
            filtered_pointcloud = filtered_pointcloud[height_mask]

        # Add filtered points to global pointcloud
        if self.global_pointcloud is None:
            self.global_pointcloud = filtered_pointcloud
        else:
            self.global_pointcloud = np.vstack((self.global_pointcloud, filtered_pointcloud))

        # Downsample the global pointcloud
        self.global_pointcloud = self.voxel_downsample(self.global_pointcloud)

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