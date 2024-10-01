#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from scipy.spatial import cKDTree
import message_filters
import tf2_ros

class PointCloudStitcher:
    def __init__(self):
        self.pointcloud_sub = rospy.Subscriber('/segmented_pointcloud', PointCloud2, self.segmented_pointcloud_callback)
        self.terrain_map_sub = rospy.Subscriber('/terrain_map', PointCloud2, self.terrain_map_callback)
        self.liosam_map_sub = rospy.Subscriber('/lio_sam/mapping/map_global', PointCloud2, self.liosam_map_callback)
        self.stitched_pointcloud_pub = rospy.Publisher('/traversability_map', PointCloud2, queue_size=10)
        self.global_pointcloud = None
        self.liosam_map = None
        self.liosam_tree = None
        self.latest_terrain_map = None
        self.latest_segmented_pointcloud = None
        self.voxel_size = 0.25  # 25cm voxel size, adjust as needed
        self.distance_threshold = 0.5  # 10cm threshold for point matching
        self.height_threshold = 0.25 # 1 meter
        self.intensity_threshold = 0.1 # over .1 is impassable
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback) # 10Hz
        # Synchronize the subscriptions
       ## self.ts = message_filters.ApproximateTimeSynchronizer([self.pointcloud_sub, self.terrain_map_sub], 10, 0.1)
        #self.ts.registerCallback(self.pointcloud_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


    def timer_callback(self, event):
        self.process_pointclouds()
        
    def segmented_pointcloud_callback(self, segmented_msg):
        try:
            self.latest_segmented_pointcloud = segmented_msg
            rospy.logdebug("Received segmented pointcloud")
        except Exception as e:
            rospy.logerr(f"Error in segmented_pointcloud_callback: {str(e)}")

    def terrain_map_callback(self, terrain_map_msg):
        try:
            self.latest_terrain_map = terrain_map_msg
            rospy.logdebug("Received terrain map")
        except Exception as e:
            rospy.logerr(f"Error in terrain_map_callback: {str(e)}")

    def liosam_map_callback(self, map_msg):
        self.liosam_map = self.pointcloud2_to_array(map_msg)
        self.liosam_tree = cKDTree(self.liosam_map[:, :3])

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            return np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get robot position: {e}")
            return None
        
    def process_pointclouds(self):
        if self.liosam_tree is None or self.latest_terrain_map is None or self.latest_segmented_pointcloud is None:
            rospy.logwarn("LIO-SAM map not received yet. Skipping this point cloud.")
            return

        
        local_classes= self.pointcloud2_to_array(self.latest_segmented_pointcloud)
        local_terrain = self.intensity_pointcloud2_to_array(self.latest_terrain_map)
        rgb_terrain = self.intensity_to_rgb(local_terrain)
        merged_cloud = self.merge_pointclouds(local_classes, rgb_terrain)
        self.update_global_pointcloud(merged_cloud)
        self.publish_stitched_pointcloud(self.latest_segmented_pointcloud.header)

        # Clear the processed data
        self.latest_segmented_pointcloud = None
        self.latest_terrain_map = None
    def intensity_to_rgb(self, pointcloud):
        xyz = pointcloud[:, :3]
        intensity = pointcloud[:, 3]
        
        rgb = np.zeros((len(intensity), 3), dtype=np.float32)
        rgb[intensity > self.intensity_threshold] = [1, 0, 0]  # Red for high intensity
        rgb[intensity <= self.intensity_threshold] = [0, 1, 0]  # Green for low intensity
        
        return np.hstack((xyz, rgb))
    
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
    
    def update_global_pointcloud(self, new_pointcloud):
        # Filter new_pointcloud based on LIO-SAM map
        distances, _ = self.liosam_tree.query(new_pointcloud[:, :3])
        filtered_pointcloud = new_pointcloud[distances < self.distance_threshold]
        robot_position = self.get_robot_position()
        if robot_position is not None:
                height_threshold = self.height_threshold
                height_mask = filtered_pointcloud[:, 2] <= (robot_position[2] + height_threshold)
                filtered_pointcloud = filtered_pointcloud[height_mask]

        if self.global_pointcloud is None:
            self.global_pointcloud = filtered_pointcloud
        else:
            # Combine with existing global pointcloud
            self.global_pointcloud = np.vstack((self.global_pointcloud, filtered_pointcloud))

        # Downsample the combined pointcloud
        self.global_pointcloud = self.voxel_downsample(self.global_pointcloud)

    def voxel_downsample(self, pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:])
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        return np.hstack([np.asarray(downsampled_pcd.points), np.asarray(downsampled_pcd.colors)])

    def publish_stitched_pointcloud(self, header):
        header.frame_id = 'map'  # Ensure the stitched pointcloud is in the map frame
        #downsampled_global = self.voxel_downsample(self.global_pointcloud)
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
        points_list = []
        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "r", "g", "b")):
            points_list.append(point)
        return np.array(points_list, dtype=np.float32)
    @staticmethod
    def intensity_pointcloud2_to_array(cloud_msg):
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