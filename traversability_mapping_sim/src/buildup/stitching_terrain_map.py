#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

class PointCloudStitcher:
    def __init__(self):
        self.pointcloud_sub = rospy.Subscriber('/segmented_pointcloud', PointCloud2, self.pointcloud_callback)
        self.stitched_pointcloud_pub = rospy.Publisher('/terrain_classification', PointCloud2, queue_size=10)
        self.global_pointcloud = None
        self.voxel_size = 0.25  # 25cm voxel size, adjust as needed

    def pointcloud_callback(self, pointcloud_msg):
        try:
            local_pointcloud = self.pointcloud2_to_array(pointcloud_msg)
            rospy.loginfo(f"Received pointcloud with shape: {local_pointcloud.shape}")
            self.update_global_pointcloud(local_pointcloud)
            self.publish_stitched_pointcloud(pointcloud_msg.header)
        except Exception as e:
            rospy.logerr(f"Error in pointcloud_callback: {str(e)}")


    def update_global_pointcloud(self, new_pointcloud):
        try:
            if self.global_pointcloud is None:
                self.global_pointcloud = new_pointcloud
            else:
                # Combine with existing global pointcloud
                self.global_pointcloud = np.vstack((self.global_pointcloud, new_pointcloud))
        
        except Exception as e:
            rospy.logerr(f"Error in update_global_pointcloud: {str(e)}")

    def publish_stitched_pointcloud(self, header):
        try:
            header.frame_id = 'map'  # Ensure the stitched pointcloud is in the map frame
            fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                      pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
            
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