#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

def pointcloud_callback(msg):
    # Convert ROS PointCloud2 message to numpy array, including intensity
    pc = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    points_with_intensity = np.array(list(pc))

    # Separate points and intensity
    points = points_with_intensity[:, :3]
    intensities = points_with_intensity[:, 3]

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add intensity as a scalar field
    pcd.colors = o3d.utility.Vector3dVector(np.vstack((intensities, intensities, intensities)).T)

    # Save as PLY file
    filename = f"newfigure6path{rospy.get_time()}.ply"
    o3d.io.write_point_cloud(filename, pcd)
    rospy.loginfo(f"Saved pointcloud with intensity to {filename}")

def main():
    rospy.init_node('pointcloud_saver', anonymous=True)
    rospy.Subscriber('/trav_map', PointCloud2, pointcloud_callback)
    rospy.spin()

if __name__ == '__main__':
    main()