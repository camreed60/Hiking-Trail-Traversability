#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from plyfile import PlyData

def publish_ply_as_pointcloud2():
    rospy.init_node('ply_publisher', anonymous=True)
    pub = rospy.Publisher('point_cloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # Load PLY file
    ply_data = PlyData.read('/home/cam/ros_ws/src/traversability_mapping_sim/fullsemantic_5568.18.ply')
    
    # Extract vertex data
    vertices = ply_data['vertex'].data

    # Convert vertex data to numpy array
    points = np.array([(v[0], v[1], v[2]) for v in vertices])

    # Check if there's a fourth element to use as intensity
    if len(vertices[0]) >= 4:
        intensity = np.array([v[3] for v in vertices])
    else:
        rospy.logwarn("PLY file does not have a fourth coordinate. Using default intensity.")
        intensity = np.ones(len(points)) * 255  # Default intensity if not available

    # Combine points and intensity
    cloud_points = np.column_stack((points, intensity))

    # Create PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"  # Set your desired frame_id

    # Define fields for PointCloud2
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
    ]

    # Create PointCloud2 message
    pc2_msg = pc2.create_cloud(header, fields, cloud_points)

    while not rospy.is_shutdown():
        pub.publish(pc2_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_ply_as_pointcloud2()
    except rospy.ROSInterruptException:
        pass