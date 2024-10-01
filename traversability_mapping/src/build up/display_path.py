#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

def publish_line_strip():
    rospy.init_node('line_strip_publisher', anonymous=True)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    
    # Create a Marker message
    marker = Marker()
    marker.header.frame_id = "map"  # Use the correct frame ID
    marker.header.stamp = rospy.Time.now()
    marker.ns = "sphere"
    marker.id = 0
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD

    # Set the scale (sphere diameter)
    marker.scale.x = 1.2  # Sphere diameter in x direction
    marker.scale.y = 1.2  # Sphere diameter in y direction
    marker.scale.z = 1.2  # Sphere diameter in z direction

    # Set the color (RGBA)
    marker.color.r = 0.3
    marker.color.g = 0.0
    marker.color.b = 0.5
    marker.color.a = 1.0

    # Define the points (replace these with your actual point coordinates)
    points = [
        [0, 0, 0],
        [1, 1, 0],
        [2, 0, 0],
        [3, 1, 0],
    ]
    points = np.loadtxt('waypoint3.txt')

    print("Array:")
    print(points)


    # Create a Marker message for the line
    line_marker = Marker()
    line_marker.header.frame_id = "map"
    line_marker.header.stamp = rospy.Time.now()
    line_marker.ns = "line_strip"
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD

    # Set the line scale (line width)
    line_marker.scale.x = 0.05

    # Set the line color (red)
    line_marker.color.r = 0.0
    line_marker.color.g = 0.0
    line_marker.color.b = 0.0
    line_marker.color.a = 1.0

    # Add the points to the marker
    for p in points:
        pt = Point()
        pt.x, pt.y, pt.z = p
        marker.points.append(pt)
        line_marker.points.append(pt)




    # Publish the marker at 1Hz
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()  # Update timestamp
        marker_pub.publish(marker)
        marker_pub.publish(line_marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_line_strip()
    except rospy.ROSInterruptException:
        pass