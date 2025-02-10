#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import threading

class PointCloudRepublisher:
    def __init__(self):
        rospy.init_node('pointcloud_republisher', anonymous=True)
        
        self.input_topic = rospy.get_param('~input_topic', '/trav_map')
        self.output_topic = rospy.get_param('~output_topic', '/trav_map_replay')
        self.publish_rate = rospy.get_param('~publish_rate', 10)  # Hz
        
        self.captured_cloud = None
        self.capture_lock = threading.Lock()
        
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.capture_callback)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        
        rospy.loginfo(f"Waiting to capture point cloud from {self.input_topic}...")

    def capture_callback(self, msg):
        with self.capture_lock:
            if self.captured_cloud is None:
                self.captured_cloud = msg
                rospy.loginfo("Point cloud captured! Starting to republish...")
                self.sub.unregister()  # Stop listening for new messages

    def republish(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            with self.capture_lock:
                if self.captured_cloud is not None:
                    self.captured_cloud.header.stamp = rospy.Time.now()
                    self.pub.publish(self.captured_cloud)
            rate.sleep()

if __name__ == '__main__':
    try:
        republisher = PointCloudRepublisher()
        republisher.republish()
    except rospy.ROSInterruptException:
        pass