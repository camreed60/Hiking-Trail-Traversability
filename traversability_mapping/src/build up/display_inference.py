#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge
import cv2
from ultralytics.utils.plotting import Annotator

# set latest image variable
latest_image = None

# load the YOLOv8 model
model = YOLO("/home/cam/ros_ws/src/traversability_mapping/src/2000imgs_weights.pt") #    change this to actual weights

# initialize the CvBridge
bridge = CvBridge()

def image_callback(msg):
    global latest_image

    # convert image message to numpy array
    latest_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def main():
    global latest_image
    rospy.init_node("image_collector", anonymous=True)
    
    # initializing as subscriber
    rospy.Subscriber("zed2i/zed_node/rgb_raw/image_raw_color", Image, image_callback)

    # initializing as publisher
    segmented_image_pub = rospy.Publisher("/segmented_image", Image, queue_size=10)

    # performing inference and streaming the annotated video 
    while not rospy.is_shutdown():
        if latest_image is not None:
            results = model(latest_image, stream=True)
            for result in results:
                annotated_frame = result.plot(boxes=False)
                
                # convert the annotated frame to ROS image message
                annotated_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")

                # publish the annotated image
                segmented_image_pub.publish(annotated_msg)

        rospy.sleep(0.1) # a small delay to avoid spinning too fast

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User exit")
            break


if __name__ == "__main__":
    main()