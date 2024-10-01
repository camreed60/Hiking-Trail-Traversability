#!/usr/bin/env python3
import rospy
import numpy as np
from ultralytics import YOLO
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
import open3d as o3d

# Camera parameters
fx, fy = 205.47, 205.47  # Focal lengths
cx, cy = 320.5, 180.5  # Principal points

# Model weights
model_path = "/home/cam/ros_ws/src/traversability_mapping_sim/src/trail.pt"

class SegmentationPointCloud:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.pointcloud_pub = rospy.Publisher('/segmented_terrain', PointCloud2, queue_size=10)
        depth_sub = Subscriber('/rgbd_camera/depth/image', Image) # /rgbd_camera/depth/image
        rgb_sub = Subscriber('/rgbd_camera/color/image', Image) # /rgbd_camera/color/image
        ts = ApproximateTimeSynchronizer([depth_sub, rgb_sub], 10, 0.01)  # Reduced tolerance
        ts.registerCallback(self.synchronized_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.height_threshold = 0.75

        # Define colors for each class
        self.class_colors = {
            0: (0, 0, 255),    # red (255, 0, 0) for metrics # yellow : grass   (255. 255. 0) 
            1: (255, 255, 255),    # white (255, 255, 255) for metrics # Orange : trail  ( 255, 128, 0)
            2: (0, 255, 0)    # red for metrics# green : tree (0, 255, 0)
        }

    def create_colored_pointcloud(self, timestamp):
        pointcloud, depth_mask = self.rgbd_to_pointcloud(self.depth_image, fx, fy, cx, cy)

        binary_masks, class_labels, class_names = self.get_segmentation_masks_and_classes(self.rgb_image)

        # Initialize colored_pointcloud with the correct size
        colored_pointcloud = np.zeros((pointcloud.shape[0], 6))  # XYZ + RGB
        colored_pointcloud[:, :3] = pointcloud

        # Initialize a color array for all points
        colors = np.zeros((self.depth_image.shape[0], self.depth_image.shape[1], 3), dtype=np.float32)
        
        for mask, class_label in zip(binary_masks, class_labels):
            mask_bool = mask.reshape(self.depth_image.shape).astype(bool)
            color = self.class_colors.get(class_label, (128, 128, 128))
            color_float = np.array(color, dtype=np.float32) / 255.0
            colors[mask_bool] = color_float

        # Apply the colors to the pointcloud
        colored_pointcloud[:, 3:] = colors[depth_mask]

        # Remove points with no color assigned (black)
        colored_pointcloud = colored_pointcloud[np.any(colored_pointcloud[:, 3:] > 0, axis=1)]

        # Apply voxel grid filtering
        colored_pointcloud = self.voxel_grid_filter(colored_pointcloud)

        # Apply statistical outlier removal
        colored_pointcloud = self.statistical_outlier_removal(colored_pointcloud)

        # Transform pointcloud to account for robot movement
        try:
            transform = self.tf_buffer.lookup_transform('map', 'rgbd_camera', timestamp)
            colored_pointcloud[:, :3] = self.transform_pointcloud(colored_pointcloud[:, :3], transform)

            robot_position = self.get_robot_position()
            if robot_position is not None:
                height_threshold = self.height_threshold
                height_mask = colored_pointcloud[:, 2] <= (robot_position[2] + height_threshold)
                colored_pointcloud = colored_pointcloud[height_mask]
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to transform pointcloud: {e}")

        if colored_pointcloud.size == 0:
            rospy.logwarn("No points remaining after all filtering steps. Skipping this point cloud.")
            return  # Skip publishing if there are no points left

        header = std_msgs.msg.Header()
        header.stamp = timestamp
        header.frame_id = 'map'
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
        pointcloud_msg = pc2.create_cloud(header, fields, colored_pointcloud)
        self.pointcloud_pub.publish(pointcloud_msg)

    def synchronized_callback(self, depth_msg, rgb_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        self.create_colored_pointcloud(depth_msg.header.stamp)
        
    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'vehicle', rospy.Time(0))
            return np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get robot position: {e}")
            return None

    def get_segmentation_masks_and_classes(self, img):
        results = self.model.predict(img)
        class_names = self.model.names
        binary_masks, class_labels = [], []
        if results[0].masks is not None:
            masks = results[0].masks.data
            classes = results[0].boxes.cls
            masks_np = masks.cpu().numpy().astype(np.uint8)
            masks_np = np.array([cv2.resize(mask, (self.depth_image.shape[1], self.depth_image.shape[0])) for mask in masks_np])
            binary_masks = masks_np.reshape(masks_np.shape[0], -1)
            class_labels = classes.cpu().numpy().astype(int)
        return binary_masks, class_labels, class_names

    def rgbd_to_pointcloud(self, depth_image, fx, fy, cx, cy):
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth_image

        # Add depth filtering
        min_depth = 0.25 # 25 cm
        max_depth = 8  # 4 meters
        mask = (Z > min_depth) & (Z < max_depth)

        X = np.where(mask, (u - cx) * Z / fx, 0)
        Y = np.where(mask, (v - cy) * Z / fy, 0)
        
        valid_points = np.column_stack((X[mask], Y[mask], Z[mask]))
        return valid_points, mask

    def voxel_grid_filter(self, pointcloud, voxel_size=0.5):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:6])
        
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        return np.hstack((np.asarray(downsampled_pcd.points), np.asarray(downsampled_pcd.colors)))
    
    def statistical_outlier_removal(self, pointcloud, nb_neighbors=10, std_ratio=2.0): # 20, 2.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:6])
        
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        return np.hstack((np.asarray(cl.points), np.asarray(cl.colors)))

    def transform_pointcloud(self, points, transform):
        # Convert transform to 4x4 matrix
        t = transform.transform.translation
        r = transform.transform.rotation
        T = np.array([
            [1-2*(r.y**2+r.z**2), 2*(r.x*r.y-r.z*r.w), 2*(r.x*r.z+r.y*r.w), t.x],
            [2*(r.x*r.y+r.z*r.w), 1-2*(r.x**2+r.z**2), 2*(r.y*r.z-r.x*r.w), t.y],
            [2*(r.x*r.z-r.y*r.w), 2*(r.y*r.z+r.x*r.w), 1-2*(r.x**2+r.y**2), t.z],
            [0, 0, 0, 1]
        ])

        # Add homogeneous coordinate
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Transform points
        points_transformed = np.dot(T, points_h.T).T
        
        return points_transformed[:, :3]

if __name__ == '__main__':
    rospy.init_node('segmentation_pointcloud_node', anonymous=True)
    spc = SegmentationPointCloud()
    rospy.spin()