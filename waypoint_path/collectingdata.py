#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import tf

class GlobalPlannerWaypoints:
    def __init__(self):
        rospy.init_node('global_planner_waypoints')
        
        self.tf_listener = tf.TransformListener()
        
        # Wait for the make_plan service to become available
        rospy.wait_for_service('/move_base/make_plan')
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        
        self.waypoint_pub = rospy.Publisher('/way_point', PointStamped, queue_size=10)
        self.odom_sub = rospy.Subscriber('/state_estimation', Odometry, self.odom_callback)
        self.trav_cloud_sub = rospy.Subscriber('/trav_map_replay', PointCloud2, self.trav_cloud_callback)
        self.terrain_cloud_sub = rospy.Subscriber('/terrain_truth', PointCloud2, self.terrain_cloud_callback)
        
        self.robot_position = None
        self.trav_cloud = None
        self.terrain_cloud = None
        
        # Metrics
        self.start_time = None
        self.total_time = None
        self.trail_counter = 0
        self.start_dist = None
        self.total_distance_traversed = None
        self.sim_total_distance_traversed = None
        
        rospy.Subscriber("/traveling_distance", Float32, self.travel_distance_callback)
        
        rospy.loginfo("GlobalPlannerWaypoints initialized")

    def odom_callback(self, msg):
        self.robot_position = msg.pose.pose

    def trav_cloud_callback(self, data):
        self.trav_cloud = data

    def terrain_cloud_callback(self, data):
        self.terrain_cloud = data

    def travel_distance_callback(self, msg):
        self.sim_total_distance_traversed = msg.data

    def get_plan(self, start, goal):
        start_pose = PoseStamped()
        start_pose.header.frame_id = "map"
        start_pose.pose.position.x = start[0]
        start_pose.pose.position.y = start[1]
        start_pose.pose.orientation.w = 1.0

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = goal[0]
        goal_pose.pose.position.y = goal[1]
        goal_pose.pose.orientation.w = 1.0

        tolerance = 0.5  # 0.5 meter tolerance for the goal

        plan = self.make_plan(start_pose, goal_pose, tolerance)
        return plan.plan

    def extract_waypoints(self, path, max_waypoints=10):
        if not path.poses:
            return []

        waypoints = [path.poses[0].pose.position]
        path_length = len(path.poses)
        step = max(1, path_length // (max_waypoints - 1))

        for i in range(step, path_length, step):
            waypoints.append(path.poses[i].pose.position)

        if waypoints[-1] != path.poses[-1].pose.position:
            waypoints.append(path.poses[-1].pose.position)

        return waypoints

    def publish_waypoints(self, waypoints):
        for i, point in enumerate(waypoints):
            point_msg = PointStamped()
            point_msg.header.frame_id = "map"
            point_msg.header.stamp = rospy.Time.now()
            point_msg.point.x = point.x
            point_msg.point.y = point.y
            point_msg.point.z = 0  # Assuming 2D navigation

            self.waypoint_pub.publish(point_msg)
            rospy.loginfo(f"Published waypoint {i+1}/{len(waypoints)}: ({point.x}, {point.y})")
            rospy.sleep(0.1)  # Small delay between waypoints

    def start_timer(self):
        self.start_time = rospy.Time.now()

    def end_timer(self):
        self.total_time = (rospy.Time.now() - self.start_time).to_sec()

    def start_distance_traversed(self):
        self.start_dist = self.sim_total_distance_traversed

    def end_distance_traversed(self):
        self.total_distance_traversed = self.sim_total_distance_traversed - self.start_dist
        return self.total_distance_traversed

    def run_navigation(self, start, goal, max_waypoints=10):
        rospy.loginfo(f"Planning path from {start} to {goal}")
        
        self.start_timer()
        self.start_distance_traversed()
        
        path = self.get_plan(start, goal)
        if not path.poses:
            rospy.logwarn("No path found. Skipping this run.")
            return None
        
        waypoints = self.extract_waypoints(path, max_waypoints)
        self.publish_waypoints(waypoints)
        
        self.end_timer()
        distance = self.end_distance_traversed()
        
        metrics = {
            "total_time": self.total_time,
            "distance_traversed": distance,
            "num_waypoints": len(waypoints)
        }
        rospy.loginfo(f"Navigation run completed. Metrics: {metrics}")
        return metrics

if __name__ == '__main__':
    try:
        planner = GlobalPlannerWaypoints()
        
        rospy.loginfo("Waiting for initial messages...")
        rospy.wait_for_message('/trav_map_replay', PointCloud2, timeout=10)
        rospy.wait_for_message('/terrain_truth', PointCloud2, timeout=10)
        rospy.wait_for_message('/traveling_distance', Float32, timeout=10)
        rospy.wait_for_message('/odometry/filtered', Odometry, timeout=10)
        rospy.loginfo("Received initial messages from all required topics")
        
        start = [26, 10]
        goal = [8, -11]
        max_waypoints = 25
        
        all_metrics = []
        
        for i in range(5):  # 5 round trips
            rospy.loginfo(f"Starting round trip {i+1}")
            
            # Start to Goal
            rospy.loginfo(f"  Run {2*i+1}: Start to Goal")
            metrics = planner.run_navigation(start, goal, max_waypoints)
            if metrics:
                all_metrics.append(metrics)
            
            rospy.sleep(5)  # Wait for 5 seconds
            
            # Goal to Start
            rospy.loginfo(f"  Run {2*i+2}: Goal to Start")
            metrics = planner.run_navigation(goal, start, max_waypoints)
            if metrics:
                all_metrics.append(metrics)
            
            rospy.sleep(5)  # Wait for 5 seconds
        
        rospy.loginfo("All runs completed. Final metrics:")
        for i, metrics in enumerate(all_metrics):
            rospy.loginfo(f"Run {i+1}:")
            rospy.loginfo(f"  Total time: {metrics['total_time']:.2f} seconds")
            rospy.loginfo(f"  Distance traversed: {metrics['distance_traversed']:.2f} meters")
            rospy.loginfo(f"  Number of waypoints: {metrics['num_waypoints']}")
        
        # Calculate and display average metrics
        avg_time = sum(m['total_time'] for m in all_metrics) / len(all_metrics)
        avg_distance = sum(m['distance_traversed'] for m in all_metrics) / len(all_metrics)
        avg_waypoints = sum(m['num_waypoints'] for m in all_metrics) / len(all_metrics)
        
        rospy.loginfo("Average metrics across all runs:")
        rospy.loginfo(f"  Avg. Total time: {avg_time:.2f} seconds")
        rospy.loginfo(f"  Avg. Distance traversed: {avg_distance:.2f} meters")
        rospy.loginfo(f"  Avg. Number of waypoints: {avg_waypoints:.2f}")
        
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except rospy.ROSException as e:
        rospy.logerr(f"ROS exception occurred: {str(e)}")