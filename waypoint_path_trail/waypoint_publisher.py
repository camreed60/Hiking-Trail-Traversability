#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl
import random
import math
import time
from scipy.spatial import cKDTree

class WaypointPublisher:
    def __init__(self):
        rospy.init_node('waypoint_publisher', anonymous=True)
        
        self.pointcloud = None
        self.pointcloud_array = None
        self.start = None
        self.goal = None
        
        self.scale = int(rospy.get_param('~scale', 5.0))
        
        self.goal_radius = 3.0 * self.scale
        self.min_step_size = 2.0 * self.scale
        self.step_size = 3.0 * self.scale
        self.search_radius = 3.0 * self.scale
        
        self.nodes = {}
        self.kdtree = None
        self.max_cost = 1.0  # Initialize max_cost, will be updated in pointcloud_callback
        
        rospy.Subscriber('/trav_map', PointCloud2, self.pointcloud_callback)
        self.waypoint_pub = rospy.Publisher('/waypoints', PointStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        
        # Interactive marker server
        self.server = InteractiveMarkerServer("waypoint_controls")
        self.menu_handler = MenuHandler()
        
        # Create interactive markers
        self.create_interactive_marker("start", (0, 0, 0), (1, 0, 0))  # Red for start
        self.create_interactive_marker("goal", (10, 10, 10), (0, 0, 1))  # Blue for goal
        
        self.server.applyChanges()

    def create_interactive_marker(self, name, position, color):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = name
        int_marker.description = f"{name.capitalize()} Point"
        int_marker.pose.position = Point(*position)

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.make_sphere(color))
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        self.server.insert(int_marker, self.marker_feedback)
        self.menu_handler.apply(self.server, int_marker.name)

    def make_sphere(self, color):
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        return marker

    def marker_feedback(self, feedback):
        if feedback.marker_name == "start":
            self.start = (feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z)
        elif feedback.marker_name == "goal":
            self.goal = (feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z)
        self.server.applyChanges()

    def pointcloud_callback(self, msg):
        try:
            self.pointcloud = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            if len(self.pointcloud) > 0:
                rospy.loginfo(f"Received pointcloud with {len(self.pointcloud)} points")
                self.pointcloud_array = np.array(self.pointcloud)
                self.max_cost = np.max(self.pointcloud_array[:, 3])
                rospy.loginfo(f"Maximum cost in pointcloud: {self.max_cost}")
            else:
                rospy.logwarn("Received empty pointcloud")
            
            if self.pointcloud_array.shape[1] < 4:
                rospy.logwarn(f"Pointcloud does not have enough columns. Shape: {self.pointcloud_array.shape}")
            else:
                self.kdtree = cKDTree(self.pointcloud_array[:, :3])
        except Exception as e:
            rospy.logerr(f"Error in pointcloud callback: {e}")

    def cost(self, node):
        return self.nodes[node]['cost'] if node in self.nodes else float('inf')

    def set_parent(self, node, parent):
        current_node = parent
        while current_node:
            if current_node == node:
                return False
            current_node = self.get_parent(current_node)
        if node not in self.nodes:
            self.nodes[node] = {'parent': parent, 'cost': float('inf')}
        self.nodes[node]['parent'] = parent
        return True

    def set_cost(self, node, cost_value):
        self.nodes[node]['cost'] = cost_value

    def get_parent(self, node):
        return self.nodes[node]['parent'] if node in self.nodes else None

    def distance(self, node1, node2):
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def reconstruct_path(self):
        path = []
        current_node = self.goal
        while current_node:
            path.append(current_node)
            current_node = self.get_parent(current_node)
        path.reverse()
        return path

    def sample_free(self):
        bounds = np.array(self.pointcloud)[:, :3].T
        return tuple(random.uniform(min(bound), max(bound)) for bound in bounds)

    def nearest(self, node):
        return min(self.nodes, key=lambda n: self.distance(n, node))

    def steer(self, from_node, to_node):
        distance = self.distance(from_node, to_node)
        if self.min_step_size < distance < self.step_size:
            return to_node
        direction = np.array(to_node) - np.array(from_node)
        if distance > self.min_step_size:
            return tuple(np.array(from_node) + self.step_size * direction / np.linalg.norm(direction))
        else:
            return tuple(np.array(from_node) + self.min_step_size * direction / np.linalg.norm(direction))

    def near_nodes(self, new_node):
        return [node for node in self.nodes if self.distance(node, new_node) <= self.search_radius]

    def get_traversability(self, node):
        if self.pointcloud_array is None or self.kdtree is None:
            rospy.logwarn("Pointcloud data not available")
            return 1.0  # Least traversable if data is not available
        
        try:
            _, idx = self.kdtree.query(node)
            if idx < len(self.pointcloud_array):
                cost = self.pointcloud_array[idx][3]
                # Invert and normalize the cost
                traversability = 1 - (cost / self.max_cost)
                return max(0, min(traversability, 1))  # Ensure the value is between 0 and 1
            else:
                rospy.logwarn(f"Invalid index {idx} for pointcloud of length {len(self.pointcloud_array)}")
                return 0.0  # Least traversable
        except Exception as e:
            rospy.logerr(f"Error in get_traversability: {e}")
            return 0.0  # Least traversable in case of error

    def calculate_cost(self, node, parent):
        distance_cost = self.distance(parent, node)
        node_traversability = self.get_traversability(node)
        parent_traversability = self.get_traversability(parent)
        if node_traversability == 1 and parent_traversability == 1:
            return self.cost(parent) + distance_cost
        else:
            try:
                traversability_cost = self.step_size / ((node_traversability + parent_traversability) / 2)
            except ZeroDivisionError:
                traversability_cost = float('inf')
            return self.cost(parent) + distance_cost + traversability_cost

    def plan_path(self):
        print("Generating a path...")
        distance = self.distance(self.start, self.goal)
        if distance < self.step_size:
            return [self.start, self.goal]

        time_limit = 60
        if distance < 120:
            time_limit = distance / 2
        if time_limit > 10:
            # Estimate overall traversability
            traversability_map = np.mean(1 - (self.pointcloud_array[:, 3] / self.max_cost))
            if traversability_map >= 0.75:
                time_limit = 10
        
        # The time limit is currently being overwritten to be 2 minutes
        time_limit = 120

        start_time = time.time()
        goal_found = False
        self.nodes = {self.start: {'parent': None, 'cost': 0}}

        while True:
            random_point = self.sample_free()
            nearest_node = self.nearest(random_point)
            new_node = self.steer(nearest_node, random_point)
            
            traversability = self.get_traversability(new_node)
            if traversability == 0:
                current_time = time.time() - start_time
                if current_time > time_limit:
                    break
                continue

            new_cost = self.calculate_cost(new_node, nearest_node)
            
            if new_node not in self.nodes or new_cost < self.cost(new_node):
                if self.set_parent(new_node, nearest_node):
                    self.set_cost(new_node, new_cost)
                    for near_node in self.near_nodes(new_node):
                        if near_node == nearest_node:
                            continue
                        
                        traversability_near = self.get_traversability(near_node)
                        if traversability_near == 0:
                            continue
                        
                        new_near_cost = self.calculate_cost(near_node, new_node)
                        
                        if new_near_cost < self.cost(near_node):
                            if self.set_parent(near_node, new_node):
                                self.set_cost(near_node, new_near_cost)
                    
                    if not goal_found:
                        if self.distance(new_node, self.goal) < self.goal_radius:
                            goal_traversability = self.get_traversability(self.goal)
                            if goal_traversability > 0:
                                self.nodes[self.goal] = {'parent': new_node, 'cost': new_cost + self.distance(new_node, self.goal) / goal_traversability}
                                goal_found = True
                                
                                for near_node in self.near_nodes(self.goal):
                                    if near_node == new_node:
                                        continue
                                    
                                    traversability_near = self.get_traversability(near_node)
                                    if traversability_near == 0:
                                        continue
                                    
                                    new_near_cost = self.calculate_cost(near_node, self.goal)
                                    if new_near_cost < self.cost(self.goal):
                                        if self.set_parent(self.goal, near_node):
                                            self.set_cost(self.goal, new_near_cost)
            
            current_time = time.time() - start_time
            if current_time > time_limit:
                break
        
        if not goal_found:
            nearest_node = self.nearest(self.goal)
            self.nodes[self.goal] = {'parent': nearest_node, 'cost': self.cost(nearest_node) + self.distance(nearest_node, self.goal)}
        
        for node in self.nodes:
            for near_node in self.near_nodes(node):
                if near_node == self.get_parent(node):
                    continue
                traversability_near = self.get_traversability(near_node)
                if traversability_near == 0:
                    continue
                new_near_cost = self.calculate_cost(near_node, node)
                if new_near_cost < self.cost(near_node):
                    self.set_parent(near_node, node)
                    self.set_cost(near_node, new_near_cost)
        
        end_time = time.time()
        planning_time = end_time - start_time
        path = self.reconstruct_path()
        
        print(f"Path planning completed:")
        print(f"  Time taken: {planning_time:.2f} seconds")
        print(f"  Nodes explored: {len(self.nodes)}")
        print(f"  Path length: {len(path)}")
        
        return path

    def visualize_path(self, path):
        marker_array = MarkerArray()
        
        for i, point in enumerate(path):
            # Create a sphere marker for each waypoint
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.3  # Diameter of sphere
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0  # Alpha (opacity)
            marker.color.r = 0.0
            marker.color.g = 1.0  # Green color
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position = Point(*point)
            marker.id = i  # Give each marker a unique id
            
            marker_array.markers.append(marker)
        
        # Add start point marker (red sphere)
        start_marker = Marker()
        start_marker.header.frame_id = "map"
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.scale.x = 0.5
        start_marker.scale.y = 0.5
        start_marker.scale.z = 0.5
        start_marker.color.a = 1.0
        start_marker.color.r = 1.0  # Red color
        start_marker.color.g = 0.0
        start_marker.color.b = 0.0
        start_marker.pose.orientation.w = 1.0
        start_marker.pose.position = Point(*self.start)
        start_marker.id = len(path)
        marker_array.markers.append(start_marker)
        
        # Add goal point marker (blue sphere)
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.scale.x = 0.5
        goal_marker.scale.y = 0.5
        goal_marker.scale.z = 0.5
        goal_marker.color.a = 1.0
        goal_marker.color.r = 0.0
        goal_marker.color.b = 1.0  # Blue color
        goal_marker.pose.orientation.w = 1.0
        goal_marker.pose.position = Point(*self.goal)
        goal_marker.id = len(path) + 1
        marker_array.markers.append(goal_marker)
        
        self.marker_pub.publish(marker_array)

    def plan_and_publish_path(self):
        if self.pointcloud_array is None or self.start is None or self.goal is None:
            rospy.logwarn("Waiting for pointcloud data and start/goal points...")
            return

        try:
            path = self.plan_path()
            self.visualize_path(path)

            for point in path:
                waypoint = PointStamped()
                waypoint.header.frame_id = "map"
                waypoint.header.stamp = rospy.Time.now()
                waypoint.point.x, waypoint.point.y, waypoint.point.z = point
                self.waypoint_pub.publish(waypoint)
                rospy.sleep(0.1)  # Small delay between waypoints
        except Exception as e:
            rospy.logerr(f"Error in plan_and_publish_path: {e}")

    def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            try:
                self.plan_and_publish_path()
            except Exception as e:
                rospy.logerr(f"Error in run method: {e}")
            rate.sleep()

if __name__ == '__main__':
    try:
        waypoint_publisher = WaypointPublisher()
        waypoint_publisher.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in main: {e}")