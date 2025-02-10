#!/usr/bin/env python3

import rospy
import time
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Constructs a 2D Traversability Map and then 
# returns it to the waypoint path node for use with 
# the advanced RRT Star Path Planner.

class TraversabilityListener:
    def __init__(self, scale_value):
        self.points_list = []
        self.scale_value = int(scale_value)
        self.terrain_points_list = []
        self.traversability_values = None
        self.travs_x = None
        self.travs_y = None

        # Subscribe to the /traversability_map topic
        rospy.Subscriber("/trav_map_replay", PointCloud2, self.callback_classes)
    
    # Pointcloud callback
    def callback_classes(self, point_cloud):
        new_points = list(pc2.read_points(point_cloud, skip_nans=True))
        self.points_list.extend(new_points)

    # Get the traversability value of a point
    def extract_trav_value(self, point):
        return point[3]
    
    # Generate an empty map in the case where a traversability map cannot be constructed
    def generate_empty_map(self, width, height):
        return np.random.uniform(low=1, high=1, size=(width,height))
    
    def controller(self):
        map_q1, map_q2, map_q3, map_q4 = self.build_traversability_map()
        self.optimize_quadrants(map_q1, map_q2, map_q3, map_q4)
        return map_q1, map_q2, map_q3, map_q4

    # Optimize the quadrants to build the traversability maps for the planner
    def optimize_quadrants(self, map_q1, map_q2, map_q3, map_q4):
        # Create a KDTree for the points
        points_array = np.array(self.points_list)
        kdtree = KDTree(points_array[:, :2])

        max_trav = min(point[3] for point in self.points_list)
        min_trav = max(point[3] for point in self.points_list)

        # Process each quadrant
    def process_quadrant(self, map_q, x_multiplier, y_multiplier):
        height, width = map_q.shape
        for x in range(width):
            for y in range(height):
                x_min = (x * x_multiplier) / self.scale_value
                x_max = ((x + 1) * x_multiplier) / self.scale_value
                y_min = (y * y_multiplier) / self.scale_value
                y_max = ((y + 1) * y_multiplier) / self.scale_value

                box_min = np.array([min(x_min, x_max), min(y_min, y_max)])
                box_max = np.array([max(x_min, x_max), max(y_min, y_max)])

                indices = self.kdtree.query_ball_point((box_min + box_max) / 2, np.linalg.norm(box_max - box_min) / 2)
                
                if indices:
                    relevant_points = self.points_array[indices]
                    if len(relevant_points) > 0:
                        traversability_values = [self.extract_trav_value(point) for point in relevant_points]
                       # print(traversability_values)
                        avg_trav = np.mean(traversability_values)
                        map_q[y, x] = avg_trav
                else:
                    # If no points found, assign the highest traversability value
                    map_q[y, x] = self.max_trav

        return map_q

        # Start timer to construct traversability map
        start_time = time.time()
        try:
            print("Scaling traversability map for the path planner...")
            process_quadrant(map_q1, 1, 1)
            process_quadrant(map_q2, -1, 1)
            process_quadrant(map_q3, -1, -1)
            process_quadrant(map_q4, 1, -1)
            current_time = time.time() - start_time
            print("It took", current_time, "seconds to scale the traversability map for the planner.")
        except Exception as e:
            print("There was a problem:", e)
            current_time = time.time() - start_time
            print("Process ran for", current_time, "seconds.")

    def has_sufficient_data(self):
        return len(self.points_list) > 1000  # Adjust this threshold as needed

    # Build the traversability map
    def build_traversability_map(self):
        print(f"Building map with {len(self.points_list)} points")
        points_array = np.array(self.points_list)
        self.points_array = points_array
        self.kdtree = KDTree(points_array[:, :2])

        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        traversability_values = [self.extract_trav_value(point) for point in self.points_list]
        self.min_trav, self.max_trav = np.min(traversability_values), np.max(traversability_values)

        # Initialize empty maps for each quadrant
        result = self.initialize_empty_quadrants(min_x, max_x, min_y, max_y, self.scale_value, 0.001)
        map_q1, map_q2, map_q3, map_q4 = result

        # Process each quadrant
        map_q1 = self.process_quadrant(map_q1, 1, 1)
        map_q2 = self.process_quadrant(map_q2, -1, 1)
        map_q3 = self.process_quadrant(map_q3, -1, -1)
        map_q4 = self.process_quadrant(map_q4, 1, -1)

        return map_q1, map_q2, map_q3, map_q4
    
    def initialize_empty_quadrants(self, min_x, max_x, min_y, max_y, scale_value, initial_value):
        if (min_y > 0 and max_y > 0 and min_x > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
        elif (min_y > 0 and max_y > 0 and max_x <= 0):
            map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
            map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
            map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
            map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
        elif (max_y <= 0 and min_x > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
        elif (min_y > 0 and max_y > 0 and min_x <= 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
        elif (min_y <= 0 and max_y > 0 and min_x > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
        elif (max_y > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
        elif max_y <= 0 and max_x > 0:
            map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
        elif max_y <= 0 and max_x <= 0:
            map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
            map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
            map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
            map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
        elif max_y > 0 and max_x <= 0:
            map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
            map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
            map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
            map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
        return map_q1, map_q2, map_q3, map_q4
        #Function to initialize empty quadrant maps
    #def initialize_empty_quadrants(self, min_x, max_x, min_y, max_y, scale_value, initial_value):
       # if (min_y > 0 and max_y > 0 and min_x > 0 and max_x > 0):
      #      map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
      #      map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
      #      map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
       #     map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
        #elif (min_y > 0 and max_y > 0 and max_x <= 0):
         #   map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
          #  map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
           # map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
        #    map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
      #  elif (max_y <= 0 and min_x > 0 and max_x > 0):
       #     map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
        #    map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
         #   map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
          #  map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
#        elif (min_y > 0 and max_y > 0 and min_x <= 0 and max_x > 0):
 #           map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
  #          map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
   #         map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
    #        map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
     #   elif (min_y <= 0 and max_y > 0 and min_x > 0 and max_x > 0):
      #      map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
       #     map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
        #    map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
         #   map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
#        elif (max_y > 0 and max_x > 0):
 #           map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
#            map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
 #       elif max_y <= 0 and max_x > 0:
  #          map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
   #         map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
    ##       map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
      ##     map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
        #    map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
         #   map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
          #  map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
        #elif max_y > 0 and max_x <= 0:
         #   map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
          #  map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
           ##map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
        #return map_q1, map_q2, map_q3, map_q4