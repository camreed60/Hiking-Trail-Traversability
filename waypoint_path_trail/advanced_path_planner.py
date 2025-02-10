#!/usr/bin/env python3

import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

class AdvancedRRTStarPathPlanner:
    def __init__(self, initial_x, initial_y, final_x, final_y, traversability_weight, 
                 travs_quadrant1 = None, travs_quadrant2 = None, travs_quadrant3 = None, 
                 travs_quadrant4 = None, scale = None):

        # Set the traversability maps for the four quadrants
        # If the traversability maps are not provided, set them to a 1x1 zero matrix
        if travs_quadrant1 is not None:
            self.travs_quadrant1 = travs_quadrant1
        else:
            self.travs_quadrant1 = np.zeros((1, 1))
        if travs_quadrant2 is not None:
            self.travs_quadrant2 = travs_quadrant2
        else:
            self.travs_quadrant2 = np.zeros((1, 1))
        if travs_quadrant3 is not None:
            self.travs_quadrant3 = travs_quadrant3
        else:
            self.travs_quadrant3 = np.zeros((1, 1))
        if travs_quadrant4 is not None:
            self.travs_quadrant4 = travs_quadrant4
        else:
            self.travs_quadrant4 = np.zeros((1, 1))
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1
        # Get the broad corner x, y coordinate from the four traversability quadrant maps
        max_x = max(self.travs_quadrant1.shape[1], self.travs_quadrant4.shape[1])
        min_x = max(self.travs_quadrant2.shape[1], self.travs_quadrant3.shape[1])
        min_x = - min_x
        max_y = max(self.travs_quadrant1.shape[0], self.travs_quadrant2.shape[0])
        min_y = max(self.travs_quadrant3.shape[0], self.travs_quadrant4.shape[0])
        min_y = - min_y
        initial_x = initial_x * scale
        initial_y = initial_y * scale
        final_x = final_x * scale
        final_y = final_y * scale

        self.initial_node = (initial_x, initial_y)
        self.goal_node = (final_x, final_y)

        # Set the boundary coordinates based on the x, y coordinates
        self.boundary_coordinates = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        self.traversability_weight = traversability_weight
        
        # Initialize nodes with the initial position
        self.nodes = {self.initial_node: {'parent': None, 'cost': 0}}
        self.goal_radius = 3.0 * self.scale # Radius for goal proximity
        self.min_step_size = 2.0 * self.scale # Minimum step size for each iteration
        self.step_size = 3.0 * self.scale # Maximum step size for each iteration
        self.search_radius = 3.0 * self.scale # Search radius for nearby nodes

    # Function to get the cost of a node
    def cost(self, node):
        return self.nodes[node]['cost'] if node in self.nodes else float('inf')

    def set_parent(self, node, parent):
        # Check for cycles before setting the parent
        current_node = parent
        while current_node:
            if current_node == node:
                return False  # Cycle detected, do not set parent
            current_node = self.get_parent(current_node)
        if node not in self.nodes:
            self.nodes[node] = {'parent': parent, 'cost': float('inf')}
        self.nodes[node]['parent'] = parent
        return True

    # Function to set the cost of a node
    def set_cost(self, node, cost_value):
        self.nodes[node]['cost'] = cost_value

    # Function to get the parent of a node
    def get_parent(self, node):
        return self.nodes[node]['parent'] if node in self.nodes else None

    # Function to calculate the distance between two nodes
    def distance(self, node1, node2):
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

    # Function to reconstruct the path from the goal to the start
    def reconstruct_path(self):
        path = []
        current_node = self.goal_node
        while current_node:
            path.append((current_node[0], current_node[1]))
            current_node = self.get_parent(current_node)
        path.reverse()
        return path

    # Function to sample a free point in the environment
    def sample_free(self):
        min_x = min(coordinate[0] for coordinate in self.boundary_coordinates)
        max_x = max(coordinate[0] for coordinate in self.boundary_coordinates)
        min_y = min(coordinate[1] for coordinate in self.boundary_coordinates)
        max_y = max(coordinate[1] for coordinate in self.boundary_coordinates)
        
        while True:
            point = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if self.is_within_boundary(point):
                return point

    # Function to check if a point is within the boundary
    def is_within_boundary(self, point):
        min_x = min(coordinate[0] for coordinate in self.boundary_coordinates)
        max_x = max(coordinate[0] for coordinate in self.boundary_coordinates)
        min_y = min(coordinate[1] for coordinate in self.boundary_coordinates)
        max_y = max(coordinate[1] for coordinate in self.boundary_coordinates)
        x, y = point
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    # Function to find the nearest node to a given node
    def nearest(self, node):
        return min(self.nodes, key=lambda n: self.distance(n, node))

    # Function to steer from one node to another
    def steer(self, from_node, to_node):
        distance = self.distance(from_node, to_node)
        if distance > self.min_step_size and distance < self.step_size:
            return to_node
        theta = math.atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
        if distance > self.min_step_size:
            calculated_node = (from_node[0] + self.step_size * math.cos(theta), from_node[1] + self.step_size * math.sin(theta))
            return calculated_node
            #return self.steer_travs(from_node, calculated_node)
        else:
            calculated_node = (from_node[0] + self.min_step_size * math.cos(theta), from_node[1] + self.min_step_size * math.sin(theta))
            return calculated_node
            #return self.steer_travs(from_node, calculated_node)

    # Function to steer new node towards the highest traversable area
    def steer_travs(self, from_node, calculated_node):
        # Get the frontier around the from node
        frontier = self.get_frontier(from_node)
        minX = from_node[0] - int(self.step_size)
        minY = from_node[1] - int(self.step_size)
        # Compare the traversability of the frontier with the traversability of the calculated node
        if self.get_traversability(calculated_node) < np.max(frontier):
            coordinates = self.get_highest_coordinates(frontier)
            # Get the x and y coordinates of the maximum traversability value
            max_x = coordinates[0]
            max_y = coordinates[1]
            # make a list of nodes
            list_nodes = []
            for x in max_x:
                for y in max_y:
                    # If the node is within the boundary
                    if (minX + x, minY + y) in self.boundary_coordinates:
                        xy_node = (minX + x, minY + y)
                        list_nodes.append(xy_node)
            new_calculated_node = calculated_node
            lowest_distance = float('inf')
            # Return the closest node within the highest traversable area
            for nodes in list_nodes:
                # Calculate the distance between the calculated node and the node in the list
                # Determine if it is less than the lowest distance
                if self.distance(nodes, calculated_node) < lowest_distance:
                    # Set the lowest cost to the distance
                    lowest_distance = self.distance(nodes, calculated_node)
                    # Set the calculated node to the node in the list
                    new_calculated_node = nodes
            return new_calculated_node
        return calculated_node

    # Function to find nodes near a new node
    def near_nodes(self, new_node):
        return [node for node in self.nodes if self.distance(node, new_node) <= self.search_radius]

    # Function to get the traversability value of a node (It will be between 0 <= x <= 1)
    def get_traversability(self, node):
        try:
            x, y = int(node[0]), int(node[1])
            # If X and Y are positive, set traversability map to quadrant 1
            if x >= 0 and y >= 0:
                travs_map = self.travs_quadrant1
            # If X is negative and Y is positive, set traversability map to quadrant 2
            elif x < 0 and y >= 0:
                travs_map = self.travs_quadrant2
            # If X and Y are negative, set traversability map to quadrant 3
            elif x < 0 and y < 0:
                travs_map = self.travs_quadrant3
            # If X is positive and Y is negative, set traversability map to quadrant 4
            else:
                travs_map = self.travs_quadrant4
            # Set X and Y to their absolute values
            x = abs(x)
            y = abs(y)
            # Check if the node borders the boundary
            if (x == 0 or x == travs_map.shape[1] - 1 or y == 0 or y == travs_map.shape[0] - 1):
                return travs_map[y][x]
            # Get the smallest adjacent block's traversability value
            smallest_value = travs_map[y][x]
            return smallest_value
        except IndexError:
            return 0
    
    # Function to calculate the cost of a new node
    def calculate_cost(self, node, parent):
        distance_cost = self.distance(parent, node)
        node_traversability = self.get_traversability(node)
        parent_traversability = self.get_traversability(parent)
        if (node_traversability == 1 and parent_traversability == 1):
            return self.cost(parent) + distance_cost
        else:
            try:
                traversability_cost = self.traversability_weight / ((node_traversability + parent_traversability) / 2)
            except ZeroDivisionError:
                traversability_cost = float('inf')
            return self.cost(parent) + distance_cost + traversability_cost

    # Function to get the frontier of the traversability map around the node (Based on Traversability Map)
    def get_frontier(self, node):
        x, y = int(node[0]), int(node[1])
        # If X and Y are positive, set traversability map to quadrant 1
        if x >= 0 and y >= 0:
            travs_map = self.travs_quadrant1
        # If X is negative and Y is positive, set traversability map to quadrant 2
        elif x < 0 and y >= 0:
            travs_map = self.travs_quadrant2
        # If X and Y are negative, set traversability map to quadrant 3
        elif x < 0 and y < 0:
            travs_map = self.travs_quadrant3
        # If X is positive and Y is negative, set traversability map to quadrant 4
        else:
            travs_map = self.travs_quadrant4
        # Set X and Y to their absolute values
        x = abs(x)
        y = abs(y)
        # Get a slice of the traversability map around the node
        # Based on the max step size
        x_range = slice(max(0, x - int(self.step_size)), min(travs_map.shape[1], x + int(self.step_size) + 1))
        y_range = slice(max(0, y - int(self.step_size)), min(travs_map.shape[0], y + int(self.step_size) + 1))
        return travs_map[y_range, x_range]

    # Function to get the coordinates from the highest coordinates of the frontier
    def get_highest_coordinates(self, frontier):
        # Get the maximum traversability value
        max_value = np.max(frontier)
        # Get the x and y coordinates of the maximum traversability value
        max_x, max_y = np.where(frontier == max_value)
        return max_x, max_y

    # Function to plan the path from the start to the goal using RRT*
    def plan_path(self):
        print("Generating a path...")
        # Calculate the distance between the start and the goal
        distance = self.distance(self.initial_node, self.goal_node)
        # If the distance is less than the step size
        if distance < self.step_size:
            return [self.initial_node, self.goal_node]
        time_limit = 60
        # Based on the distance, calculate the time limit
        if distance < 120:
            time_limit = distance / 2
        if time_limit > 10:
            # Merge four quandrants together to check what percentage of the environment is traversable
            traversability_map = (np.count_nonzero(self.travs_quadrant1 == 1) + np.count_nonzero(self.travs_quadrant2 == 1)
                                + np.count_nonzero(self.travs_quadrant3 == 1) + np.count_nonzero(self.travs_quadrant4 == 1)) / (
                                    self.travs_quadrant1.size + self.travs_quadrant2.size + self.travs_quadrant3.size + 
                                    self.travs_quadrant4.size)
            # If 75% or more of the environment is traversable, set the time limit to 10 seconds
            if traversability_map >= 0.75:
                time_limit = 10
        
        # The time limit is currently being overwritten to be 2 minutes
        time_limit = 120
        start_time = time.time()
        goal_found = False
        while True:
            random_point = self.sample_free()  # Sample a random point
            nearest_node = self.nearest(random_point)  # Find the nearest node to the random point
            new_node = self.steer(nearest_node, random_point)  # Steer towards the random point
            # Calculate the cost of the new node considering the traversability
            traversability = self.get_traversability(new_node)
            if traversability == 0:  # Skip nodes that are in impassable areas
                # Get current time in relation to the start time
                current_time = time.time() - start_time
                # If the current time is greater than 60 seconds
                # This ensures that the loop does not run indefinitely
                if current_time > time_limit:
                    break
                continue
            new_cost = self.calculate_cost(new_node, nearest_node)
            
            # If the new node is not in nodes or its cost is less than the current cost
            if new_node not in self.nodes or new_cost < self.cost(new_node):
                if self.set_parent(new_node, nearest_node):
                    self.set_cost(new_node, new_cost)
                    # For each node near the new node
                    for near_node in self.near_nodes(new_node):
                        if near_node == nearest_node:
                            continue
                        
                        # Check traversability between new_node and near_node
                        traversability_near = self.get_traversability(near_node)
                        if traversability_near == 0:  # Skip nodes that are in impassable areas
                            continue
                        
                        new_near_cost = self.calculate_cost(near_node, new_node) # Calculate the new near cost
                        
                        # If the new near cost is less than the current cost of the near node
                        if new_near_cost < self.cost(near_node):
                            if self.set_parent(near_node, new_node): # Set the parent of the near node to the new node
                                self.set_cost(near_node, new_near_cost) # Set the cost of the near node to the new near cost
                    
                    if (not goal_found):
                        # If the new node is within the goal radius
                        if self.distance(new_node, self.goal_node) < self.goal_radius:
                            goal_traversability = self.get_traversability(self.goal_node)
                            if goal_traversability > 0:  # Ensure the goal node is in a traversable area
                                self.nodes[self.goal_node] = {'parent': new_node, 'cost': new_cost + self.distance(new_node, self.goal_node) / goal_traversability}
                                goal_found = True # Set goal found to True
                                
                                # Rewire the tree considering traversability
                                for near_node in self.near_nodes(self.goal_node):
                                    if near_node == new_node:
                                        continue
                                    
                                    traversability_near = self.get_traversability(near_node)
                                    if traversability_near == 0:  # Skip nodes that are in impassable areas
                                        continue
                                    
                                    new_near_cost = self.calculate_cost(near_node, self.goal_node) # Calculate the new near cost
                                    if new_near_cost < self.cost(self.goal_node):
                                        if (self.set_parent(self.goal_node, near_node)):
                                            self.set_cost(self.goal_node, new_near_cost)
            
            current_time = time.time() - start_time
            if current_time > time_limit:
                break
        
        # If the goal was not found, find the nearest node to the goal
        if not goal_found:
            nearest_node = self.nearest(self.goal_node)
            # Add the final point to nodes
            self.nodes[self.goal_node] = {'parent': nearest_node, 'cost': self.cost(nearest_node) + self.distance(nearest_node, self.goal_node)}
        
        # Rewire the whole tree
        for node in self.nodes:
            for near_node in self.near_nodes(node):
                if near_node == self.get_parent(node):
                    continue
                traversability_near = self.get_traversability(near_node)
                if traversability_near == 0:  # Skip nodes that are in impassable areas
                    continue
                new_near_cost = self.calculate_cost(near_node, node) # Calculate the new near cost
                # If the new near cost is less than the current cost of the near node
                if new_near_cost < self.cost(near_node):
                    self.set_parent(near_node, node) # Set the parent of the near node to the new node
                    self.set_cost(near_node, new_near_cost) # Set the cost of the near node to the new near cost
        
        current_time = time.time() - start_time
        # Print the time it took to find a solution
        print("It took", current_time, "seconds to find a path.")
        '''
        self.plot_whole_tree(self.nodes, [
            (-1000, 0),
            (-1000, 1000),
            (1000, 1000),
            (1000, 0)
        ])
        '''
         
        # Return the reconstructed path from the goal to the start
        return self.reconstruct_path()

    # Function to plot the whole search tree for debug purposes
    def plot_whole_tree(self, nodes, boundary_coordinates):
        # Create a new figure
        plt.figure()

        # Plot the boundary
        boundary_x = [coordinate[0] for coordinate in boundary_coordinates]
        boundary_y = [coordinate[1] for coordinate in boundary_coordinates]
        plt.plot(boundary_x + [boundary_x[0]], boundary_y + [boundary_y[0]], 'k--', label='Boundary')  # close the boundary

        plt.title('Whole Tree')
        plt.xlabel('X')
        plt.ylabel('Y')
        # Plot the nodes and the edges
        for node, data in nodes.items():
            if data['parent'] is not None:
                parent_node = data['parent']
                plt.plot([parent_node[0], node[0]], [parent_node[1], node[1]], 'r-')  # edge from parent to child
            plt.plot(node[0], node[1], 'bo')  # node itself

        # Set labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Whole Tree')
        plt.legend()

        # Show the plot
        plt.show()
