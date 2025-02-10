#!/usr/bin/env python3

import math
import random
import time

class RRTStarPathPlanner:
    def __init__(self, initial_x, initial_y, final_x, final_y, boundary_coordinates):
        self.initial_node = (initial_x, initial_y)
        self.goal_node = (final_x, final_y)
        self.boundary_coordinates = boundary_coordinates
        
        # Initialize nodes with the initial position
        self.nodes = {self.initial_node: {'parent': None, 'cost': 0}}
        self.goal_radius = 1.0  # Radius for goal proximity
        self.min_step_size = 2.0  # Minimum step size for each iteration
        self.step_size = 4.0  # Maximum step size for each iteration
        self.search_radius = 8.0  # Search radius for nearby nodes

    # Function to get the cost of a node
    def cost(self, node):
        return self.nodes[node]['cost'] if node in self.nodes else float('inf')

    # Function to set the parent of a node
    def set_parent(self, node, parent):
        self.nodes[node]['parent'] = parent

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

    def sample_free(self):
        # Use boundary box to sample points
        min_x = min(coordinate[0] for coordinate in self.boundary_coordinates)
        max_x = max(coordinate[0] for coordinate in self.boundary_coordinates)
        min_y = min(coordinate[1] for coordinate in self.boundary_coordinates)
        max_y = max(coordinate[1] for coordinate in self.boundary_coordinates)
        
        return (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

    # Function to find the nearest node to a given node
    def nearest(self, node):
        return min(self.nodes, key=lambda n: self.distance(n, node))

    # Function to steer from one node to another
    def steer(self, from_node, to_node):
        if self.distance(from_node, to_node) >  self.min_step_size and self.distance(from_node, to_node) < self.step_size:
            return to_node
        theta = math.atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
        if self.distance(from_node, to_node) > self.min_step_size:
            return (from_node[0] + self.step_size * math.cos(theta), from_node[1] + self.step_size * math.sin(theta))
        else:
            return (from_node[0] + self.min_step_size * math.cos(theta), from_node[1] + self.min_step_size * math.sin(theta))

    # Function to find nodes near a new node
    def near_nodes(self, new_node):
        return [node for node in self.nodes if self.distance(node, new_node) <= self.search_radius]

    # Function to plan the path from the start to the goal using RRT*
    def plan_path(self):
        start_time = time.time()
        goal_found = False
        while not goal_found:
            random_point = self.sample_free()  # Sample a random point
            nearest_node = self.nearest(random_point)  # Find the nearest node to the random point
            new_node = self.steer(nearest_node, random_point)  # Steer towards the random point
            new_cost = self.cost(nearest_node) + self.distance(nearest_node, new_node)  # Calculate the cost of the new node
            
            # If the new node is not in nodes or its cost is less than the current cost
            if new_node not in self.nodes or new_cost < self.cost(new_node):
                self.nodes[new_node] = {'parent': nearest_node, 'cost': new_cost}  # Add the new node to nodes

                # For each node near the new node
                for near_node in self.near_nodes(new_node):
                    if near_node == nearest_node:
                        continue
                    new_near_cost = new_cost + self.distance(new_node, near_node)
                    # If the new near cost is less than the current cost of the near node
                    if new_near_cost < self.cost(near_node):
                        self.set_parent(near_node, new_node) # Set the parent of the near node to the new node
                        self.set_cost(near_node, new_near_cost) # Set the cost of the near node to the new near cost
                
                # If the new node is within the goal radius
                if self.distance(new_node, self.goal_node) < self.goal_radius:
                    self.nodes[self.goal_node] = {'parent': new_node, 'cost': new_cost + self.distance(new_node, self.goal_node)}
                    goal_found = True # Set goal found to True
                    # Rewire the tree
                    for near_node in self.near_nodes(self.goal_node):
                        if near_node == new_node:
                            continue
                        new_near_cost = self.cost(near_node) + self.distance(self.goal_node, near_node)
                        if new_near_cost < self.cost(self.goal_node):
                            self.set_parent(self.goal_node, near_node)
                            self.set_cost(self.goal_node, new_near_cost)
                    break
            
            # Get current time in relation to the start time
            current_time = time.time() - start_time
            # If the current time is greater than 60 seconds, break out
            # This ensures that the loop does not run indefinitely
            if current_time > 60:
                break
        
        # If the goal was not found, find the nearest node to the goal
        if not goal_found:
            nearest_node = self.nearest(self.goal_node)
            # Add the final point to nodes
            self.nodes[self.goal_node] = {'parent': nearest_node, 'cost': self.cost(nearest_node) + self.distance(nearest_node, self.goal_node)}
        
        # Add an additional 1,000 nodes to improve the tree
        for _ in range(1000):
            random_point = self.sample_free()
            nearest_node = self.nearest(random_point)
            new_node = self.steer(nearest_node, random_point)
            new_cost = self.cost(nearest_node) + self.distance(nearest_node, new_node)
            
            if new_node not in self.nodes or new_cost < self.cost(new_node):
                self.nodes[new_node] = {'parent': nearest_node, 'cost': new_cost}

                for near_node in self.near_nodes(new_node):
                    if near_node == nearest_node:
                        continue
                    new_near_cost = new_cost + self.distance(new_node, near_node)
                    if new_near_cost < self.cost(near_node):
                        self.set_parent(near_node, new_node)
                        self.set_cost(near_node, new_near_cost)
        
        # Get current time in relation to the start time
        current_time = time.time() - start_time
        # If the current time is less than 30 seconds, rewire the whole tree
        if current_time < 30:
            for node in self.nodes:
                for near_node in self.near_nodes(node):
                    if near_node == self.get_parent(node):
                        continue
                    new_near_cost = self.cost(node) + self.distance(node, near_node)
                    if new_near_cost < self.cost(near_node):
                        self.set_parent(near_node, node)
                        self.set_cost(near_node, new_near_cost)
        
        # Return the reconstructed path from the goal to the start
        return self.reconstruct_path()