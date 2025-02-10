#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PointStamped
import time
import numpy as np
import matplotlib.pyplot as plt
from pose_listener import PoseListener
from path_planner import RRTStarPathPlanner
from advanced_path_planner import AdvancedRRTStarPathPlanner
from traversability_listener import TraversabilityListener
from metrics import MetricCollection

# Function to display four quadrant traversability map
def display_four_quadrant_traversability_map(travs_quad1, travs_quad2, travs_quad3, travs_quad4):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    cmap = plt.cm.viridis_r  # Reversed colormap: darker colors for lower values (more traversable)
    
    axs[0, 1].imshow(travs_quad1, cmap=cmap, origin='lower')
    axs[0, 1].set_title('Quadrant 1')
    
    axs[0, 0].imshow(np.fliplr(travs_quad2), cmap=cmap, origin='lower')
    axs[0, 0].set_title('Quadrant 2')
    
    axs[1, 0].imshow(np.flip(travs_quad3), cmap=cmap, origin='lower')
    axs[1, 0].set_title('Quadrant 3')
    
    axs[1, 1].imshow(np.flipud(travs_quad4), cmap=cmap, origin='lower')
    axs[1, 1].set_title('Quadrant 4')
    
    fig.suptitle('Traversability Map (Darker = More Traversable)')
    fig.supxlabel('X')
    fig.supylabel('Y')
    
    plt.tight_layout()
   # plt.show()

# Function to display four quadrant traversability map with path overlayed on it
def final_map_with_travs(travs_x, travs_y, travs_values, path, scale):
    plt.figure(figsize=(10, 8))
    plt.scatter(travs_x, travs_y, c=travs_values, cmap=plt.cm.viridis_t, s=10, alpha=0.8)
    plt.colorbar(label='Traversability Value (0 = Most Traversable)')
    plt.title('Final Map (Darker = More Traversable)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    x_coords = [node[0] / scale for node in path]
    y_coords = [node[1] / scale for node in path]
    plt.plot(x_coords, y_coords, 'o-', color='red')
    plt.plot(x_coords[0], y_coords[0], 'go')  # Start in green
    plt.plot(x_coords[-1], y_coords[-1], 'bo')  # End in blue
    plt.show()

# Graph overlay function
def graph_overlay(traversability_map, path, width, height):
    plt.imshow(traversability_map, cmap=plt.cm.viridis_r, origin='lower')
    plt.colorbar(label='Traversability (0 = Most Traversable)')
    plt.title('Traversability Map (Darker = More Traversable)')
    plt.xlabel('X')
    plt.ylabel('Y')
    x_coords = [node[0] for node in path]
    y_coords = [node[1] for node in path]
    plt.plot(x_coords, y_coords, 'o-', color='red')
    #plt.show()

# Function that calculates the distance between two points in a 2D space
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Function that displays plot of RRT* graph solution
def plot_solution(path, scale):
    # Extract x and y coordinates from the path and scale them
    x_coords = [node[0] / scale for node in path]
    y_coords = [node[1] / scale for node in path]

    # Create a new figure
    plt.figure()

    # Plot the path
    plt.plot(x_coords, y_coords, 'o-', color='blue')

    # Plot start and end points
    plt.plot(x_coords[0], y_coords[0], 'go')  # Start in green
    plt.plot(x_coords[-1], y_coords[-1], 'ro')  # End in red

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Path')

    # Calculate the limits for the plot based on scaled coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Add padding to the limits
    padding = 10
    plt.xlim(min_x - padding, max_x + padding)
    plt.ylim(min_y - padding, max_y + padding)

    # Show the plot
    #plt.show()

def random_waypoint_publisher():
    # Initialize publisher node
    rospy.init_node('random_waypoint_publisher', anonymous=True)
    
    # To publish to waypoint topic
    way_pub = rospy.Publisher('/way_point', PointStamped, queue_size=1)
    rate = rospy.Rate(1)

    finalX = rospy.get_param('~finalX', 10.0)  # Default to 10.0 if parameter is not found
    finalY = rospy.get_param('~finalY', 10.0)  # Default to 10.0 if parameter is not found
    scale = int(rospy.get_param('~scale', 5.0))  # Default to 5.0 if parameter is not found

    # Initialize the pose listener
    poseListener = PoseListener()
    # Set a two second pause before this line is executed
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()

    # Pause for 2 seconds
    time.sleep(2)
    # Get the current position of the vehicle
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    # Initialize the traversability listener
    traversListener = TraversabilityListener(scale)
    # Pause for 5 seconds
    time.sleep(5)
    # Generate a blank traversability map (Use in the case where one cannot be constructed)
    traversability_map = traversListener.generate_empty_map(100 * scale, 100 * scale)
    try:
        quad1, quad2, quad3, quad4 = traversListener.controller()
        display_four_quadrant_traversability_map(quad1, quad2, quad3, quad4)
    except:
        rospy.loginfo("Cannot find a stitched pointcloud. Using a blank traversability map.")
        quad1 = traversability_map
        quad2 = traversability_map
        quad3 = traversability_map
        quad4 = traversability_map
        
    # Semantic Based
    semantic_based_data(quad1, quad2, quad3, quad4, way_pub, rate, poseListener, scale, traversListener)
    # Geometric Based
    #geometric_based_data(way_pub, poseListener, quad1, quad2, quad3, quad4, scale)

    # If the agent is not close to the final coordinates,
    

# Function that publishes waypoints sequentially from the start position to the goal position
def navigate_path(path, way_pub, rate, poseListener, scale):
    # Iterate through path in order
    for point in path:
        # Start a timer
        start_time = time.time()
        while not rospy.is_shutdown():
            vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
            rospy.loginfo("Current vehicle position: ({}, {}, {})".format(vehicleX, vehicleY, vehicleZ))

            # Publish waypoint
            waypoint = PointStamped()
            waypoint.header.stamp = rospy.Time.now()
            waypoint.header.frame_id = "map"
            waypoint.point.x = point[0] / scale
            waypoint.point.y = point[1] / scale
            waypoint.point.z = vehicleZ
            way_pub.publish(waypoint)
            rospy.loginfo("Published waypoint: {}".format(waypoint))
            
            # Check if the vehicle is near the current waypoint
            if (abs(vehicleX - (point[0] / scale)) < 1.0 and
                abs(vehicleY - (point[1] / scale)) < 1.0 ):
                break  # Move to the next waypoint

            # Get current time in relation to the start time
            current_time = time.time() - start_time
            # If the current time is greater than 15 seconds, move to the next waypoint
            # This ensures that the loop does not run indefinitely
            if current_time > 15:
                rospy.loginfo("Waypoint not reached, moving to the next one...")
                break
            
            rate.sleep()

def semantic_based_data(quad1, quad2, quad3, quad4, way_pub, rate, poseListener, scale, traversListener):
    # Semantic Based
    traverse_time_list = []
    percent_time_list = []
    distance_traversed_list = []
    for i in range(0, 10):
        print("Trial:",i+1)
        # Get the current position of the vehicle
        vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
        if (i == 0 or i == 2 or i== 4 or i== 6 or i == 8 or i == 10):
            finalX = -15
            finalY = -57
        else:
            finalX = -34
            finalY = -27
        advanced_planner = AdvancedRRTStarPathPlanner(vehicleX, vehicleY, finalX, finalY, 100000, quad1, quad2, quad3, quad4, scale)
        # Plan the path
        path = advanced_planner.plan_path()
        if (i == 0 or i == 1):
            # Display solution using Matplotlib
            plot_solution(path, scale)
            travs_x = traversListener.travs_x
            travs_y = traversListener.travs_y
            travs_values = traversListener.traversability_values
            try:
                final_map_with_travs(travs_x, travs_y, travs_values, path, scale)
            except:
                pass
        # Start metrics collection
        metrics = MetricCollection()
        rate = rospy.Rate(10)  # 10 Hz
        while (not metrics.latest_cloud or metrics.sim_total_distance_traversed is None) and not rospy.is_shutdown():
            if not metrics.latest_cloud:
                rospy.loginfo("Waiting for point cloud data...")
            if metrics.sim_total_distance_traversed is None:
                rospy.loginfo("Waiting for distance data...")
            rate.sleep()
        metrics.start_timer()
        metrics.start_distance_traversed()
        metrics.start_time_on_trail()
        navigate_path(path, way_pub, rate, poseListener, scale)
        metrics.end_timer()
        metrics.end_time_on_trail()
        total_distance_traversed = metrics.end_distance_traversed()
        print("Total time:", metrics.total_time)
        print("Percent time on trail:", metrics.percent_time_on_trail)
        print("Total distance traversed:", total_distance_traversed)
        traverse_time_list.append(metrics.total_time)
        percent_time_list.append(metrics.percent_time_on_trail)
        distance_traversed_list.append(total_distance_traversed)   

    print("Traverse Times List")
    for i in traverse_time_list:
        print(i)
    print("Percent List")
    for i in percent_time_list:
        print(i)
    print("Distance List")
    for i in distance_traversed_list:
        print(i)  

if __name__=='__main__':
    try:
        random_waypoint_publisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except Exception as e:
        rospy.logerr("An error occurred: {}".format(str(e)))