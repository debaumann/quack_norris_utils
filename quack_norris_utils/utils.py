#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely.geometry as sg
import shapely.ops as so
import shapely.affinity as sa
from typing import List, Tuple, Optional, Union

import rospy

from enum import Enum


class SETransform:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __call__(self, geom):
        return sa.rotate(sa.translate(geom, self.x, self.y), self.theta, origin=(0, 0))



class CircleType(Enum):
    LEFT = -1
    RIGHT = 1

class DubinCircle():
    def __init__(self, center: Tuple[float, float], radius: float, type: CircleType):
        self.center = center
        self.type = type.value
        self.x = center[0]
        self.y = center[1]
        self.radius = radius

class DuckieDriverObstacle():
    def __init__(self, pose: SETransform, speed: float, lookahead_time: float, lookahead_point: SETransform):
        self.pose = pose
        self.length = 0.2
        self.width = 0.15
        self.radius = self.width + 0.1
        self.speed = speed
        self.dt = 0.1
        self.lookahead_time = lookahead_time
        self.path = []
        self.end = None
        self.lookahead_point = lookahead_point
        self.obstacle = self.get_obstacle()
        
    def get_obstacle(self):
        d_over_time = self.speed*self.lookahead_time + self.length
        x = self.pose.x + d_over_time*np.cos(self.pose.theta)
        y = self.pose.y + d_over_time*np.sin(self.pose.theta)
        self.end = SETransform(x, y, self.pose.theta)
        obst = sg.LineString([(self.pose.x,self.pose.y), (x, y)]).buffer(self.width)
        return obst
    def get_poses_dubins(self, ambulance_pose: SETransform):
        placement_x = self.pose.x + self.radius*np.cos(ambulance_pose.theta + np.pi/2)
        placement_y = self.pose.y + self.radius*np.sin(ambulance_pose.theta + np.pi/2)
        placement_theta = ambulance_pose.theta
        if self.lookahead_point is not None:

            end_placement_theta = np.arctan2(self.lookahead_point.y - placement_y, self.lookahead_point.x - placement_x) - 0.1
            print('end placement theta', np.rad2deg(end_placement_theta))
        else:
            end_placement_theta = ambulance_pose.theta
            
        end_placement_x = self.end.x + self.radius*np.cos(end_placement_theta +np.pi/2)
        end_placement_y = self.end.y + self.radius*np.sin(end_placement_theta + np.pi/2)
       

        return SETransform(placement_x, placement_y, placement_theta), SETransform(end_placement_x,end_placement_y,end_placement_theta)

        
class DuckieSegment:
    def __init__(self, start: SETransform, end: SETransform, radius: float, type: str, path: sg.LineString, cost: float,speed:float):   
        self.start = start
        self.end = end
        self.radius = radius
        self.type = type
        self.sector = end.theta - start.theta
        self.speed = speed
        self.shapely_path= path
        self.cost = None
        self.dt = 0.1
        self.cost = cost
    def get_path_array(self):
        if self.type == 'STRAIGHT':
            path_coords = np.array(self.shapely_path.coords)
            x = path_coords[:, 0]
            y = path_coords[:, 1]
            theta = np.ones(len(x))*self.start.theta
            angular_speed = np.zeros(len(x))
            radius = np.ones(len(x))*np.inf
            return np.vstack([x, y, theta,angular_speed,radius]).T
        elif self.type == 'LEFT':
            path_coords = np.array(self.shapely_path.coords)
            x = path_coords[:, 0]
            y = path_coords[:, 1]
            angles = np.linspace(0, self.sector, len(x))
            angular_speed =  self.speed/self.radius * np.ones(len(x))
            theta = self.start.theta + angles 
            radius = self.radius*np.ones(len(x))
            return np.vstack([x, y, theta,angular_speed,radius]).T
        elif self.type == 'RIGHT':
            path_coords = np.array(self.shapely_path.coords)
            x = path_coords[:, 0]
            y = path_coords[:, 1]
            angles = np.linspace(0, self.sector, len(x))
            angular_speed =  -self.speed/self.radius * np.ones(len(x))
            theta = self.start.theta - angles 
            radius = self.radius*np.ones(len(x))
            return np.vstack([x, y, theta,angular_speed,radius]).T

class DuckieCorner:
    def __init__(self, pose:SETransform, radius: float, type: str):
        self.pose = pose
        self.radius = radius
        self.type = type
        self.shapely_obs = self.get_obs() 
        self.placement = self.get_poses_dubins()
    def get_poses_dubins(self):
        if self.type == 'LEFT':
            self.placement_x = self.pose.x + self.radius*np.cos(self.pose.theta - np.pi/2)
            self.placement_y = self.pose.y + self.radius*np.sin(self.pose.theta - np.pi/2)
        elif self.type == 'RIGHT':
            self.placement_x = self.pose.x + self.radius*np.cos(self.pose.theta + np.pi/2)
            self.placement_y = self.pose.y + self.radius*np.sin(self.pose.theta + np.pi/2)
        elif self.type == 'STRAIGHT':
            self.placement_x = self.pose.x
            self.placement_y = self.pose.y
        self.placement_theta = self.pose.theta

        return SETransform(self.placement_x, self.placement_y, self.placement_theta)
        
    def get_obs(self):
        str_to_int = {"LEFT": 1, "STRAIGHT": 0, "RIGHT": 1}
        return sg.Point((self.pose.x,self.pose.y)).buffer((self.radius-0.1)*str_to_int[self.type])
    def update_corners(self, new_radius):
        self.radius = new_radius
        self.shapely_obs = self.get_obs() 
        self.placement = self.get_poses_dubins()
    
class DuckieObstacle:
    def __init__(self, pose: SETransform, radius: float): #type is left or right
        self.pose = pose
        self.radius = radius
        self.shapely_obs = sg.Point(pose.x, pose.y).buffer(radius)
        self.obs_speed = 0.1

    def check_collision(self, path: sg.LineString):
        
        return path.intersects(self.shapely_obs)
    def get_poses_dubins(self, duckie_pose: SETransform):
        placement_x = self.pose.x + self.radius*np.cos(duckie_pose.theta + np.pi/2)
        placement_y = self.pose.y + self.radius*np.sin(duckie_pose.theta + np.pi/2)
        placement_theta = duckie_pose.theta
        return SETransform(placement_x, placement_y, placement_theta)

class DuckieNode:
    def __init__(self,pose: SETransform, parent = None, next = None, tag_id = None):
        self.x = pose.x
        self.y = pose.y
        self.theta = pose.theta
        self.pose = pose
        self.parent = None
        self.next = None
        self.tag_id = tag_id
        self.corner = None
    def insert_corner(self, corner: DuckieCorner):
        self.corner = corner
    def insert_next(self, node):
        self.next = node
    def insert_parent(self, node):
        self.parent = node
    def insert_tag(self, tag_id):
        self.tag_id = tag_id
    
    
class dubins:
    def __init__(self,start: SETransform, goal: SETransform,n_segments: int, speed: float,radius1: float,radius2: float):
        self.start = start
        self.goal = goal
        self.speed = speed
        self.n_segments = n_segments
        self.radius1 = radius1
        self.radius2 = radius2
        self.path = []
        self.circle_radii = [0.3,0.6,0.9]

    def get_circles(self,pose: SETransform, radius: float) -> List[patches.Circle]:
        pose_normal_vector = np.array([np.cos(pose.theta), np.sin(pose.theta)])
        pose_center = np.array([pose.x, pose.y])
        left_center = pose_center + radius * np.array([-pose_normal_vector[1], pose_normal_vector[0]])
        right_center = pose_center + radius * np.array([pose_normal_vector[1], -pose_normal_vector[0]])
        left_circle = DubinCircle(left_center, radius, CircleType.LEFT)
        right_circle = DubinCircle(right_center, radius, CircleType.RIGHT)
        return [left_circle, right_circle]
    def approximate_straight(self,start:SETransform, end:SETransform):
        n_seg = 20
        dist = np.linalg.norm(np.array([start.x, start.y]) - np.array([end.x, end.y]))
        straight_cost = dist
        x = np.linspace(start.x, end.x, n_seg+1)
        y = np.linspace(start.y, end.y, n_seg+1)
        points = []
        for i in range(n_seg+1):
            points.append([x[i], y[i]])
        line_string = sg.LineString(points)
        return line_string, straight_cost
    def approximate_circle(self,start:SETransform, end:SETransform,circle: DubinCircle):
        #get the angle between the start and end
        start_angle = np.arctan2(start.y - circle.y, start.x - circle.x)
        end_angle = np.arctan2(end.y - circle.y, end.x - circle.x)
        if circle.type == CircleType.LEFT.value:
            if start_angle < 0:
                start_angle += 2*np.pi
            if end_angle < 0:
                end_angle += 2*np.pi
            if start_angle > end_angle:
                end_angle += 2*np.pi
            angle = end_angle - start_angle
            if angle < 0:
                angle += 2*np.pi
        else:
            if start_angle > 0:
                start_angle -= 2*np.pi
            if end_angle > 0:
                end_angle -= 2*np.pi
            if start_angle < end_angle:
                end_angle -= 2*np.pi
            angle = end_angle - start_angle
            if angle > 0:
                angle -= 2*np.pi
        
        n_seg = 20
        circ_cost = np.abs(angle*circle.radius)
        angle_step = angle/n_seg
        points = []
        for i in range(n_seg+1):
            theta = start_angle + i*angle_step
            x = circle.x + circle.radius*np.cos(theta)
            y = circle.y + circle.radius*np.sin(theta)
            points.append([x, y])
        line_string = sg.LineString(points)
        return line_string,circ_cost
    def cost(self, result):
        #sum the segments of start circle 
        sector1 = result['sector1']
        c1  = np.abs(sector1[1]* sector1[0])
        sector2 = result['sector2']
        c2 =np.abs(sector2[1]* sector2[0])
        p1 = np.array([result['p1'].x, result['p1'].y])
        p2 = np.array([result['p2'].x, result['p2'].y])
        c3 = np.linalg.norm(p1 - p2)

        return c1 + c2 + c3
       
    def solve_single(self, radius1: float, radius2: float):
        circle1 = self.get_circles(self.start, self.radius1)
        circle2 = self.get_circles(self.goal, self.radius2)
        results = {}
        
        c1 = circle1
        c2 = circle2

        

        solve_result = self.external_solve(self.start, self.goal, c1[0], c2[0])
        temp_result = {}
        if solve_result is not None:

            temp_result['path'] = solve_result
            cost = 0
            for duckie_p in solve_result:
                cost += duckie_p.cost
            temp_result['cost'] = cost
            results['ll'] = temp_result


        temp_result = {}
        solve_result = self.internal_solve(self.start, self.goal, c1[0], c2[1])
        if solve_result is not None:
            temp_result['path'] = solve_result
            cost = 0
            for duckie_p in solve_result:
                cost += duckie_p.cost
            temp_result['cost'] = cost
            results['lr'] = temp_result

        temp_result = {}
        solve_result = self.external_solve(self.start, self.goal, c1[1], c2[1])
        if solve_result is not None:
            temp_result['path'] = solve_result
            cost = 0
            for duckie_p in solve_result:
                cost += duckie_p.cost
            temp_result['cost'] = cost
            results['rr'] = temp_result

        temp_result = {}
        solve_result = self.internal_solve(self.start, self.goal, c1[1], c2[0])
        if solve_result is not None:
            temp_result['path'] = solve_result
            cost = 0
            for duckie_p in solve_result:
                cost += duckie_p.cost
            temp_result['cost'] = cost
            results['rl'] = temp_result
            
        if results == {}:
            return None
        best_key = min(results, key=lambda k: results[k]['cost'])
        best_result = results[best_key]
        return best_result
    
        
    def solve(self, multi_circles: bool = False, obstacle_pose:  int = None):
        best_paths = []
        #obstacle_pose is the index of the circle that is the obstacle 0 for none
        if obstacle_pose is not None and multi_circles:
            if obstacle_pose == 0:
                for r in self.circle_radii:
                    result = self.solve_single(r, self.radius2)
                    
                    
                    if result is not None:
                        print('solving with radius', r, 'cost of path'  ,result['cost'])
                        best_paths.append(result)
            elif obstacle_pose == 1: 
                for r in self.circle_radii:
                    result = self.solve_single(self.radius1, r)
                    if result is not None:
                        best_paths.append(result)
        elif multi_circles and obstacle_pose is None:
            for r1 in self.circle_radii:
                for r2 in self.circle_radii:
                    result = self.solve_single(r1, r2)
                    if result is not None:
                        best_paths.append(result)
        else:
            result = self.solve_single(self.radius1, self.radius2)
            if result is not None:
                best_paths.append(result)
        
        if best_paths == []:
            return None
        best_result = min(best_paths, key=lambda x: x['cost'])

        return best_result['path']
        

            
        #find the shortest path with n segments
    
    def external_solve(self,start,end,circle1,circle2):
        #find the tangent lines connecting the circles
         #for case of same radii 
        y_mir = 1
        alpha  = np.arctan2(circle2.y - circle1.y, circle2.x - circle1.x)
        if circle1.radius == circle2.radius:
            #find the two possible circles
            alpha  = np.arctan2(circle2.y - circle1.y, circle2.x - circle1.x)
            d = np.linalg.norm(np.array(circle1.center) - np.array(circle2.center))
            p1_x =  circle1.x + circle1.radius * np.cos(alpha + circle1.type*np.pi/2)
            p1_y =  circle1.y + circle1.radius * np.sin(alpha + circle1.type*np.pi/2)
            p2_x =  circle2.x + circle2.radius * np.cos(alpha + circle2.type*np.pi/2)
            p2_y =  circle2.y + circle2.radius * np.sin(alpha + circle2.type*np.pi/2)
        else: 
            if circle1.radius < circle2.radius:
                circle1,circle2 = circle2,circle1
                y_mir = -1
                switcheroo = True
            else:
                switcheroo = False
            
            d = np.linalg.norm(np.array(circle1.center) - np.array(circle2.center))
            # if d < np.min([circle1.radius,circle2.radius])/2:
            #     return None
            new_radius = np.abs(circle1.radius - circle2.radius)
            inbetween_center = [circle1.x + (circle2.x-circle1.x)/2, circle1.y + (circle2.y-circle1.y)/2]
            inbetween_radius = d/2
            d1 = (new_radius**2 )/(2*inbetween_radius)
            if new_radius**2 - d1**2 < 0:
                print('error','newrad', new_radius,'d1',d1, 'inbetween',inbetween_radius)
                return None

            h = np.sqrt(new_radius**2 - d1**2)
            rel_angle = np.arctan2(circle1.type*h,y_mir*d1)

            theta = rel_angle + alpha
            if switcheroo:
                circle1, circle2 = circle2, circle1
            p1_x =  circle1.x + circle1.radius * np.cos(theta)
            p1_y =  circle1.y + circle1.radius * np.sin(theta)
            p2_x =  circle2.x + circle2.radius * np.cos(theta)
            p2_y =  circle2.y + circle2.radius * np.sin(theta)
        start_angle = start.theta + circle1.type*np.pi/2
        end_angle = end.theta + circle1.type*np.pi/2
        sector1 = [np.arctan2(p1_y - circle1.y, p1_x - circle1.x) - start_angle, circle1.radius]
        sector2 = [np.arctan2(p2_y - circle2.y, p2_x - circle2.x) - end_angle, circle2.radius]
        p1_angle = np.arctan2(p1_y - circle1.y, p1_x - circle1.x) - circle1.type * np.pi/2
        p2_angle = np.arctan2(p2_y - circle2.y, p2_x - circle2.x) - circle1.type * np.pi/2
        p1 = SETransform(p1_x, p1_y, p1_angle)
        p2 = SETransform(p2_x, p2_y, p2_angle)
        line_1,_ = self.approximate_straight(p1,p2)
        start_circle,start_circle_cost = self.approximate_circle(start,p1, circle1)
        end_circle,end_circle_cost= self.approximate_circle(p2,end, circle2)
        straight_cost = np.linalg.norm(np.array([p1_x, p1_y]) - np.array([p2_x, p2_y]))
        if circle1.type == 1:
            c1_type = 'RIGHT'
        else:
            c1_type = 'LEFT'
        if circle2.type == 1:
            c2_type = 'RIGHT'
        else:
            c2_type = 'LEFT'
        cost = start_circle_cost + end_circle_cost + straight_cost 
        res_dict = {'line': line_1, 'sector1': sector1, 'sector2': sector2, 'p1': p1, 'p2': p2,'start': start, 'goal': end,
                    'start_circle': start_circle, 'end_circle': end_circle,'cost': cost, 'c1_type': c1_type, 'c2_type': c2_type}
        
        duckie_path= []
        
        duckie_path.append(DuckieSegment(start, p1, sector1[1], c1_type ,start_circle,start_circle_cost,self.speed))
        duckie_path.append(DuckieSegment(p1, p2, 0, 'STRAIGHT', line_1,straight_cost,self.speed))
        duckie_path.append(DuckieSegment(p2,end, sector2[1], c2_type, end_circle,end_circle_cost,self.speed))
    
        return duckie_path
    
    def internal_solve(self,start,end, circle1,circle2):
    #find the shortest path between two circles
        alpha  = np.arctan2(circle2.y - circle1.y, circle2.x - circle1.x)
        d = np.linalg.norm(np.array(circle1.center) - np.array(circle2.center))
        if d < circle1.radius + circle2.radius:
                print('error for internal')
                return None
        new_radius = np.abs(circle1.radius +circle2.radius)
        inbetween_center = [circle1.x + (circle2.x-circle1.x)/2, circle1.y + (circle2.y-circle1.y)/2]
        inbetween_radius = d/2
        d1 = (new_radius**2 )/(2*inbetween_radius)
        

        h = np.sqrt(new_radius**2 - d1**2)
        rel_angle = np.arctan2(h,d1)

        theta = circle1.type*rel_angle + alpha
        p1_x =  circle1.x + circle1.radius * np.cos(theta)
        p1_y =  circle1.y + circle1.radius * np.sin(theta)
        p2_x =  circle2.x + circle2.radius * np.cos(theta+np.pi)
        p2_y =  circle2.y + circle2.radius * np.sin(theta+np.pi)

        start_angle = start.theta + circle1.type*np.pi/2
        end_angle = end.theta + circle1.type*np.pi/2
        sector1 = [np.arctan2(p1_y - circle1.y, p1_x - circle1.x) - start_angle,circle1.radius]
        sector2 = [np.arctan2(p2_y - circle2.y, p2_x - circle2.x) - end_angle,circle2.radius]
        p1_angle = np.arctan2(p1_y - circle1.y, p1_x - circle1.x) - circle1.type * np.pi/2
        p2_angle = np.arctan2(p2_y - circle2.y, p2_x - circle2.x) - circle2.type * np.pi/2
        p1 = SETransform(p1_x, p1_y, p1_angle)
        p2 = SETransform(p2_x, p2_y, p2_angle)
        line_1,_ = self.approximate_straight(p1,p2)
        start_circle,start_circle_cost = self.approximate_circle(start,p1, circle1)
        end_circle,end_circle_cost= self.approximate_circle(p2,end, circle2)
        straight_cost = np.linalg.norm(np.array([p1_x, p1_y]) - np.array([p2_x, p2_y]))
        if circle1.type == 1:
            c1_type = 'RIGHT'
        else:
            c1_type = 'LEFT'
        if circle2.type == 1:
            c2_type = 'RIGHT'
        else:
            c2_type = 'LEFT'
        cost = start_circle_cost + end_circle_cost + straight_cost 
        res_dict = {'line': line_1, 'sector1': sector1, 'sector2': sector2, 'p1': p1, 'p2': p2,'start': start, 'goal': end,
                    'start_circle': start_circle, 'end_circle': end_circle,'cost': cost, 'c1_type': c1_type, 'c2_type': c2_type}
        duckie_path= []
        
        duckie_path.append(DuckieSegment(start, p1, sector1[1], c1_type ,start_circle,start_circle_cost,self.speed))
        duckie_path.append(DuckieSegment(p1, p2, 0, 'STRAIGHT', line_1,straight_cost,self.speed))
        duckie_path.append(DuckieSegment(p2,end, sector2[1], c2_type, end_circle,end_circle_cost,self.speed))
    
        return duckie_path
            
    def checker(self, circle1, circle2):
        #check if the circles intersect
        if np.linalg.norm(np.array(circle1.center) - np.array(circle2.center)) < circle1.radius + circle2.radius:
            return True
        else:
            return False
        



###################################################################################################################
###################################################################################################################
################################################ Map service utils ################################################
###################################################################################################################
###################################################################################################################
from quack_norris.srv import Map, MapResponse # type: ignore
from quack_norris.msg import Node, Corner # type: ignore
from geometry_msgs.msg import Pose2D

import matplotlib
matplotlib.use('Agg')

###################################################### Data #######################################################
TILE_DATA = {"TILE_SIZE": 0.585, "D_CENTER_TO_CENTERLINE": 0.2925, "CENTERLINE_WIDTH": 0.025, "LANE_WIDTH": 0.205} # [m]

###################################################### Main #######################################################
def initialize_map(start_node: DuckieNode, goal_node: DuckieNode) -> List[DuckieNode]:
    return _calculate_shortest_path(True, start_node, goal_node)

# def update_map(current_node: DuckieNode):
#     return _calculate_shortest_path(False, current_node, None)
def update_map(current_position: Union[SETransform, DuckieNode], faulty_node: DuckieNode) -> List[DuckieNode]:
    current_node = DuckieNode(current_position)
    return _calculate_shortest_path(False, current_node, faulty_node)

def _calculate_shortest_path(reset: bool, start_node: DuckieNode, goal_node: DuckieNode) -> Optional[List[DuckieNode]]:
    rospy.wait_for_service("map_service")
    try:
        map_service = rospy.ServiceProxy("map_service", Map)
        # rospy.loginfo(f"Calling map service with start node ({start_node.tag_id}) and goal node ({goal_node.tag_id})")
        response = map_service(reset=reset, start_node=duckienode_to_node(start_node), goal_node=duckienode_to_node(goal_node))
        return _respone_to_nodelist(response)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

################################################## Helper classes #################################################
class MapNode:
    def __init__(self, index_x: int, index_y: int, apriltag_id: int):
        self.center_index:  Tuple[int, int]     = (index_x, index_y)
        self.apriltag_id:   int                 = apriltag_id
        self.neighbors:     List[MapNode]       = []

    def __repr__(self):
        return f"MapNode(Center: {self.center_index}, ID: {self.apriltag_id})"

#################################################### Converters ###################################################
def _respone_to_nodelist(map) -> List[DuckieNode]:
    nodes = [node_to_duckienode(n) for n in map.nodes]
    if len(nodes) == 0:
        rospy.logwarn("No nodes in map")
    elif len(nodes) == 1:
        rospy.logwarn("Only one node in map")
    else:
        for ind, node in enumerate(nodes):
            if ind != 0:
                node.insert_parent(nodes[ind-1])
            if ind != len(nodes)-1:
                node.insert_next(nodes[ind+1])
    # rospy.loginfo(f"Nodes: {[n.tag_id for n in nodes]}")
    # rospy.loginfo(f"Corners: {[n.corner.type for n in nodes]}")
    return nodes

def corner_to_duckiecorner(corner: Corner) -> DuckieCorner:
    int_to_str = {-1: "LEFT", 0: "STRAIGHT", 1: "RIGHT"}
    return DuckieCorner(pose=SETransform(corner.pose.x, corner.pose.y, corner.pose.theta),
                        radius=corner.radius,
                        type=int_to_str[corner.type])

def duckiecorner_to_corner(duckiecorner: DuckieCorner) -> Corner:
    if duckiecorner is not None:    
        return Corner(pose=Pose2D(x=duckiecorner.pose.x, y=duckiecorner.pose.y, theta=duckiecorner.pose.theta),
                      radius=duckiecorner.radius,
                      type=duckiecorner.type)
    return None

def node_to_duckienode(node: Node) -> DuckieNode:
    duckienode = DuckieNode(pose=SETransform(node.pose.x, node.pose.y, node.pose.theta),
                            tag_id=node.apriltag_id)
    duckienode.insert_corner(corner_to_duckiecorner(node.corner))
    return duckienode
    

def duckienode_to_node(duckienode: DuckieNode) -> Node:
    if duckienode is not None:
        return Node(pose=Pose2D(x=duckienode.pose.x, y=duckienode.pose.y, theta=duckienode.pose.theta),
                    apriltag_id=duckienode.tag_id,
                    corner=duckiecorner_to_corner(duckienode.corner))
    return None

def node_to_mapnode(node: Node) -> MapNode:
    index_x, index_y = tile_pos_to_index((node.pose.x, node.pose.y))
    return MapNode(index_x, index_y, node.apriltag_id)

def mapnode_to_node(mapnode: MapNode, corner: Corner) -> Node:
    xabs, yabs = tile_index_to_pos(mapnode.center_index)
    return Node(pose=Pose2D(x=xabs, y=yabs, theta=0), # Theta?
                apriltag_id=mapnode.apriltag_id,
                corner=corner)

def tile_pos_to_index(pos: Tuple[float, float]) -> Tuple[int, int]:
    rospy.loginfo(f"Converting position {pos} to index ({round(pos[0] / TILE_DATA['TILE_SIZE'] - 0.5)}, {round(pos[1] / TILE_DATA['TILE_SIZE'] - 0.5)})")
    return (round(pos[0] / TILE_DATA["TILE_SIZE"] - 0.5), round(pos[1] / TILE_DATA["TILE_SIZE"] - 0.5))

def tile_index_to_pos(index: Tuple[int, int]) -> Tuple[float, float]:
    return ((index[0] + 0.5) * TILE_DATA["TILE_SIZE"], (index[1] + 0.5) * TILE_DATA["TILE_SIZE"])

################################################ Plotters, Printers ###############################################
def plot_solved_graph(all_nodes: List[MapNode], path: List[MapNode], fig_save_path: str, map_file: str) -> None:
    """
    Plots the graph and the A* path.

    Args:
        all_nodes (List[MapNode]): List of all nodes in the graph.
        path (List[MapNode]): Path found by A* search.
        fig_save_path (str): Path to save the figure (including the file name with extension).
    """
    plt.figure(figsize=(8, 8))

    # Extract map image
    if map_file == "circle_map":
        img = plt.imread("/home/duckie/repos/quack-norris/map_files/quack_small.png")
        img_height, img_width = img.shape[:2]
        factor = 0.0027
        extent = [0, img_width * factor, 0, img_height * factor]
    elif map_file == "oval_map":
        img = plt.imread("/home/duckie/repos/quack-norris/map_files/quack_middle_cropped.png")
        img = np.rot90(img, k=1)
        img_height, img_width = img.shape[:2]
        factor = 0.00454
        extent = [0, img_width * factor, 0, img_height * factor]
    else:
        img = plt.imread("/home/duckie/repos/quack-norris/map_files/quack_big_cropped.png")
        img = np.rot90(img, k=3)
        img_height, img_width = img.shape[:2]
        factor = 0.0072
        extent = [0, img_width * factor, 0, img_height * factor]

    # Display the map image
    plt.imshow(img, extent=extent)

    # Scale the node coordinates
    def scale_coordinates(node):
        x = (node.center_index[0] + 0.5) * TILE_DATA["TILE_SIZE"]
        y = (node.center_index[1] + 0.5) * TILE_DATA["TILE_SIZE"]
        return (x, y)
        # return (node.center_index[0], node.center_index[1])

    # Plot all edges (connections between neighbors) in light gray
    for node in all_nodes:
        for neighbor in node.neighbors:
            x_values = [scale_coordinates(node)[0], scale_coordinates(neighbor)[0]]
            y_values = [scale_coordinates(node)[1], scale_coordinates(neighbor)[1]]
            plt.plot(x_values, y_values, color='blue', linestyle='--', linewidth=2)

    # Highlight the A* path in red
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        x_values = [scale_coordinates(node)[0], scale_coordinates(next_node)[0]]
        y_values = [scale_coordinates(node)[1], scale_coordinates(next_node)[1]]
        plt.plot(x_values, y_values, color='lightgreen', linestyle='-', linewidth=5)

    # Plot all nodes as points
    for node in all_nodes:
        plt.scatter(scale_coordinates(node)[0], scale_coordinates(node)[1], color='blue', s=200)
        plt.text(scale_coordinates(node)[0], scale_coordinates(node)[1], str(node.apriltag_id),
                    fontsize=10, ha='center', va='center', color='white')

    # Highlight start and goal nodes
    if path:
        plt.scatter(scale_coordinates(path[0])[0], scale_coordinates(path[0])[1], color='green', s=300, label='Start')
        plt.scatter(scale_coordinates(path[-1])[0], scale_coordinates(path[-1])[1], color='orange', s=300, label='Goal')

    plt.title("A* Search Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_save_path)
    plt.close()

def plot_solved_graph_with_progress(
    all_nodes: List[MapNode],
    path: List[MapNode],
    fig_save_dir: str,
    map_file: str,
    step_size: float = 0.1
) -> None:
    """
    Plots the graph and the A* path progressively, with the start node moving along the path.
    Saves multiple images showing the progress.

    Args:
        all_nodes (List[MapNode]): List of all nodes in the graph.
        path (List[MapNode]): Path found by A* search.
        fig_save_dir (str): Directory to save the figures.
        map_file (str): Map file to use for the background image.
        step_size (float): Step size for splitting the path into finer points.
    """
    # Create the directory if it doesn't exist
    os.makedirs(fig_save_dir, exist_ok=True)

    # Extract map image
    if map_file == "circle_map":
        img = plt.imread("/home/duckie/repos/quack-norris/map_files/quack_small.png")
        img_height, img_width = img.shape[:2]
        factor = 0.0027
        extent = [0, img_width * factor, 0, img_height * factor]
    elif map_file == "oval_map":
        img = plt.imread("/home/duckie/repos/quack-norris/map_files/quack_middle_cropped.png")
        img = np.rot90(img, k=1)
        img_height, img_width = img.shape[:2]
        factor = 0.00454
        extent = [0, img_width * factor, 0, img_height * factor]
    else:
        img = plt.imread("/home/duckie/repos/quack-norris/map_files/quack_big_cropped.png")
        img = np.rot90(img, k=3)
        img_height, img_width = img.shape[:2]
        factor = 0.0072
        extent = [0, img_width * factor, 0, img_height * factor]

    # Function to scale the node coordinates
    def scale_coordinates(node):
        x = (node.center_index[0] + 0.5) * TILE_DATA["TILE_SIZE"]
        y = (node.center_index[1] + 0.5) * TILE_DATA["TILE_SIZE"]
        return (x, y)

    # Function to interpolate points along a segment
    def interpolate_points(p1, p2, step_size):
        """
        Generate intermediate points between p1 and p2 with a given step size.
        Args:
            p1: (x, y) coordinates of the starting point.
            p2: (x, y) coordinates of the ending point.
            step_size: Distance between successive points.
        Returns:
            List of intermediate (x, y) points.
        """
        x1, y1 = p1
        x2, y2 = p2
        distance = np.hypot(x2 - x1, y2 - y1)
        steps = int(distance / step_size)
        return [
            (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            for t in np.linspace(0, 1, steps + 1)
        ]

    # Generate all sub-points along the path
    sub_points = []
    for i in range(len(path) - 1):
        p1 = scale_coordinates(path[i])
        p2 = scale_coordinates(path[i + 1])
        sub_points.extend(interpolate_points(p1, p2, step_size))

    # Generate frames for the progressive path visualization
    for idx, current_position in enumerate(sub_points):
        plt.figure(figsize=(8, 8))

        # Display the map image
        plt.imshow(img, extent=extent)

        # Plot all edges (connections between neighbors) in light gray
        for node in all_nodes:
            for neighbor in node.neighbors:
                x_values = [scale_coordinates(node)[0], scale_coordinates(neighbor)[0]]
                y_values = [scale_coordinates(node)[1], scale_coordinates(neighbor)[1]]
                plt.plot(x_values, y_values, color='blue', linestyle='--', linewidth=2)

        # Plot the green path from the moving start to the end
        current_index = sub_points.index(current_position)
        for i in range(current_index, len(sub_points) - 1):
            x_values = [sub_points[i][0], sub_points[i + 1][0]]
            y_values = [sub_points[i][1], sub_points[i + 1][1]]
            plt.plot(x_values, y_values, color='lightgreen', linestyle='-', linewidth=5)

        # Plot the blue path from the static start to the current position
        for i in range(current_index):
            x_values = [sub_points[i][0], sub_points[i + 1][0]]
            y_values = [sub_points[i][1], sub_points[i + 1][1]]
            plt.plot(x_values, y_values, color='blue', linestyle='--', linewidth=2)

        # Plot all nodes as points
        for node in all_nodes:
            plt.scatter(scale_coordinates(node)[0], scale_coordinates(node)[1], color='blue', s=200)
            plt.text(scale_coordinates(node)[0], scale_coordinates(node)[1], str(node.apriltag_id),
                     fontsize=10, ha='center', va='center', color='white')

        # Highlight the moving start node
        plt.scatter(current_position[0], current_position[1], color='green', s=400, label='Duckiebot')

        # Highlight the goal node
        plt.scatter(scale_coordinates(path[-1])[0], scale_coordinates(path[-1])[1], color='orange', s=300, label='Goal')

        # Add title and labels
        plt.title(f"A* Search Path - Step {idx + 1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

        # Save the figure
        fig_save_path = os.path.join(fig_save_dir, f"path_step_{idx + 1:03d}.png")
        plt.savefig(fig_save_path)
        plt.close()

    # Save a final frame with the complete path
    plt.figure(figsize=(8, 8))
    plt.imshow(img, extent=extent)

    # Plot all edges
    for node in all_nodes:
        for neighbor in node.neighbors:
            x_values = [scale_coordinates(node)[0], scale_coordinates(neighbor)[0]]
            y_values = [scale_coordinates(node)[1], scale_coordinates(neighbor)[1]]
            plt.plot(x_values, y_values, color='blue', linestyle='--', linewidth=2)

    # Plot the entire green path
    for i in range(len(path) - 1):
        x_values = [scale_coordinates(path[i])[0], scale_coordinates(path[i + 1])[0]]
        y_values = [scale_coordinates(path[i])[1], scale_coordinates(path[i + 1])[1]]
        plt.plot(x_values, y_values, color='lightgreen', linestyle='-', linewidth=5)

    # Plot all nodes
    for node in all_nodes:
        plt.scatter(scale_coordinates(node)[0], scale_coordinates(node)[1], color='blue', s=200)
        plt.text(scale_coordinates(node)[0], scale_coordinates(node)[1], str(node.apriltag_id),
                 fontsize=10, ha='center', va='center', color='white')

    # Highlight start and goal nodes
    plt.scatter(scale_coordinates(path[0])[0], scale_coordinates(path[0])[1], color='green', s=300, label='Start')
    plt.scatter(scale_coordinates(path[-1])[0], scale_coordinates(path[-1])[1], color='orange', s=300, label='Goal')

    # Add title and labels
    plt.title("A* Search Path - Complete")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    final_save_path = os.path.join(fig_save_dir, "path_complete.png")
    plt.savefig(final_save_path)
    plt.close()

def print_path(path: List[DuckieNode]) -> None:
    path_str = "Calculated shortest path:\n"
    for index, node in enumerate(path):
        path_str += f"{node.tag_id}: ({node.pose.x}, {node.pose.y})"
        if index != len(path) - 1:
            path_str += f" -> "
    rospy.loginfo(path_str)

################################################## Miscellaneous ##################################################
def find_closest_points(current_point: Tuple[int, int], points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    points_array = np.array(points)
    x0, y0 = current_point
    result = []

    # Define direction conditions as tuples (dx, dy) and their corresponding comparison functions
    directions = [
        (-1, 0, lambda x: np.argmax(x)),  # -x: largest x < x0
        (1, 0, lambda x: np.argmin(x)),   # +x: smallest x > x0
        (0, -1, lambda y: np.argmax(y)),  # -y: largest y < y0
        (0, 1, lambda y: np.argmin(y))    # +y: smallest y > y0
    ]
    
    # Check all directions
    for dx, dy, arg_fn in directions:
        mask = ((points_array[:, 0] < x0 if dx == -1 else points_array[:, 0] > x0 if dx == 1 else points_array[:, 0] == x0) &
                (points_array[:, 1] < y0 if dy == -1 else points_array[:, 1] > y0 if dy == 1 else points_array[:, 1] == y0))
        
        if np.any(mask):  # If there are points in this direction
            filtered = points_array[mask]
            idx = arg_fn(filtered[:, 0] if dx else filtered[:, 1])
            result.append(tuple(filtered[idx]))
    
    return result

def fill_path_corners(path: List[MapNode]) -> List[Node]:
    """
    Fill in the corners of each node in the path. Used in the local planner later.

    Args:
        path (List[MapNode]): Path to fill with corners

    Returns:
        List[Node]: Path with corners
    """
    filled_path = [mapnode_to_node(path[0], Corner())]
    for node in path[1:-1]:
        prev_node = path[path.index(node) - 1]
        next_node = path[path.index(node) + 1]

        # Find the corner between the previous and next node
        prev_node_pos = np.array(prev_node.center_index)
        next_node_pos = np.array(next_node.center_index)
        curr_node_pos = np.array(node.center_index)
        if np.array_equal(prev_node_pos, curr_node_pos) or np.array_equal(curr_node_pos, next_node_pos):
            raise ValueError("Two nodes in the path have the same position - Should not happen. Exiting...")
        vec_to = (curr_node_pos - prev_node_pos) / np.linalg.norm(curr_node_pos - prev_node_pos)
        vec_from = (next_node_pos - curr_node_pos) / np.linalg.norm(next_node_pos - curr_node_pos)
        dir = int(np.cross(vec_from, vec_to))
        
        corner_pos = np.array(tile_index_to_pos(curr_node_pos)) + (vec_from - vec_to) * TILE_DATA["TILE_SIZE"] / 2
        # corner_theta = -dir * np.pi / 4 # +/-?
        angle_vec_from = np.arctan2(vec_from[1], vec_from[0])
        corner_theta = angle_vec_from + dir * np.pi / 2
        corner_radius = abs(dir)*(3/4)*TILE_DATA["TILE_SIZE"] # (TILE_DATA["D_CENTER_TO_CENTERLINE"] + TILE_DATA["CENTERLINE_WIDTH"] / 2 + TILE_DATA["LANE_WIDTH"] / 2)
        corner_type = dir # -1: LEFT, 0: STRAIGHT, 1: RIGHT
        corner = Corner(pose=Pose2D(x=corner_pos[0], y=corner_pos[1], theta=corner_theta),
                        radius=corner_radius,
                        type=corner_type)
        filled_path.append(mapnode_to_node(node, corner))
        
    filled_path.append(mapnode_to_node(path[-1], Corner()))
    return filled_path