#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely.geometry as sg
import shapely.ops as so
import shapely.affinity as sa
from typing import List, Tuple


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

class DuckieSegment:
    def __init__(self, start: SETransform, end: SETransform, radius: float, type: str, path: sg.LineString, cost: float):   
        self.start = start
        self.end = end
        self.radius = radius
        self.type = type
        self.sector = end.theta - start.theta
        self.speed = 0.1
        self.shapely_path= path
        self.cost = None
        self.dt = 0.1
        self.cost = cost
    def get_path_array(self):
        if self.type == 'STRAIGHT':
            dist = np.linalg.norm(np.array([self.start.x, self.start.y]) - np.array([self.end.x, self.end.y]))
            n_steps = int(dist/self.speed/self.dt)
            x = np.linspace(self.start.x, self.end.x, n_steps)
            y = np.linspace(self.start.y, self.end.y, n_steps)
            theta = np.ones(n_steps)*self.start.theta
            return np.vstack([x, y, theta]).T
        elif self.type == 'LEFT':
            start = self.start.theta
            end = self.end.theta
            if start < 0:
                start+= 2*np.pi
            if end < 0:
                end += 2*np.pi
            
            angle = np.linspace(start,end, int(np.abs(self.sector)/self.speed/self.dt))
            angle = np.mod(angle, 2*np.pi)  # Normalize angles to be within [0, 2*pi]
            x = self.start.x + self.radius*np.cos(start + np.pi/2) + self.radius*np.cos(angle - np.pi/2)
            y = self.start.y + self.radius*np.sin(start + np.pi/2)+ self.radius*np.sin(angle - np.pi/2)
            theta = angle
            return np.vstack([x, y, theta]).T
        elif self.type == 'RIGHT':
            start = self.start.theta
            end = self.end.theta
            if start > 0:
                start-= 2*np.pi
            if end > 0:
                end -= 2*np.pi
            angle = np.linspace(start,end, int(np.abs(self.sector)/self.speed/self.dt))
            angle = np.mod(angle, 2*np.pi)
            x = self.start.x + self.radius*np.cos(start - np.pi/2) + self.radius*np.cos(angle + np.pi/2)
            y = self.start.y + self.radius*np.sin(start - np.pi/2)+ self.radius*np.sin(angle + np.pi/2)
            theta = angle
            return np.vstack([x, y, theta]).T

class DuckieCorner:
    def __init__(self, pose:SETransform, radius: float, type: str):
        self.pose = pose
        self.radius = radius
        self.type = type
        self.shapely_obs = self.get_obs() 
        self.placement = self.get_poses_dubins()
    def get_poses_dubins(self):
        
        placement_x = self.pose.x + self.radius*np.cos(self.pose.theta - np.pi/2)
        placement_y = self.pose.y + self.radius*np.sin(self.pose.theta - np.pi/2)
        placement_theta = self.pose.theta
        return SETransform(placement_x, placement_y, placement_theta)
        
    def get_obs(self):
        return sg.Point((self.pose.x,self.pose.y)).buffer(self.radius-0.1)
    
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
        n_seg = 36
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
        
        n_seg = 36
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
        
        duckie_path.append(DuckieSegment(start, p1, sector1[1], c1_type ,start_circle,start_circle_cost))
        duckie_path.append(DuckieSegment(p1, p2, 0, 'STRAIGHT', line_1,straight_cost))
        duckie_path.append(DuckieSegment(p2,end, sector2[1], c2_type, end_circle,end_circle_cost))
    
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
        
        duckie_path.append(DuckieSegment(start, p1, sector1[1], c1_type ,start_circle,start_circle_cost))
        duckie_path.append(DuckieSegment(p1, p2, 0, 'STRAIGHT', line_1,straight_cost))
        duckie_path.append(DuckieSegment(p2,end, sector2[1], c2_type, end_circle,end_circle_cost))
    
        return duckie_path
            
    def checker(self, circle1, circle2):
        #check if the circles intersect
        if np.linalg.norm(np.array(circle1.center) - np.array(circle2.center)) < circle1.radius + circle2.radius:
            return True
        else:
            return False
        
