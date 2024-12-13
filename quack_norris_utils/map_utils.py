import rospy
from test_service.srv import Map
from test_service.msg import Node, Corner
from geometry_msgs.msg import Pose2D

from typing import List

class SETransform:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

class DuckieCorner:
    def __init__(self, pose: SETransform, radius: float, type: str):
        self.pose = pose
        self.radius = radius
        self.type = type

class DuckieNode:
    def __init__(self,pose: SETransform, parent = None, next = None, tag_id = None):
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

# # Node message
# Pose2D  pose
# int32   apriltag_id
# Corner  corner

# # Corner message
# Pose2D  pose
# float64 radius  # 'None' if straight
# int32   type    # {-1, 0, 1} = {'LEFT', 'STRAIGHT', 'RIGHT'}

def corner_to_duckiecorner(corner: Corner) -> DuckieCorner:
    return DuckieCorner(pose=SETransform(corner.pose.x, corner.pose.y, corner.pose.theta),
                        radius=corner.radius,
                        type=corner.type)

def duckiecorner_to_corner(duckiecorner: DuckieCorner) -> Corner:
    return Corner(pose=Pose2D(x=duckiecorner.pose.x, y=duckiecorner.pose.y, theta=duckiecorner.pose.theta),
                  radius=duckiecorner.radius,
                  type=duckiecorner.type)

def node_to_duckienode(node: Node) -> DuckieNode:
    return DuckieNode(pose=SETransform(node.pose.x, node.pose.y, node.pose.theta),
                      tag_id=node.apriltag_id)

def duckienode_to_node(duckienode: DuckieNode) -> Node:
    return Node(pose=Pose2D(x=duckienode.pose.x, y=duckienode.pose.y, theta=duckienode.pose.theta),
                apriltag_id=duckienode.tag_id,
                corner=duckiecorner_to_corner(duckienode.corner))

def respone_to_nodelist(map) -> List[DuckieNode]:    
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

    return nodes

def call_map_service(start: bool, node: DuckieNode = None):
    rospy.wait_for_service("map_service")
    try:
        map_service = rospy.ServiceProxy("map_service", Map)
        if start:
            response = map_service(start=start, node=node)
        else:
            response = map_service(start=start, node=duckienode_to_node(node))
        # rospy.loginfo(f"Service response: {response.message}")
        nodes = respone_to_nodelist(response)
        # for n in nodes:
        #     rospy.loginfo(f"Node in map:\nAprilTag ID {n.tag_id},\nPose: ({n.pose.x}, {n.pose.y}, {n.pose.theta})\n")
        return nodes
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def initialize_map():
    rospy.loginfo("Initializing the map...")
    call_map_service(start=True)

def update_map(node: DuckieNode):
    rospy.logwarn("Recalculating optimal path...")
    call_map_service(start=False, node=node)