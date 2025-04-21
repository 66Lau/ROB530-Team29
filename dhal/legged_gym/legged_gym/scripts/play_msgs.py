
from time import time
from warnings import WarningMessage
import numpy as np
import os
import sys
import json

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict
import random

from scipy.spatial.transform import Rotation as R

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from sensor_msgs.msg import JointState
# If you want to subscribe to IMU or contact arrays:
from sensor_msgs.msg import Imu

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../drift/ROS/drift/build'))
from _Contact import Contact
from _ContactArray import ContactArray
from geometry_msgs.msg import TwistWithCovarianceStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped

# ros
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu  
from nav_msgs.msg import Path

def publish_sim_data(file_path):
    # ROS publisher
    pub_joint = rospy.Publisher('/JointState', JointState, queue_size=10)
    pub_imu = rospy.Publisher('/Imu', Imu, queue_size=10)
    pub_contact = rospy.Publisher('/Contacts', ContactArray, queue_size=10)
    pub_vel = rospy.Publisher('/Twist', TwistWithCovarianceStamped, queue_size=10)
    pub_vel_rough = rospy.Publisher('/Twist_rough', TwistWithCovarianceStamped, queue_size=10)
    pub_pose = rospy.Publisher('/GT_pose', PoseWithCovarianceStamped, queue_size=10)

    rospy.init_node('sim_data_republisher', anonymous=True)
    rate = rospy.Rate(500)  # 500 Hz publish frequency

    # read log file
    sim_data = []
    with open(file_path, 'r') as f:
        for line in f:
            sim_data.append(json.loads(line))
    rospy.loginfo("Loaded %d simulation steps", len(sim_data))

    # read data in sequence and publish
    for entry in sim_data:
        # construct and publish JointState message
        js = JointState()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = entry["joint_state"]["header"].get("frame_id", "")
        js.header = header
        js.name = entry["joint_state"]["name"]
        js.position = entry["joint_state"]["position"]
        js.velocity = entry["joint_state"]["velocity"]
        if "effort" in entry["joint_state"]:
            js.effort = entry["joint_state"]["effort"]
        pub_joint.publish(js)

        # Construct and publish IMU message
        imu = Imu()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = entry["imu"]["header"]["frame_id"]
        imu.header = header
        imu.orientation.x = entry["imu"]["orientation"]["x"]
        imu.orientation.y = entry["imu"]["orientation"]["y"]
        imu.orientation.z = entry["imu"]["orientation"]["z"]
        imu.orientation.w = entry["imu"]["orientation"]["w"]
        imu.angular_velocity.x = entry["imu"]["angular_velocity"]["x"]
        imu.angular_velocity.y = entry["imu"]["angular_velocity"]["y"]
        imu.angular_velocity.z = entry["imu"]["angular_velocity"]["z"]
        imu.linear_acceleration.x = entry["imu"]["linear_acceleration"]["x"]
        imu.linear_acceleration.y = entry["imu"]["linear_acceleration"]["y"]
        imu.linear_acceleration.z = entry["imu"]["linear_acceleration"]["z"]
        pub_imu.publish(imu)


        if not entry["twist"]["header"]["frame_id"]  == "None":
            twist = TwistWithCovarianceStamped()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = entry["twist"]["header"]["frame_id"]
            twist.header = header

            # Linear velocity
            twist.twist.twist.linear.x = entry["twist"]["linear"]["x"]
            twist.twist.twist.linear.y = entry["twist"]["linear"]["y"]
            twist.twist.twist.linear.z = entry["twist"]["linear"]["z"]

            # Angular velocity
            twist.twist.twist.angular.x = entry["twist"]["angular"]["x"]
            twist.twist.twist.angular.y = entry["twist"]["angular"]["y"]
            twist.twist.twist.angular.z = entry["twist"]["angular"]["z"]
            twist.twist.covariance = entry["twist"]["covariance"]
            pub_vel.publish(twist)

        twist = TwistWithCovarianceStamped()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = entry["twist_rough"]["header"]["frame_id"]
        twist.header = header

        # Linear velocity
        twist.twist.twist.linear.x = entry["twist_rough"]["linear"]["x"]
        twist.twist.twist.linear.y = entry["twist_rough"]["linear"]["y"]
        twist.twist.twist.linear.z = entry["twist_rough"]["linear"]["z"]

        # Angular velocity
        twist.twist.twist.angular.x = entry["twist_rough"]["angular"]["x"]
        twist.twist.twist.angular.y = entry["twist_rough"]["angular"]["y"]
        twist.twist.twist.angular.z = entry["twist_rough"]["angular"]["z"]
        twist.twist.covariance = entry["twist_rough"]["covariance"]
        pub_vel_rough.publish(twist)


        pose = PoseWithCovarianceStamped()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = entry["pose"]["header"]["frame_id"]
        pose.header = header

        # Position
        pose.pose.pose.position.x = entry["pose"]["position"]["x"]
        pose.pose.pose.position.y = entry["pose"]["position"]["y"]
        pose.pose.pose.position.z = entry["pose"]["position"]["z"]

        # Orientation
        pose.pose.pose.orientation.x = entry["pose"]["orientation"]["x"]
        pose.pose.pose.orientation.y = entry["pose"]["orientation"]["y"]
        pose.pose.pose.orientation.z = entry["pose"]["orientation"]["z"]
        pose.pose.pose.orientation.w = entry["pose"]["orientation"]["w"]

        # Covariance
        pose.pose.covariance = entry["pose"]["covariance"]

        pub_pose.publish(pose)

        # Construct and publish ContactArray message
        contact_array = ContactArray()
        header = Header()
        header.stamp = rospy.Time.now()
        contact_array.header = header
        for c_dict in entry["contact"]["contacts"]:
            contact = Contact()
            contact.id = c_dict["id"]
            contact.indicator = c_dict["indicator"]
            contact_array.contacts.append(contact)
        pub_contact.publish(contact_array)

        # Navigation path
        if "path" in entry:

            path_msg = Path()
            path_msg.header = Header()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = entry["path"]["header"]["frame_id"]

            for p in entry["path"]["poses"]:
                pose_stamped = PoseStamped()
                pose_stamped.header = Header()
                pose_stamped.header.stamp = rospy.Time.from_sec(p["stamp"])
                pose_stamped.header.frame_id = path_msg.header.frame_id

                pose_stamped.pose.position.x = p["position"]["x"]
                pose_stamped.pose.position.y = p["position"]["y"]
                pose_stamped.pose.position.z = p["position"]["z"]

                pose_stamped.pose.orientation.x = p["orientation"]["x"]
                pose_stamped.pose.orientation.y = p["orientation"]["y"]
                pose_stamped.pose.orientation.z = p["orientation"]["z"]
                pose_stamped.pose.orientation.w = p["orientation"]["w"]

                path_msg.poses.append(pose_stamped)

            if not hasattr(publish_sim_data, "_path_pub"):
                publish_sim_data._path_pub = rospy.Publisher("/GT_path", Path, queue_size=10)

            publish_sim_data._path_pub.publish(path_msg)

        rate.sleep()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: rosrun your_package sim_data_republisher.py <data_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    try:
        publish_sim_data(file_path)
    except rospy.ROSInterruptException:
        pass
