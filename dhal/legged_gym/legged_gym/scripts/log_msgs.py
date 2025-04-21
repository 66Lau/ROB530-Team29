
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
from datetime import datetime

class PoseLogger:
    def __init__(self):
        self.pose_data = []
        self.sub_topic = "/GT_pose"

        rospy.Subscriber(self.sub_topic, PoseWithCovarianceStamped, self.pose_callback)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = f"gt_pose.jsonl"

        rospy.on_shutdown(self.save_data)

    def pose_to_dict(self, msg: PoseWithCovarianceStamped):
        return {
            "stamp": msg.header.stamp.to_sec(),
            "frame_id": msg.header.frame_id,
            "position": {
                "x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y,
                "z": msg.pose.pose.position.z,
            },
            "orientation": {
                "x": msg.pose.pose.orientation.x,
                "y": msg.pose.pose.orientation.y,
                "z": msg.pose.pose.orientation.z,
                "w": msg.pose.pose.orientation.w,
            },
            "covariance": msg.pose.covariance
        }

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.pose_data.append(self.pose_to_dict(msg))

    def save_data(self):
        rospy.loginfo("Saving pose data to %s", self.save_path)
        with open(self.save_path, "w") as f:
            for pose_entry in self.pose_data:
                f.write(json.dumps(pose_entry) + "\n")
        rospy.loginfo("Finished saving %d pose entries.", len(self.pose_data))        


if __name__ == '__main__':
    rospy.init_node("pose_logger", anonymous=True)
    logger = PoseLogger()
    rospy.loginfo("Pose logger started. Listening to " + logger.sub_topic)
    rospy.spin()