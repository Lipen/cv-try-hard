#!/usr/bin/env python
import roslib
import rospy
from roslib.rospack import rospack_depends

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cvision.msg import Object
from cvision.msg import ListObjects
from cvision.msg import Orientation
from pycv_ng import Recognize

def talker():
	
	rospy.init_node('cv_recognizer', anonymous=False)
	bridge = CvBridge()
	
	cam = cv2.VideoCapture(1)
	
	pup_opencv_cam = rospy.Publisher('see_main_webcam', Image, queue_size=1)

	rospy.loginfo('/see_main_webcam')
	Recognize('/see_main_webcam', True)

	_, frame = cam.read()

	# rospy.loginfo("image properties")
	# rospy.loginfo(frame.shape)

	while not rospy.is_shutdown():
		_, frame = cam.read()
		msg_cv = bridge.cv2_to_imgmsg(frame, "bgr8")
		pup_opencv_cam.publish(msg_cv)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass