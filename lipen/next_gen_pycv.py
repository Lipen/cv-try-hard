import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial import distance as dist
from imutils import perspective
from sensor_msgs.msg import Image
from cvision.msg import Object
from cvision.msg import ListObjects
from cvision.msg import Orientation


class NextGenRecognize:

    def __init__(self, source, is_ros_msg=False):
        self.is_ros_msg = is_ros_msg
        _, _, self.diag_mm = self.getWHImage(420, dfov=68)

        self.sub_camera = rospy.Subscriber(source, Image, self.callback_camera)
        self.sub_orientation = rospy.Subscriber('/orientation', Orientation, self.callback_orientation, queue_size=1)

        self.pub_main = rospy.Publisher('list_objects', ListObjects, queue_size=1)
        self.pub_view_main = rospy.Publisher('see_main', Image, queue_size=1)

    def callback_camera(self, data):
        if self.is_ros_msg:
            cv_image = self.getCVImage(data)
        else:
            cv_image = data

        objects_list, magic = self.get_objects_list(cv_image, True)

        msg = ListObjects()
        msg = objects_list
        rospy.loginfo('Send {} objects'.format(len(msg)))
        self.pub_main.publish(msg)
        msg_image = self.getMsgImage(magic)
        self.pub_view_main.publish(msg_image)

    def callback_orientation(self, data):
        l = data.length
        self.imshape_mm = self.getWHImage(l, dfov=68)

    def getWHImage(self, l, hfov=58, vfov=45, dfov=70):
        hfov_rad = math.radians(hfov)
        vfov_rad = math.radians(vfov)
        dfov_rad = math.radians(dfov)
        width = 2 * l * math.tan(hfov_rad / 2)
        height = 2 * l * math.tan(vfov_rad / 2)
        diag = 2 * l * math.tan(dfov_rad / 2)
        return width, height, diag

    def getCVImage(self, data):
        bridge = CvBridge()
        cv_image = None
        try:
            cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
            h, w = cv_image[:2]
            diag_px = math.hypot(w, h)
            self.imshape_px = (w, h, diag_px)
        except CvBridgeError as e:
            rospy.loginfo('Conversion failed: {}'.format(e.message))
        return cv_image

    def midpoint(self, ptA, ptB):
        return (ptA + ptB) / 2

    def put_text(self, image, pos, args, fmt='{:.0f}mm', fontScale=0.5, color1=(0, 0, 0), color2=(0, 255, 0), thickness1=1, thickness2=2):
        pos = (int(pos[0]), int(pos[1]))
        cv2.putText(image, fmt.format(*args), pos, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontScale, color=color1, thickness=thickness1)
        cv2.putText(image, fmt.format(*args), pos, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontScale, color=color2, thickness=thickness2)

    def get_objects_list(self, image):
        objects_list = []

        base_contour = np.array([[[43, 17]], [[31, 24]], [[27, 33]], [[85, 183]], [[75, 194]], [[84, 217]], [[129, 206]], [[139, 197]], [[136, 180]], [[132, 174]], [[123, 175]], [[118, 169]], [[61, 20]], [[56, 16]]])

        blur = 3
        blurred = cv2.medianBlur(image, blur)
        sigma = .33
        v = np.median(image)
        canny_low = int(max(0, (1 - sigma) * v))
        canny_high = int(min(255, (1 + sigma) * v))
        edged = cv2.Canny(blurred, canny_low, canny_high)
        edged = cv2.dilate(edged, None, iterations=3)
        edged = cv2.erode(edged, None, iterations=2)

        _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        y0, x0 = image.shape[:2]
        diag0 = math.hypot(x0, y0)
        # l = 420
        # dfov = 68
        # diag_mm = getImageDiag(l, dfov)
        _, _, diag_mm = self.imshape_mm
        ratio = diag_mm / diag0

        best = None
        contours_to_draw = []  # (ret, contour)

        for contour in contours:
            if cv2.contourArea(contour) < 700:
                continue

            ret = cv2.matchShapes(contour, base_contour, 1, 0)
            if best is None or ret < best[0]:
                best = (ret, contour)

            contours_to_draw.append((ret, contour))

        for ret, contour in contours_to_draw:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)
            dA = math.hypot(tltrX - blbrX, tltrY - blbrY)
            dB = math.hypot(tlblX - trbrX, tlblY - trbrY)
            dD = math.hypot(dA, dB)

            xoc, yoc = self.midpoint(tl, br)
            xcc = xoc - x0 / 2
            ycc = y0 / 2 - yoc

            alpha = math.atan2(dB, dA)
            dimD = dD * ratio
            dimA = dimD * math.sin(alpha)
            dimB = dimD * math.cos(alpha)

            for x, y in box:
                cv2.circle(image, (x, y), 5, (255, 0, 255), 2)

            cv2.drawContours(image, [box.astype('int')], -1, (0, 255, 0), 2)
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

            self.put_text(image, (tltrX - 15, tltrY - 10), (dimA,), color2=(255, 100, 0), fontScale=0.5, thickness1=2, thickness2=4)  # width
            self.put_text(image, (trbrX + 10, trbrY), (dimB,), color2=(255, 100, 0), fontScale=0.5, thickness1=2, thickness2=4)  # width
            self.put_text(image, (xoc, yoc), (xcc * ratio, ycc * ratio), fmt='({:.0f},{:.0f})', color2=(255, 200, 0), fontScale=0.7, thickness1=2, thickness2=4)  # center coordinates
            # #######

            M = cv2.moments(contour)
            text_pos = (int(M['m10'] / M['m00']),
                        int(M['m01'] / M['m00'])) if M['m00'] else (0, 0)

            mask = np.zeros(edged.shape, np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            color = cv2.mean(image, mask)

            if contour is best[1]:
                text = 'Screw'
                cv2.drawContours(image, [contour], -1, color, 2)
                cv2.putText(image, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(0, 0, 0), thickness=4)
                cv2.putText(image, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(224, 255, 224), thickness=2)
            else:
                text = '{:.3f}'.format(ret)
                cv2.drawContours(image, [contour], -1, color, 2)
                cv2.putText(image, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(0, 0, 0), thickness=4)
                cv2.putText(image, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(255, 255, 255), thickness=2)

        return objects_list
