import numpy as np
import cv2
import math
from scipy.spatial import distance as dist
from imutils import perspective
import copy

SZ = 600

def detect(c):
	# initialize the shape name and approximate the contour
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)

	# if the shape is a triangle, it will have 3 vertices
	if len(approx) == 3:
		shape = "triangle"

	# if the shape has 4 vertices, it is either a square or
	# a rectangle
	elif len(approx) == 4:
		# compute the bounding box of the contour and use the
		# bounding box to compute the aspect ratio
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w / float(h)

		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

	# if the shape is a pentagon, it will have 5 vertices
	elif len(approx) == 5:
		shape = "pentagon"

	# otherwise, we assume the shape is a circle
	else:
		shape = "circle"

	# return the name of the shape
	return shape

def nothing(x):
    pass

def midpoint(ptA, ptB):
		return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		# no deskewing needed. 
		return img.copy()
	# Calculate skew based on central momemts. 
	skew = m['mu11']/m['mu02']
	# Calculate affine transform to correct skewness. 
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	# Apply affine transform
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

class Object:
	"""contains parameters of the object"""
	pass


cam = cv2.VideoCapture(0)

_, frame = cam.read()

imshape_px = [0, 0, 0]
imshape_mm = [0, 0, 0]
diag_px = math.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)
imshape_px[0], imshape_px[1], imshape_px[2] = (frame.shape[0], frame.shape[1], diag_px)
imshape_mm = (54.5, 42.3, 66.17)
detail = True;

cv2.namedWindow('bars',cv2.WINDOW_AUTOSIZE) 
# create trackbars
cv2.createTrackbar('threshold_min', 'bars', 0, 255, nothing)
cv2.createTrackbar('invert', 'bars', 0, 1, nothing)
cv2.createTrackbar('blur', 'bars', 0, 30, nothing)

print "press q for quit"

while(True):
	# Capture frame-by-frame
	_, frame = cam.read()

	# Our operations on the frame come here
	# binary = cv2.cvtColor(frame, cv2.)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#TRUING
	threshold_min = cv2.getTrackbarPos('threshold_min', 'bars')
	invert = cv2.getTrackbarPos('invert', 'bars')
	blur = cv2.getTrackbarPos('blur', 'bars')
	blur = blur * 2 + 1
	# threshold_min = 100
	gray = cv2.medianBlur(gray, blur)
	ret,edged = cv2.threshold(gray, threshold_min, 255, cv2.THRESH_BINARY)
	if invert == 1:
		edged = cv2.bitwise_not(edged)
	#END TRUING

	edged_safe = copy.deepcopy(edged)
	_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		o = Object()
		if cv2.contourArea(contour) < 500:
			continue
		box = cv2.minAreaRect(contour)
		box = cv2.boxPoints(box)
		box = np.array(box, dtype="int")    
		box = perspective.order_points(box)
		# coordinates of corners
		(tl, tr, br, bl) = box
		# middles of sides
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		# center of frame
		x0 = imshape_px[1] / 2
		y0 = imshape_px[0] / 2
		# center of object in center frame coord. system
		# coord. system is 5th joint of manipulator
		(xoc, yoc)= midpoint(tl, br)
		xcc = x0 - yoc
		ycc = y0 - xoc
		# [mm]
		ratio = imshape_mm[2] / imshape_px[2]
		dD = math.sqrt(dA ** 2 + dB ** 2)
		dimD = dD * ratio
		#rospy.loginfo("O_diag_mm = %s", dimD)
		alpha = math.atan2(dB, dA)
		dimA = dimD * math.sin(alpha)
		dimB = dimD * math.cos(alpha)
		# ***
		# TODO detecting and analezing objects
		shape = detect(contour)

		o.shape = 'undefined'
		o.dimensions = (dimA, dimB, 1)
		o.coordinates_center_frame = (xcc*ratio*0.001, ycc*ratio*0.001, 0)
		o.coordinates_frame = (tltrX*ratio*0.001, tlblY*ratio*0.001, 0)
		#info = "Height:%d\tWidth:%d\n" % (dimA, dimB)
		#rospy.loginfo(info)
		if detail:
			# ------------------
			# draw on source frame
			cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
			# draw corner points
			for (x, y) in box:
				cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
				"""
				# draw the midpoints on the image
				cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
				cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
				cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
				cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
				cv2.circle(image, (int(x0), int(y0)), 5, (0, 255, 0), -1)
				
				# draw lines between the midpoints
				cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
				cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

				cv2.putText(image, "{:.1f}px;{:.1f}px".format(self.imshape_px[0], self.imshape_px[1]),
							(int(0 + 15), int(0 + 20)),
							cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
				cv2.putText(image, "{:.1f}mm;{:.1f}mm".format(self.imshape_mm[0], self.imshape_mm[1]),
							((0 + 15), int(0 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
				"""
				cv2.putText(frame, shape, (int(trbrX - 50), int(trbrY + 20)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
				# draw the object sizes on the image
				# cv2.putText(frame, "{:.1f}px;{:.1f}mm".format(dA, dimA), (int(tltrX - 15), int(tltrY - 10)),
				# 			cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
				# cv2.putText(frame, "{:.1f}px;{:.1f}mm".format(dB, dimB), (int(trbrX + 10), int(trbrY)),
				# 			cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)



	# Display the resulting frame
	cv2.imshow('frame',frame)
	cv2.imshow('edged',edged_safe)
	# test = deskew(gray)
	# cv2.imshow('test',test)
	# height, width = test.shape
	# print height, width
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()