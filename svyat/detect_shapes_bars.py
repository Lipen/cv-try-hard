from shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);

def nothing(x):
	pass

cv2.namedWindow('bars',cv2.WINDOW_AUTOSIZE) 
# create trackbars for color change
cv2.createTrackbar('threshold_min', 'bars', 0, 255, nothing)
cv2.createTrackbar('invert', 'bars', 0, 1, nothing)
cv2.createTrackbar('blurtype', 'bars', 0, 3, nothing)
cv2.createTrackbar('blur', 'bars', 0, 30, nothing)
cv2.createTrackbar('size', 'bars', 480, 1280, nothing)
# cv2.createTrackbar('mirror', 'bars', 0, 1, nothing)

while(True):
	ret, image = cap.read()
	invert = 0
	#taking definitions of bars 
	threshold_min = cv2.getTrackbarPos('threshold_min', 'bars')
	invert = cv2.getTrackbarPos('invert', 'bars')
	blurtype = cv2.getTrackbarPos('blurtype', 'bars')
	blur = cv2.getTrackbarPos('blur', 'bars')
	blur = blur * 2 + 1
	size = cv2.getTrackbarPos('size', 'bars')
	if size == 0:
		size = 2
	image = imutils.resize(image, width=size)
	
	ratio = 1

	if blurtype == 0:
		blurred = cv2.GaussianBlur(image, (blur, blur), 0)
	elif blurtype == 1:
		blurred = cv2.medianBlur(image, blur)
	elif blurtype == 2:
		blurred = cv2.blur(image,(blur, blur))
	elif blurtype == 3:
		blurred = cv2.bilateralFilter(image, 9, blur,blur)

	gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

	ret,thresh = cv2.threshold(gray, threshold_min, 255, cv2.THRESH_BINARY)

	if invert == 1:
		thresh = cv2.bitwise_not(thresh)

	contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if imutils.is_cv2() else contours[1]
	sd = ShapeDetector()

	# loop over the contours
	for contour in contours:
		M = cv2.moments(contour)
		cX=50
		cY=50
		if M["m00"]!=0:
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)

		shape = sd.detect(contour)

		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		contour = contour.astype("float")
		contour *= ratio
		contour = contour.astype("int")
		cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
		cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0,0,0), 4)
		cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (255, 255, 255), 2)		
	cv2.imshow("Image", image)
	cv2.imshow("Blurred", blurred)
	cv2.imshow("Treshholded", thresh)
	if cv2.waitKey(1) & 0xFF == 27:
		break
cv2.destroyAllWindows()
cap.release()