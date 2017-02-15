import imutils
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);

while(True):
	ret, image = cap.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
			param1=80,
            param2=100,
            minRadius=15,
            maxRadius=300)
	if circles is not None:
		circles = np.round(circles[0, :]).astype("int") 
		for (x, y, r) in circles:
			cv2.circle(image, (x, y), r, (0, 255, 0), 4)
			# cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	cv2.imshow("circles",image)
	if cv2.waitKey(1) & 0xFF == 27:
		break
cv2.destroyAllWindows()
cap.release()