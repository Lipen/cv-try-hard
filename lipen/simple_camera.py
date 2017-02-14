import cv2

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise ValueError('Failed to open a capture object.')

while True:
    _, image = cam.read()
    cv2.imshow("MAIN", image)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cam.release()
