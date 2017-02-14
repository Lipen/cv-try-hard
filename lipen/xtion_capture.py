import cv2

# for i in range(1, 1000):
#     cam = cv2.VideoCapture(i)
#     if cam.isOpened():
#         print("Found camera @ {}".format(i))
# raise ValueError("Search done.")

cam = cv2.VideoCapture(cv2.CAP_OPENNI_ASUS)
# cam = cv2.VideoCapture(-1)
if cam.isOpened():
    print('Successfully opened a OpenNi capture object.')
else:
    raise ValueError('Failed to open a OpenNi capture object.')
cam.set(cv2.CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv2.CAP_OPENNI_VGA_30HZ)

while True:
    if not cam.grab():
        print('Cannot grab.')
        break
    else:
        ok1, depth_map = cam.retrieve(0, cv2.CAP_OPENNI_DEPTH_MAP)
        if ok1:
            cv2.imshow("DEPTH MAP", depth_map)

        ok2, bgr_image = cam.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)
        if ok2:
            cv2.imshow("BGR IMAGE", bgr_image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cam.release()
