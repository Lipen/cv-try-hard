import cv2


def init():
    global cam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError('Failed to open a capture object.')

    cv2.namedWindow('MAIN')

    global pattern_size
    pattern_size = (7, 5)


def main():
    init()

    while True:
        _, image = cam.read()
        image = cv2.flip(image, flipCode=1)

        found, corners = cv2.findChessboardCorners(image, pattern_size)

        cv2.drawChessboardCorners(image, pattern_size, corners, found)

        cv2.imshow('MAIN', image)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cam.release()
