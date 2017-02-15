import cv2
import numpy as np


def do_nothing(_):
    pass


def detect(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n = len(approx)

    if n == 3:
        shape = "triangle"
    elif n == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        shape = "square" if 0.9 <= w / h <= 1.1 else "rectangle"
    elif n == 5:
        shape = "pentagon"
    else:
        shape = "circle"

    return shape


def init():
    global cam

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError('Failed to open a capture object.')

    cv2.namedWindow('MAIN')
    cv2.namedWindow('BARS')

    cv2.createTrackbar('sigma', 'BARS', 33, 100, do_nothing)
    # cv2.createTrackbar('canny_low', 'BARS', 20, 100, do_nothing)
    # cv2.createTrackbar('canny_high', 'BARS', 150, 255, do_nothing)
    cv2.createTrackbar('blur', 'BARS', 3, 10, do_nothing)
    cv2.createTrackbar('invert', 'BARS', 0, 1, do_nothing)

    cv2.moveWindow('MAIN', 55, 0)
    cv2.moveWindow('BARS', 60, 550)
    cv2.resizeWindow('BARS', 1000, 1)


def main():
    init()

    while True:
        # Read frame
        _, image = cam.read()

        # Flip
        image = cv2.flip(image, flipCode=1)

        # Blur
        blur = 1 + 2 * cv2.getTrackbarPos('blur', 'BARS')
        blurred = cv2.medianBlur(image, blur)

        # Invert
        if cv2.getTrackbarPos('invert', 'BARS'):
            blurred = cv2.bitwise_not(blurred)

        # # Grayscale
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Canny and morphs
        # canny_low = cv2.getTrackbarPos('canny_low', 'BARS')
        # canny_high = 3 * canny_low
        # canny_high = max(canny_low, cv2.getTrackbarPos('canny_high', 'BARS'))
        # sigma = 0.33
        sigma = cv2.getTrackbarPos('sigma', 'BARS') / 100
        v = np.median(blurred)
        canny_low = int(max(0, (1 - sigma) * v))
        canny_high = int(min(255, (1 + sigma) * v))
        edged = cv2.Canny(blurred, canny_low, canny_high)
        edged = cv2.dilate(edged, None, iterations=3)
        edged = cv2.erode(edged, None, iterations=2)

        # Find contours
        _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            box = cv2.minAreaRect(contour)
            box = cv2.boxPoints(box)

            shape = detect(contour)
            M = cv2.moments(contour)
            if M['m00']:
                text_pos = (int(M['m10'] / M['m00']),
                            int(M['m01'] / M['m00']))
            else:
                text_pos = (50, 50)

            cv2.drawContours(image, [box.astype('int')], -1, (0, 255, 0), 2)
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
            cv2.putText(image, shape, text_pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=4)
            cv2.putText(image, shape, text_pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)

        edged_bgr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        cv2.imshow('MAIN', np.hstack([image, edged_bgr]))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()
