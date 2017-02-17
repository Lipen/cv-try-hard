import cv2
import numpy as np


def nothing(x):
    pass


def mirror_callback(value):
    global mirror
    mirror = bool(value)


def init():
    global cam, mirror
    cam = cv2.VideoCapture(1)
    if cam.isOpened():
        mirror = False
    else:
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            mirror = True
        else:
            raise ValueError('Failed to open a capture object.')

    cv2.namedWindow('Main')
    cv2.namedWindow('Bars')

    cv2.createTrackbar('threshold_min', 'Bars', 100, 255, nothing)
    cv2.createTrackbar('invert', 'Bars', 0, 1, nothing)
    cv2.createTrackbar('blur_type', 'Bars', 1, 3, nothing)
    cv2.createTrackbar('blur', 'Bars', 2, 10, nothing)
    cv2.createTrackbar('mirror', 'Bars', int(mirror), 1, mirror_callback)

    cv2.moveWindow('Main', 55, 0)
    cv2.moveWindow('Bars', 80, 550)
    cv2.resizeWindow('Bars', 800, 100)

    global base_contour
    base_contour = np.array([[[43, 17]], [[31, 24]], [[27, 33]], [[85, 183]], [[75, 194]], [[84, 217]], [[129, 206]], [[139, 197]], [[136, 180]], [[132, 174]], [[123, 175]], [[118, 169]], [[61, 20]], [[56, 16]]])


def main():
    init()

    while True:
        # Read frame
        _, image = cam.read()

        # Mirror
        if mirror:
            image = cv2.flip(image, flipCode=1)

        # Blur
        blur = 1 + 2 * cv2.getTrackbarPos('blur', 'Bars')
        blur_type = cv2.getTrackbarPos('blur_type', 'Bars')
        if blur_type == 0:
            blurred = cv2.GaussianBlur(image, (blur, blur), 0)
        elif blur_type == 1:
            blurred = cv2.medianBlur(image, blur)
        elif blur_type == 2:
            blurred = cv2.blur(image, (blur, blur))
        elif blur_type == 3:
            blurred = cv2.bilateralFilter(image, 9, blur, blur)
        else:
            raise ValueError("Unsupported blur type")

        # Grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Invert
        if cv2.getTrackbarPos('invert', 'Bars'):
            gray = cv2.bitwise_not(gray)

        # Threshold
        threshold_min = cv2.getTrackbarPos('threshold_min', 'Bars')
        _, thresh = cv2.threshold(gray, threshold_min, 255, cv2.THRESH_BINARY)

        # Find contours
        _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            ret = cv2.matchShapes(contour, base_contour, 1, 0)
            text = '{:.3f}'.format(ret)

            M = cv2.moments(contour)
            text_pos = (int(M['m10'] / M['m00']),
                        int(M['m01'] / M['m00'])) if M['m00'] else (50, 50)

            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(image, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(0, 0, 0), thickness=4)
            cv2.putText(image, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(255, 255, 255), thickness=2)

        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Main", np.hstack((image, thresh_bgr)))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()
