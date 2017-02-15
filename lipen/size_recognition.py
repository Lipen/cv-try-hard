import cv2
import math
import numpy as np
from imutils import perspective


def do_nothing(_):
    pass


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def getWHImage(l, hfov=58, vfov=45, dfov=70):
    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)
    dfov_rad = math.radians(dfov)
    width = 2 * l * math.tan(hfov_rad / 2)
    height = 2 * l * math.tan(vfov_rad / 2)
    diag = 2 * l * math.tan(dfov_rad / 2)
    return width, height, diag


def init():
    global cam

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError('Failed to open a capture object.')

    cv2.namedWindow('MAIN')
    cv2.namedWindow('BARS')

    cv2.createTrackbar('canny_low', 'BARS', 20, 100, do_nothing)
    # cv2.createTrackbar('canny_high', 'BARS', 150, 255, do_nothing)
    cv2.createTrackbar('blur (odd)', 'BARS', 3, 10, do_nothing)
    cv2.createTrackbar('invert', 'BARS', 0, 1, do_nothing)

    cv2.moveWindow('MAIN', 55, 0)
    cv2.moveWindow('BARS', 60, 550)
    cv2.resizeWindow('BARS', 1000, 100)


def main():
    init()

    while True:
        # Read frame
        _, image = cam.read()
        # Flip
        image = cv2.flip(image, flipCode=1)
        # Invert
        if cv2.getTrackbarPos('invert', 'BARS'):
            image = cv2.bitwise_not(image)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Blur
        blur = cv2.getTrackbarPos('blur', 'BARS')
        if blur % 2 == 0:
            blur += 1
        blurred = cv2.medianBlur(gray, 5)
        # Canny and morphs
        canny_low = cv2.getTrackbarPos('canny_low', 'BARS')
        canny_high = 3 * canny_low
        # canny_high = max(canny_low, cv2.getTrackbarPos('canny_high', 'BARS'))
        edged = cv2.Canny(blurred, canny_low, canny_high)
        edged = cv2.dilate(edged, None, iterations=3)
        edged = cv2.erode(edged, None, iterations=2)

        # Invert back
        if cv2.getTrackbarPos('invert', 'BARS'):
            image = cv2.bitwise_not(image)

        # Find contours
        _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x0, y0 = image.shape[:2]
        diag0 = (x0**2 + y0**2)**0.5
        l = 400
        width_mm, height_mm, diag_mm = getWHImage(l, 54.5, 42.3, 66.17)
        ratio = diag_mm / diag0

        for contour in contours:
            if cv2.contourArea(contour) < 400:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            dA = math.hypot(tltrY - blbrX, tltrY - blbrX)
            dB = math.hypot(tlblX - trbrX, tlblY - trbrY)
            dD = math.hypot(dA, dB)

            xoc, yoc = midpoint(tl, br)
            xcc = x0 - yoc
            ycc = y0 - xoc

            alpha = math.atan2(dB, dA)
            dimD = dD * ratio
            dimA = dimD * math.sin(alpha)
            dimB = dimD * math.cos(alpha)

            for x, y in box:
                cv2.circle(image, (x, y), 5, (255, 0, 255), 2)

            cv2.drawContours(image, [box.astype('int')], -1, (0, 255, 0), 2)
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

            cv2.putText(image, "{:.1f}px;{:.1f}mm".format(dA, dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(255, 0, 0), thickness=1)
            cv2.putText(image, "{:.1f}px;{:.1f}mm".format(dB, dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(255, 0, 0), thickness=1)

        images_concatenated = cv2.hconcat([image, cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('MAIN', images_concatenated)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()
