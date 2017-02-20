import cv2
import math
import numpy as np
from imutils import perspective


def do_nothing(_):
    pass


def midpoint(ptA, ptB):
    return (ptA + ptB) / 2


def getWHImage(l, hfov=58, vfov=45, dfov=70):
    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)
    dfov_rad = math.radians(dfov)
    width = 2 * l * math.tan(hfov_rad / 2)
    height = 2 * l * math.tan(vfov_rad / 2)
    diag = 2 * l * math.tan(dfov_rad / 2)
    return width, height, diag


def put_text(image, pos, *args, fmt='{:.0f}mm'):
    cv2.putText(image, fmt.format(*args), (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(255, 100, 0), thickness=1)


def put_text2(image, pos, *args, fmt='{:.0f}mm'):
    cv2.putText(image, fmt.format(*args), (int(pos[0]), int(pos[1])),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                color=(255, 255, 0), thickness=2)


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

    cv2.namedWindow('MAIN')
    cv2.namedWindow('BARS')

    # cv2.createTrackbar('canny_low', 'BARS', 20, 100, do_nothing)
    # cv2.createTrackbar('canny_high', 'BARS', 150, 255, do_nothing)
    cv2.createTrackbar('dfov', 'BARS', 68, 100, do_nothing)
    cv2.createTrackbar('l', 'BARS', 45, 100, do_nothing)
    cv2.createTrackbar('sigma', 'BARS', 33, 100, do_nothing)
    cv2.createTrackbar('blur', 'BARS', 3, 10, do_nothing)

    cv2.moveWindow('MAIN', 55, 0)
    cv2.moveWindow('BARS', 60, 500)
    cv2.resizeWindow('BARS', 1000, 10)


def main():
    init()

    while True:
        # Read frame
        _, image = cam.read()

        # Mirror
        if mirror:
            image = cv2.flip(image, flipCode=1)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur
        blur = 1 + 2 * cv2.getTrackbarPos('blur', 'BARS')
        blurred = cv2.medianBlur(gray, blur)

        # Canny and morphs
        sigma = cv2.getTrackbarPos('sigma', 'BARS') / 100
        v = np.median(blurred)
        canny_low = int(max(0, (1 - sigma) * v))
        canny_high = int(min(255, (1 + sigma) * v))
        # canny_low = cv2.getTrackbarPos('canny_low', 'BARS')
        # canny_high = 3 * canny_low
        # canny_high = max(canny_low, cv2.getTrackbarPos('canny_high', 'BARS'))
        edged = cv2.Canny(blurred, canny_low, canny_high)
        edged = cv2.dilate(edged, None, iterations=3)
        edged = cv2.erode(edged, None, iterations=2)

        # Find contours
        _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        y0, x0 = image.shape[:2]
        diag0 = math.hypot(x0, y0)
        # l = 400
        l = max(0, cv2.getTrackbarPos('l', 'BARS')) * 10
        hfov = 54.5
        vfov = 42.3
        # dfov = 66.17
        dfov = cv2.getTrackbarPos('dfov', 'BARS')
        width_mm, height_mm, diag_mm = getWHImage(l, hfov, vfov, dfov)
        ratio = diag_mm / diag0

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue

            # @rect :: tuple((center_x, center_y), (width, height), angle_deg)
            # box :: np.array([[x y], [x y], [x y], [x y]]) -- ordered as (top_left, top_right, bottom_right, bottom_left)
            # dA -- width [px]
            # dB -- height [px]
            # dD -- diagonal [px]
            # dimA, dimB, dimD -- same as above [mm]

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            dA = math.hypot(tltrX - blbrX, tltrY - blbrY)
            dB = math.hypot(tlblX - trbrX, tlblY - trbrY)
            dD = math.hypot(dA, dB)

            xoc, yoc = midpoint(tl, br)
            xcc = xoc - x0 / 2
            ycc = y0 / 2 - yoc

            alpha = math.atan2(dB, dA)
            dimD = dD * ratio
            dimA = dimD * math.sin(alpha)  # Why not cos?
            dimB = dimD * math.cos(alpha)

            for x, y in box:
                cv2.circle(image, (x, y), 5, (255, 0, 255), 2)

            cv2.drawContours(image, [box.astype('int')], -1, (0, 255, 0), 2)
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

            put_text(image, (tltrX - 15, tltrY - 10), dimA)  # width
            put_text(image, (trbrX + 10, trbrY), dimB)  # height
            put_text2(image, (xoc, yoc), xcc * ratio, ycc * ratio, fmt='({:.0f},{:.0f})')

        edged_bgr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        cv2.imshow('MAIN', np.hstack([image, edged_bgr]))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()
