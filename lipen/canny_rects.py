import cv2


def do_nothing(_):
    pass


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

        # objects = []

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            box = cv2.minAreaRect(contour)
            box = cv2.boxPoints(box)

            # o = object()

            cv2.drawContours(image, [box.astype('int')], -1, (0, 255, 0), 2)
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

        images_concatenated = cv2.hconcat([image, cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('MAIN', images_concatenated)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()
