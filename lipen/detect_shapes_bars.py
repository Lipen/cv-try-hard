import cv2
import imutils


def nothing(x):
    pass


def detect(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n = len(approx)

    if n == 3:
        shape = "triangle"
    elif n == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif n == 5:
        shape = "pentagon"
    else:
        shape = "circle"

    return shape


def init():
    global cam

    cam = cv2.VideoCapture(0)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.read()

    cv2.namedWindow('Image')
    cv2.namedWindow('Thresholded')
    cv2.namedWindow('Bars')

    cv2.createTrackbar('threshold_min', 'Bars', 100, 255, nothing)
    cv2.createTrackbar('invert', 'Bars', 0, 1, nothing)
    cv2.createTrackbar('blur_type', 'Bars', 1, 3, nothing)
    cv2.createTrackbar('blur', 'Bars', 1, 10, nothing)
    # cv2.createTrackbar('size', 'Bars', 480, 1280, nothing)
    cv2.createTrackbar('mirror', 'Bars', 1, 1, nothing)

    cv2.moveWindow('Image', 55, 0)
    cv2.moveWindow('Bars', 60, 550)
    cv2.resizeWindow('Bars', 1000, 100)


def main():
    init()

    while True:
        # Read frame
        _, image = cam.read()

        # Flip
        if cv2.getTrackbarPos('mirror', 'Bars'):
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
        invert = cv2.getTrackbarPos('invert', 'Bars')
        if invert:
            gray = cv2.bitwise_not(gray)

        # Threshold
        threshold_min = cv2.getTrackbarPos('threshold_min', 'Bars')
        _, thresh = cv2.threshold(gray, threshold_min, 255, cv2.THRESH_BINARY)

        # Find contours
        _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ratio = 1

        for contour in contours:
            shape = detect(contour)

            contour = contour.astype('float')
            contour *= ratio
            contour = contour.astype('int')

            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            M = cv2.moments(contour)
            if M['m00']:
                text_pos = (int((M['m10'] / M['m00']) * ratio),
                            int((M['m01'] / M['m00']) * ratio))
            else:
                text_pos = (50, 50)
            cv2.putText(image, shape, text_pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=4)
            cv2.putText(image, shape, text_pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)

        cv2.imshow("Image", image)
        # cv2.imshow("Blurred", blurred)
        cv2.imshow("Thresholded", thresh)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()
