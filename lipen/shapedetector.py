import cv2


class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, contour):
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
