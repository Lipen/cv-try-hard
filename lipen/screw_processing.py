import cv2

filename = 'screw_w.png'

image = cv2.imread(filename)
blurred = cv2.medianBlur(image, 5)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def get_approx(contour):
    peri = cv2.arcLength(contour, True)
    epsilon = 0.005
    approx = cv2.approxPolyDP(contour, epsilon * peri, True)
    return approx

for i, contour in enumerate(contours):
    text = '#{}'.format(i)
    # print('#{}:\n{!r}'.format(i, contour))
    approx = get_approx(contour)
    print('#{}:\n{!r}'.format(i, approx))

    M = cv2.moments(approx)
    text_pos = (int(M['m10'] / M['m00']),
                int(M['m01'] / M['m00'])) if M['m00'] else (50, 50)

    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv2.putText(image, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 0), thickness=4)
    cv2.putText(image, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(255, 255, 255), thickness=2)

cv2.imshow("Image", image)

cv2.waitKey(-1)
