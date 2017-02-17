import cv2


def mouse_callback(event, x, y, flags, param):
    global selecting_region, region_tl, region_br, mouse_pos

    if event == cv2.EVENT_RBUTTONDOWN:
        selecting_region = True
        region_tl = (x, y)
        region_br = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
        if selecting_region:
            region_br = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        selecting_region = False
        (tlx, tly), (brx, bry) = sorted((region_tl, region_br))
        print('ROI    TL:({}, {})    BR:({}, {})'.format(tlx, tly, brx, bry))
        if brx > tlx and bry > tly:
            roi = image_base[tly:bry, tlx:brx]
            cv2.imwrite(roi_name, roi)
        else:
            print('[-] Possible empty ROI')

# CAMERA INIT
cam = cv2.VideoCapture(1)
if cam.isOpened():
    mirror = False
else:
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        mirror = True
    else:
        raise ValueError('Failed to open a capture object.')

# GLOBALS
selecting_region = False
region_tl = (0, 0)
region_br = (0, 0)
roi_name = 'roi.png'
mouse_pos = (0, 0)

# WINDOW
cv2.namedWindow('MAIN')

# MOUSE CALLBACK
cv2.setMouseCallback('MAIN', mouse_callback)

while True:
    _, image = cam.read()
    if mirror:
        image = cv2.flip(image, flipCode=1)
    image_base = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.putText(image, '{}, {}'.format(*mouse_pos), mouse_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    if selecting_region:
        image = cv2.rectangle(image, region_tl, region_br, (0, 0, 255), 2)

    cv2.imshow('MAIN', image)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
