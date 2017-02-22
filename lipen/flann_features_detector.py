import cv2
import numpy as np

# CAMERA INIT
cam = cv2.VideoCapture(1)
cam.read()

# GLOBALS
wheel_filename = 'wheel.png'
MIN_MATCH_COUNT = 10
FLANN_INDEX_TREE = 0
index_params = dict(algorithm=FLANN_INDEX_TREE, trees=5)
search_params = dict(checks=50)
# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=0)

# STUFF
detector = cv2.xfeatures2d.SIFT_create()
cv2.FastFeatureDetector_create
wheel = cv2.imread(wheel_filename)
wheel_keypoints, wheel_descriptions = detector.detectAndCompute(wheel, None)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cam.read()
    image_base = cv2.flip(frame, flipCode=1)
    image = image_base.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_keypoints, image_descriptions = detector.detectAndCompute(gray, None)
    try:
        matches = flann.knnMatch(wheel_descriptions, image_descriptions, k=2)
    except:
        pass
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([wheel_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([image_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.)
        matches_mask = mask.ravel().tolist()

        h, w, *_ = wheel.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # try:
        #     dst = cv2.perspectiveTransform(pts, M)
        # except:
        #     pass

        # image = cv2.polylines(image, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    else:
        matches_mask = None

    # ##########################
    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matches_mask[i] = [1, 0]
    # ##########################

    # canvas = cv2.drawMatches(wheel, wheel_keypoints, image, image_keypoints, good, None, matchesMask=matches_mask, **draw_params)
    canvas = cv2.drawMatchesKnn(wheel, wheel_keypoints, image, image_keypoints, matches, None, matchesMask=matches_mask, **draw_params)

    cv2.imshow('FLANN', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
