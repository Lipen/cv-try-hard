import cv2
import numpy as np
import pickle


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)
    return img


def init():
    global cam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError('Failed to open a capture object.')

    global w, h
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('[*] (w, h): ({}, {})'.format(w, h))

    cv2.namedWindow('MAIN')

    global pattern_size, term_criteria, pattern_points
    pattern_size = (7, 5)
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    square_size = 1
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    print('[*] pattern_points.shape: {}'.format(pattern_points.shape))


def main():
    init()

    camera_matrix = None
    dist_coefs = None
    objp = np.zeros((7 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    while True:
        _, image = cam.read()
        image = cv2.flip(image, flipCode=1)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(image, pattern_size)

        if found:
            print('[+] Corner found.')
            # cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), term_criteria)
            img_points = [corners.reshape(-1, 2)]
            obj_points = [pattern_points]

            rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), camera_matrix, dist_coefs, flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT)

            cv2.drawChessboardCorners(image, pattern_size, corners, found)

            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
            image_undistorted = cv2.undistort(image, camera_matrix, dist_coefs, None)
            # (roi_x, roi_y, roi_w, roi_h) = roi
            # image_undistorted = image_undistorted[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # rvecs, tvecs, inliers, *_ = cv2.solvePnPRansac(objp, corners, camera_matrix, dist_coefs)
            imgpts, jac = cv2.projectPoints(axis, np.array(rvecs), np.array(tvecs), camera_matrix, dist_coefs)
            image_undistorted = draw(image_undistorted, corners, imgpts)

            cv2.imshow('MAIN', image_undistorted)
        else:
            cv2.imshow('MAIN', image)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

    print('[*] Final parameters:')
    if camera_matrix is not None:
        print('[+] RMS:\n    {}'.format(rms))
        print('[+] Camera matrix:')
        for row in camera_matrix:
            print('    {}'.format(row))
        print('[+] Distortion coefficients:\n    {}'.format(dist_coefs.ravel()))

        with open('data_distortion.pickle', 'wb') as f:
            pickle.dump((rms, camera_matrix, dist_coefs), f, pickle.HIGHEST_PROTOCOL)
    else:
        print('[-] NO DATA')

if __name__ == '__main__':
    main()
