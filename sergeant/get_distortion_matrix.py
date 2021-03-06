import cv2
import numpy as np
import pickle


def init():
    global cam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError('Failed to open a capture object.')

    global w, h
    h, w = cam.read()[1].shape[:2]
    print('[*] (w, h): {!r}'.format((w, h)))

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
