import cv2
import numpy as np
import pickle


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

    img_points = []
    obj_points = []

    while True:
        _, image = cam.read()
        if mirror:
            image = cv2.flip(image, flipCode=1)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(image, pattern_size)

        if found:
            print('[+] Corner found.')
            # TODO: Why cornerSubPix doesn`t work?
            # cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), term_criteria)
            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)

        cv2.drawChessboardCorners(image, pattern_size, corners, found)
        cv2.imshow('MAIN', image)

        if cv2.waitKey(50) == 27:  # 1/50ms = 20 FPS
            break

    cv2.destroyAllWindows()
    cam.release()

    if img_points:
        print('[+] Total {} points to use'.format(len(img_points)))
        print('[*] Calibrating...')
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

        print('[*] Final parameters:')
        print('[+] RMS:\n    {}'.format(rms))
        print('[+] Camera matrix:')
        for row in camera_matrix:
            print('    {}'.format(row))
        print('[+] Distortion coefficients:\n    {}'.format(dist_coefs.ravel()))

        filename = 'data/data_distortion.pickle'
        print('[*] Dumping to {}...'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump((rms, camera_matrix, dist_coefs), f,
                        pickle.HIGHEST_PROTOCOL)
    else:
        print('[-] NO DATA TO CALIBRATE')

    print('[@] All done.')

if __name__ == '__main__':
    main()
