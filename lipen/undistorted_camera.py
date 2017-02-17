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
    h, w = cam.read()[1].shape[:2]
    print('[*] (w, h): ({}, {})'.format(w, h))

    cv2.namedWindow('MAIN')

    global rms, camera_matrix, dist_coefs, new_camera_matrix
    with open('data_distortion.pickle', 'rb') as f:
        rms, camera_matrix, dist_coefs = pickle.load(f)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0, (w, h))


def main():
    init()

    while True:
        _, image = cam.read()
        if mirror:
            image = cv2.flip(image, flipCode=1)

        image_undistorted = cv2.undistort(image, camera_matrix, dist_coefs, None, new_camera_matrix)

        cv2.imshow('MAIN', np.hstack([image, image_undistorted]))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()
