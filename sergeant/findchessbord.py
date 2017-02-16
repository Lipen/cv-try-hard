import numpy as np
import cv2
import glob
import pickle
def draw(img, corners, imgpts):
	corner = tuple(corners[0].ravel())
	img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
	return img

def calibrate():
	params = open('params.pickle','wb')
	square_size = 1
	pattern_points = np.zeros((np.prod(chessbrdSize), 3), np.float32)
	pattern_points[:, :2] = np.indices(chessbrdSize).T.reshape(-1, 2)
	pattern_points *= square_size
	_, img = cap.read()
	camera_matrix = None
	dist_coefs = None
	while (1):
		ret, img = cap.read()
		found, corners = cv2.findChessboardCorners(img, chessbrdSize)
		obj_points = []
		img_points = []
		if found:
			img_points.append(corners.reshape(-1, 2))
			obj_points.append(pattern_points)
			term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
			# cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
			h,  w = img.shape[:2]
			rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), camera_matrix, dist_coefs, 
				flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT)
			# mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
			# undistorted_img = cv2.undistort(img, camera_matrix, dist_coefs, None, mtx)
			# x, y, w, h = roi
			# dist_coefs = dist_coefs[y:y+h, x:x+w]
			# params = np.load('params.pickle', 'wb')
			print("calibrating atm")
			img = cv2.drawChessboardCorners(img, (chessbrdSize), corners,ret)
		if not found:
			print('chessboard not found')
			# continue
		cv2.imshow('calibrating', img)
		if cv2.waitKey(1) & 0xFF == 27:
			break
	cv2.destroyAllWindows()
	pickle.dump([camera_matrix, dist_coefs],params)
	print("successfully calibrated camera")
	params.close()

def main():
	global cap, chessbrdSize
	chessbrdSize = (7,5)
	cap = cv2.VideoCapture(0)
	
	# calibrate()
	
	params = open('params.pickle','rb')
	camera_matrix, dist_coefs, = pickle.load(params)
	params.close() 
	print('camera matrix: \n', camera_matrix)
	print('dist: \n', dist_coefs)
	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
	rvecs = None 
	tvecs = None
	while (1):

		_, img = cap.read()
		# termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		objp = np.zeros((6*7,3), np.float32)
		objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
		ret, corners = cv2.findChessboardCorners(gray, (chessbrdSize),None)
		if ret == True:
			objpoints.append(objp)

			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners2)

			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (chessbrdSize), corners2,ret)
			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

			# Find the rotation and translation vectors.
			rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, dist_coefs,rvecs,tvecs)
			# rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, dist_coefs)

			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coefs)

			img = draw(img,corners2,imgpts)
		cv2.imshow('img',img)
		# cv2.imshow('gray',gray)
		if cv2.waitKey(1) & 0xFF == 27:
			break

	cv2.destroyAllWindows()
	cap.release()

if __name__ == '__main__':
	main()