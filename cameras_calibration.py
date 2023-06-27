import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

WORLD_SCALING = 0.08

def create_blob_detector():
    # Setup SimpleBlobDetector parameters.
    blobParams = cv.SimpleBlobDetector_Params()

    # Thresholds
    blobParams.minThreshold = 80
    blobParams.maxThreshold = 200

    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 100   
    blobParams.maxArea = 5000 

    # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.8

    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87

    # Filter by Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01

    # Create a detector with the parameters
    blobDetector = cv.SimpleBlobDetector_create(blobParams)
    return blobDetector

def calibrate_camera(images_folder, show_patter = False):

    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        img = cv.imread(imname, 1)
        images.append(img)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    # Circle grid shape
    rows = 8 
    columns = 6 

    # coordinates of squares in the CirclesGrid world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = WORLD_SCALING* objp
 
    # frame dimensions
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    # Pixel coordinates of CirclesGrid
    imgpoints = [] # 2d points in image plane.
 
    # coordinates of the CirclesGrid in CirclesGrid world space.
    objpoints = [] # 3d point in real world space
    
    # blob detector
    blobDetector = create_blob_detector() 

    for frame in images:
        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        keypoints = blobDetector.detect(img_gray)
 
        im_with_keypoints = cv.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)

        # find the CircleGrid
        ret, corners = cv.findCirclesGrid(im_with_keypoints, (rows, columns), None, flags = cv.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid
 
        if ret == True:
 
            corners = cv.cornerSubPix(im_with_keypoints_gray, corners, (4, 4), (-1, -1), criteria)
            if show_patter == True: # If show_patter == True show detected calibration pattern
                cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
                cv.imshow('img', frame)
                k = cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, images_path1, images_path2, show_pattern = False):

    c1_images_names = sorted(glob.glob(images_path1))
    c2_images_names = sorted(glob.glob(images_path2))

    c1_images = []
    c2_images = []

    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 8
    columns = 6
 
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = WORLD_SCALING* objp
 
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    imgpoints1 = []
    imgpoints2 = []
 
    objpoints = []
    
    blobDetector = create_blob_detector()

    for frame1, frame2 in zip(c1_images, c2_images):

        img_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY) 
        keypoints = blobDetector.detect(img_gray)
        im_with_keypoints = cv.drawKeypoints(frame1, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray1 = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)

        # find the CircleGrid
        ret1, corners1 = cv.findCirclesGrid(im_with_keypoints, (rows, columns), None, flags = cv.CALIB_CB_SYMMETRIC_GRID) 

        img_gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY) 
        keypoints2 = blobDetector.detect(img_gray2)
        im_with_keypoints2 = cv.drawKeypoints(frame2, keypoints2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray2 = cv.cvtColor(im_with_keypoints2, cv.COLOR_BGR2GRAY)

        ret2, corners2 = cv.findCirclesGrid(im_with_keypoints2, (rows, columns), None, flags = cv.CALIB_CB_SYMMETRIC_GRID) 

        if ret1 == True and ret2 == True:
            corners1 = cv.cornerSubPix(im_with_keypoints_gray1, corners1, (4, 4), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(im_with_keypoints_gray2, corners2, (4, 4), (-1, -1), criteria)

            if show_pattern == True:
                cv.drawChessboardCorners(frame1, (rows, columns), corners1, ret1)
                cv.imshow('img', frame1)
    
                cv.drawChessboardCorners(frame2, (rows, columns), corners2, ret2)
                cv.imshow('img2', frame2)
                cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

    cv.destroyAllWindows()
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
    return R, T

def save_parameters(mtx1, dist1, mtx2, dist2, R, T, path):

    data = {'camera_matrix': np.asarray(mtx1).tolist(), 'dist_coeff': np.asarray(dist1).tolist()}
    with open(os.path.join(path, "camera_1.yaml"), "w") as f:
        yaml.dump(data, f)

    data = {'camera_matrix': np.asarray(mtx2).tolist(), 'dist_coeff': np.asarray(dist2).tolist()}
    with open(os.path.join(path, "camera_2.yaml"), "w") as f:
        yaml.dump(data, f)

    data = {'rotation matrix': np.asarray(R).tolist(), 'translation vector': np.asarray(T).tolist()}
    with open(os.path.join(path, "stereovision.yaml"), "w") as f:
        yaml.dump(data, f)

def main():
    
    print("Cameras calibrating...")
    mtx1, dist1 = calibrate_camera("sekwencje/sequence_3/camera_1/calib/*")
    print("First camera calibrated")
    mtx2, dist2 = calibrate_camera("sekwencje/sequence_3/camera_2/calib/*")
    print("Second camera calibrated")

    print("Sterovison calibration...")
    R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, "sekwencje/sequence_3/camera_1/calib/*", "sekwencje/sequence_3/camera_2/calib/*")
    print("Sterovison calibration completed, Parameters saved to /calibration_parameters")


    save_parameters(mtx1, dist1, mtx2, dist2, R, T, "calibration_parameters/sequence_3")

if __name__ == "__main__":
    main()