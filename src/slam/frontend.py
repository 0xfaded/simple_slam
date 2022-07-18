from slam.frame import Frame
from slam.pose import Pose

import numpy as np
import cv2

class Frontend(object):

    class InitializationResult:
        def __init__(self, success, transform, points0, points1, triangulated_points):
            self.success = success
            self.transform = transform
            self.points0 = points0
            self.points1 = points1
            self.triangulated_points = triangulated_points

        def __bool__(self):
            return self.success

    def __init__(self, args, camera):
        self.camera = camera

        self.gaussian_blur_win_size = 5

        self.initialization_min_landmark_inliers = 50
        self.initialization_max_landmark_dist = 100.0

    def init_frame(self, frame: Frame, image):
        frame.im = cv2.GaussianBlur(image, (self.gaussian_blur_win_size,)*2, 1.0)

    def init_initialization_frame(self, frame: Frame):
        pass


    def solve_initial_transform(self, frame0: Frame, frame1: Frame):

        points0, points1 = self.initial_correspondences(frame0, frame1)

        essential_matrix, mask = cv2.findEssentialMat(points0, points1, self.camera.mat.mat, cv2.RANSAC)

        num_inliers, R, t, mask, triangulated_points = cv2.recoverPose(
                essential_matrix, points0, points1, self.camera.mat.mat,
                distanceThresh=self.initialization_max_landmark_dist, mask=mask)

        mask = mask.astype(bool).reshape((-1,))
        points0 = points0[mask]
        points1 = points1[mask]
        triangulated_points = cv2.convertPointsFromHomogeneous(triangulated_points.T[mask]).reshape((-1,3))

        print(triangulated_points.shape)
        print(points0.shape)

        X = triangulated_points
        x0 = self.camera.mat.inverse() @ points0
        y0 = cv2.convertPointsFromHomogeneous(X).reshape((-1, 2))

        transform = Pose(R, t)

        print(R)
        print(t)
        #Y = X @ R.T + t.T
        Y = transform @ X
        print('t\n',t)
        x1 = self.camera.mat.inverse() @ points1
        y1 = cv2.convertPointsFromHomogeneous(Y).reshape((-1, 2))

        xy0 = x0 - y0
        xy1 = x1 - y1

        n = points0 - points1
        print(n.T @ n)

        print('hi\n', xy0.T @ xy0)
        print('ji\n', xy1.T @ xy1)

        success = num_inliers >= self.initialization_min_landmark_inliers
        return Frontend.InitializationResult(success, transform, points0, points1, triangulated_points)


    def initial_correspondences(frame0: Frame, frame1: Frame):
        raise NotImplementedError()
