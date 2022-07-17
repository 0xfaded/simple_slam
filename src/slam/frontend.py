from slam.frame import Frame

import numpy as np
import cv2

class Frontend(object):

    class InitializationResult:
        def __init__(self, success, transform, points0, points1, triangulated_points, mask):
            self.success = success
            self.transform = transform
            self.points0 = points0[mask]
            self.points1 = points1[mask]
            self.triangulated_points = triangulated_points[mask]
            self.obs0 = np.arange(self.points0.shape[0], dtype=int)
            self.obs1 = np.copy(self.obs0)

        def __bool__(self):
            return self.success

    def __init__(self, args, camera):
        self.camera = camera

        self.gaussian_blur_win_size = 5

        self.initialization_min_landmark_inliers = 50
        self.initialization_min_landmark_dist = 0.25

    def init_frame(self, frame: Frame, image):
        frame.im = cv2.GaussianBlur(image, (self.gaussian_blur_win_size,)*2, 1.0)

    def init_initialization_frame(self, frame: Frame):
        pass


    def solve_initial_transform(self, frame0: Frame, frame1: Frame):

        points0, points1 = self.initial_correspondences(frame0, frame1)

        essential_matrix, mask = cv2.findEssentialMat(points0, points1, self.camera.cam_matrix, cv2.RANSAC)

        num_inliers, R, t, mask, triangulated_points = cv2.recoverPose(
                essential_matrix, points0, points1, self.camera.cam_matrix,
                distanceThresh=self.initialization_min_landmark_dist, mask=mask)

        success = num_inliers >= self.initialization_min_landmark_inliers
        return Frontend.InitializationResult(success, (R, t), points0, points1, triangulated_points, mask)


    def initial_correspondences(frame0: Frame, frame1: Frame):
        raise NotImplementedError()
