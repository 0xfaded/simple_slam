from slam import Frontend, Frame

import cv2
import numpy as np

class DirectFrontend(Frontend):

    def __init__(self, args, camera):
        super().__init__(args, camera)

        self.max_features = 1000

        self.feature_size = 8
        self.optical_flow_win_size = 21
        self.optical_flow_pyr_levels = 5

    def init_initialization_frame(self, frame: Frame):
        super().init_initialization_frame(frame)

    def extract_descriptors_at(self, frame: Frame, points):
        point_i = ((points + 0.5) - 0.5 * (self.feature_size)).astype(int)
        win_shape = (self.feature_size,)*2

        im = frame.im
        h, w = im.shape[:2]
        mask = np.logical_and(0 <= point_i, (point_i + win_shape) <= (w, h)).all(axis=1)

        channels = len(im.shape) < 3 and 1 or im.shape[2]
        descriptor_size = channels * self.feature_size**2

        descriptors = np.zeros((points.shape[0], descriptor_size), dtype=im.dtype)

        for i, (x, y) in enumerate(point_i):
            if not mask[i]:
                continue

            descriptors[i] = im[y:y+win_shape[1],x:x+win_shape[0]].flatten()

        return descriptors, mask


    def initial_correspondences(self, frame0: Frame, frame1: Frame):
        features, status, error = cv2.calcOpticalFlowPyrLK(frame0.im, frame1.im, frame0.features, None,
                winSize=self._lkr_win_size(), maxLevel=self.optical_flow_pyr_levels)

        mask = status.astype(bool)

        return (frame0.features[mask], features[mask])

    def locate_features(self, frame: Frame):
        return cv2.goodFeaturesToTrack(frame.im, self.max_features, 0.01, self.optical_flow_win_size)

    def _lkr_win_size(self):
        return (self.optical_flow_win_size,)*2
