from slam import Frontend, Frame

import cv2

class DirectFrontend(Frontend):

    def __init__(self, args, camera):
        super().__init__(args, camera)

        self.max_features = 1000

        self.optical_flow_win_size = 21
        self.optical_flow_pyr_levels = 5

    def init_initialization_frame(self, frame: Frame):
        super().init_initialization_frame(frame)

    def initial_correspondences(self, frame0: Frame, frame1: Frame):
        features, status, error = cv2.calcOpticalFlowPyrLK(frame0.im, frame1.im, frame0.features, None,
                winSize=self._lkr_win_size(), maxLevel=self.optical_flow_pyr_levels)

        mask = status.astype(bool)

        return (frame0.features[mask], features[mask])

    def locate_features(self, frame: Frame):
        return cv2.goodFeaturesToTrack(frame.im, self.max_features, 0.01, self.optical_flow_win_size)

    def _lkr_win_size(self):
        return (self.optical_flow_win_size,)*2
