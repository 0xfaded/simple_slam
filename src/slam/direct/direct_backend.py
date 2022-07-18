from slam import Backend, Frame

import numpy as np
import cv2

class DirectBackend(Backend):

    def __init__(self, args, frontend, landmarks):
        super().__init__(args, frontend, landmarks)

    def evaluate(self, frame: Frame):
        X = frame.pose @ self.landmarks.points[frame.observations]
        x = cv2.convertPointsFromHomogeneous(X).reshape((-1, 2))
        u = self.frontend.camera.mat @ x

        ref, mask = self.frontend.extract_descriptors_at(frame, u)
        desc = self.landmarks.descriptors[frame.observations]

        error = np.sum(np.abs(np.subtract(ref, desc, dtype=float)[mask]))

        return error

