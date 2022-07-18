from slam import Backend, Frame

import numpy as np
import cv2

class BinaryBackend(Backend):

    def __init__(self, args, frontend, landmarks):
        super().__init__(args, frontend, landmarks)

        self.bit_table = np.array([i.bit_count() for i in range(256)])

    def evaluate(self, frame: Frame):
        X = frame.pose @ self.landmarks.points[frame.observations]
        x = cv2.convertPointsFromHomogeneous(X).reshape((-1, 2))
        u = self.frontend.camera.mat @ x

        ref, mask = self.frontend.extract_descriptors_at(frame, u)
        desc = self.landmarks.descriptors[frame.observations]

        cost = np.bitwise_xor(ref[mask], desc[mask]).flatten()
        error = np.sum(self.bit_table[cost], dtype=int)

        return error / np.count_nonzero(mask)

