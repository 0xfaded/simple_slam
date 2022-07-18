import numpy as np
import cv2

class Pose:

    def __init__(self, R=None, t=None):
        if R is None: R = np.eye(3, dtype=np.float32)
        if t is None: t = np.zeros((3,), dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.t = np.array(t, dtype=np.float32).reshape(3,)

    def __matmul__(self, points):
        if points.shape[0] == 4 or points.shape[-1] == 4:
            M = np.zeros((4,4), self.R.dtype)
            M[:3,:3] = self.R
            M[:3,3] = self.t.reshape((3,1))

            if points.shape[0] == 4:
                return M @ points
            else:
                return points @ M.T

        elif points.shape[0] == 3:
            return self.R @ points + self.t.reshape((3,1))
        elif points.shape[-1] == 3:
            return points @ self.R.T + self.t.reshape((3,))
        else:
            raise NotImplementedError()

    def copy(self):
        return Pose(self.R.copy(), self.t.copy())

    def __str__(self):
        return str(np.hstack([self.R, self.t.reshape(3,1)]))

    def __repr__(self):
        return str(self)

    def inverse(self):
        return Pose(self.R.T, -self.R.T @ self.t)
