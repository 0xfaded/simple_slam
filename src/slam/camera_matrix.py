import numpy as np

class CameraMatrix:

    def __init__(self, mat):
        self.mat = np.array(mat, dtype=np.float32)

    def __matmul__(self, points):
        if points.shape[0] == 3:
            return self.mat @ points
        elif points.shape[-1] == 3:
            return points @ self.mat.T
        elif points.shape[0] == 2:
            return points * self.mat.diagonal()[:2,np.newaxis] + self.mat[:2,2,np.newaxis]
        elif points.shape[-1] == 2:
            return (self @ points.T).T

    def __str__(self):
        return str(self.mat)

    def __repr__(self):
        return repr(self.mat)

    def inverse(self):
        return CameraMatrix(np.linalg.inv(self.mat))
