from camera import Camera
from slam import CameraMatrix

from pathlib import Path
import numpy as np
import cv2

class FileCamera(Camera):

    def __init__(self, path, calib, start_frame=0, scale=1):

        (mat, dist, size) = self.read_camera_matrix(calib)

        if (dist != 0.0).any() or scale != 1:
            mat2 = mat.mat.copy()
            mat2[:2,:] /= scale

            size = (size[0] // scale, size[1] // scale)

            self.undistort = cv2.initUndistortRectifyMap(mat.mat, dist, None, mat2, size, cv2.CV_32FC1)
            mat = CameraMatrix(mat2)
        else:
            self.undistort = None

        self.mat = mat
        self.frame = start_frame

        self.fmt = path

    def next_image(self):
        path = self.fmt.format(self.frame)
        self.frame += 1

        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if self.undistort is not None:
            mx, my = self.undistort
            im = cv2.remap(im, mx, my, cv2.INTER_LINEAR)

        return im

    def read_camera_matrix(self, path):
        K = None
        dist = np.zeros((5,), dtype=np.float32)
        w, h = -1, -1

        with open(path, 'r') as calib:
            for line in calib:
                parts = line.split()
                if len(parts) == 10 and parts[0] == 'K:':
                    vals = list(map(float, parts[1:]))
                    K = np.array(vals, dtype=np.float32).reshape((3,3))
                elif len(parts) == 6 and parts[0] == 'D:':
                    vals = list(map(float, parts[1:]))
                    dist = np.array(vals, dtype=np.float32)
                elif len(parts) == 3 and parts[0] == 'S:':
                    w, h = list(map(int, parts[1:]))

        if K is None or w == -1:
            raise RuntimeError("Invalid Configuration File")

        return CameraMatrix(K), dist, (w, h)
