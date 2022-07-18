from camera import Camera
from slam import CameraMatrix

from pathlib import Path
import numpy as np
import cv2

class KittiCamera(Camera):

    def __init__(self, path, image_index=0, start_frame=0):
        path = Path(path)

        self.mat = self.read_camera_matrix(path / 'calib.txt', image_index)
        self.image_index = image_index
        self.frame = start_frame

        base, sep, _ = str(path / 'SPLIT' / 'SPLIT').rsplit('SPLIT')
        self.fmt = base + 'image_{0}' + sep + '{1:0>6}.png'

    def next_image(self):
        path = self.fmt.format(self.image_index, self.frame)
        self.frame += 1

        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def read_camera_matrix(self, path, image_index):
        tag = 'P{0}:'.format(image_index)

        with open(path, 'r') as calib:
            for line in calib:
                parts = line.split()
                if len(parts) == 13 and parts[0] == tag:
                    vals = list(map(float, parts[1:]))
                    mat = np.array(vals, dtype=np.float32).reshape((3,4))

                    return CameraMatrix(mat[:,0:3])

        raise RuntimeError('No line in {0} matched {1}'.format(path, image_index))
