from slam.frame import Frame

import numpy as np
import cv2

class Backend(object):

    def __init__(self, args, frontend, landmarks):
        self.frontend = frontend
        self.landmarks = landmarks

    def extract_descriptors_at(self, frame: Frame, points):
        raise NotImplementedError()
