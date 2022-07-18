import numpy as np

class Landmarks(object):

    def __init__(self):
        self.points = np.zeros((0,3), dtype=np.float32)
        self.descriptors = None
        self.good = None

    def initialize(self, points, descriptors, good=None):
        self.points = points
        self.descriptors = descriptors
        self.good = good is not None and good or np.full((points.shape[0],), True)

    def __len__(self):
        return self.points.shape[0]
