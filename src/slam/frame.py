import numpy as np

from slam.pose import Pose

class Frame(object):
    def __init__(self):

        self.pose = Pose()

        self.features = np.zeros((0, 2), dtype=np.float32)
        self.observations = np.zeros((0, 2), dtype=int)

        self.im = None
        self.lkr_pyramid = None

