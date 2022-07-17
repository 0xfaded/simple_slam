import numpy as np

class Frame(object):
    def __init__(self):

        self.features = np.zeros((0, 2), dtype=np.float32)
        self.observations = np.zeros((0, 2), dtype=int)

        self.im = None
        self.lkr_pyramid = None

