from enum import Enum

from slam import Frame
from slam import direct

import argparse

import cv2

class TrackingState(Enum):
    INITIALIZING = 0
    TRACKING = 1

class System:
    """The System object encapsulates two kinds of state:
    1. The "moving parts", such as the landmarks, motion model, statistics, etc.
    2. Configured sub-components, such as feature extractors or optimizers.

    SLAM systems tend to be highly configurable with many tuneable parameters.
    In our case we read these parameters from the command line, but some
    implementations can automatically tune the parameters during execution.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    class State:
        def __init__(self):
            self.reset()

        def reset(self):
            self.track_state = TrackingState.INITIALIZING
            self.set_frame0(None)

        def set_frame0(self, frame):
            self.frame0 = frame 
            self.failed_initializations = 0


    def __init__(self, camera, args = {}):

        self.max_failed_initializations = 5

        self.state = System.State()

        self.frontend = direct.DirectFrontend({}, camera)
        #self.binary_frontend = BinaryFrontend()

    def process_image(self, image):

        frame = Frame()
        self.frontend.init_frame(frame, image)

        #cv2.imshow('im', frame.im)
        #cv2.waitKey(-1)

        if self.state.track_state == TrackingState.INITIALIZING:
            self.initialize(frame)
        else:
            pass


    def initialize(self, frame: Frame):

        if (self.state.frame0 is None) or (self.state.failed_initializations >= self.max_failed_initializations):

            self.frontend.init_initialization_frame(frame)

            frame.features = self.frontend.locate_features(frame)
            self.state.set_frame0(frame)

        elif result := self.frontend.solve_initial_transform(self.state.frame0, frame):
            frame0 = self.state.frame0
            frame1 = frame

            #frame0.features = self.frontend.extract_features_at(frame, result.points0)
            frame0.observations = result.obs0

            #frame1.features = self.frontend.extract_features_at(frame, result.points1)
            frame1.observations = self.obs1

            #self.motion_model.initialize(result.transform, n_frames=self.state.failed_initializations + 1)
            #self.landmarks.initialize(result.transform)

            self.state.track_state = TrackingState.TRACKING

        else:

            self.state.failed_initializations += 1
