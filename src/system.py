from enum import Enum

from slam import Frame, Landmarks, Pose
from slam import direct

import argparse

import cv2
import numpy as np

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
        self.landmarks = Landmarks()

        self.frontend = direct.DirectFrontend({}, camera)
        self.direct_backend = direct.DirectBackend({}, self.frontend, self.landmarks)
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

            # compute landmark descriptors from frame0
            print('points0')
            print(result.points0[:5])
            descriptors, mask = self.frontend.extract_descriptors_at(frame0, result.points0)

            # only store fetures and landmarks which we successfully extracted features for
            self.landmarks.initialize(result.triangulated_points[mask], descriptors[mask])

            frame0.features = result.points0[mask]
            frame0.observations = np.arange(len(self.landmarks))

            frame1.features = result.points1[mask]
            frame1.observations = np.arange(len(self.landmarks))
            frame1.pose = result.transform

            #self.motion_model.initialize(result.transform, n_frames=self.state.failed_initializations + 1)

            self.state.track_state = TrackingState.TRACKING

            pose = frame1.pose.copy()
            for dx in range(200):
                for dy in range(200):
                    dxf = (dx - 100) / 100
                    dyf = (dy - 100) / 100

                    theta = 0.01 * dyf
                    dR = np.array([
                        [np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, 0],
                        [np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)

                    frame1.pose = Pose(dR @ pose.R, pose.t + (dxf, 0, 0))
                    print(dxf, theta, self.direct_backend.evaluate(frame1))

        else:

            self.state.failed_initializations += 1
