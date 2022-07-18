from slam import Frontend, Frame

import cv2
import numpy as np

class BinaryFrontend(Frontend):

    def __init__(self, args, camera):
        super().__init__(args, camera)

        self.scale = 4
        self.descriptor_bytes = 16
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(self.descriptor_bytes)


    def extract_descriptors_at(self, frame: Frame, points):
        h, w = frame.im.shape

        if self.scale == 1:
            im = frame.im
        else:
            im = cv2.resize(frame.im, (self.scale*w, self.scale*h), cv2.INTER_NEAREST)
            points = points * self.scale

        kps = cv2.KeyPoint.convert(points)

        # use class_id to recover filtered points
        for i, kp in enumerate(kps):
            kp.class_id = i

        # mask will need to be computed by manually determinining which
        # points were filtered by brief.compute
        kps_out, descriptors_out = self.brief.compute(im, kps)

        descriptors = np.zeros((points.shape[0], self.descriptor_bytes), dtype=np.uint8)
        mask = np.zeros((points.shape[0],), dtype=bool)

        for kp in kps_out:
            mask[kp.class_id] = True

        descriptors[mask] = descriptors_out

        return descriptors, mask
