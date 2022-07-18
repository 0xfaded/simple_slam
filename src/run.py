from system import System
from kitti import KittiCamera
from util import FileCamera

#camera = KittiCamera('/media/data/kitti/dataset/sequences/04')
camera = FileCamera(
        '/media/data/slam_sets/cirs_caves_dataset/undistorted_frames/frame_ud_{0:0>6}.png',
        '/media/data/slam_sets/cirs_caves_dataset/calib.txt',
        start_frame=2500)
        #start_frame=4780)
        #start_frame=1000)
        #start_frame=140)

simple_slam = System(camera)

simple_slam.process_image(camera.next_image())
simple_slam.process_image(camera.next_image())
