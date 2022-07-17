from system import System
from kitti import KittiCamera

camera = KittiCamera('/media/data/kitti/dataset/sequences/04')

simple_slam = System(camera)

simple_slam.process_image(camera.next_image())
simple_slam.process_image(camera.next_image())
