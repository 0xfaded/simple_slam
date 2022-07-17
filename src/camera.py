class Camera:

    def __init__(self):
        self.cam_matrix = None

    def next_image(self):
        raise NotImplementedError
