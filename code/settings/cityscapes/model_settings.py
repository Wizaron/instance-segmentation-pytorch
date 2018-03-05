import os
import numpy as np
from data_settings import DataSettings


class ModelSettings(DataSettings):

    def __init__(self):
        super(ModelSettings, self).__init__()

        # self.MEAN = [0.485, 0.456, 0.406]
        # self.STD = [0.229, 0.224, 0.225]
        self.MEAN = [0.28434713162, 0.323142312507, 0.281661828689]
        self.STD = [0.0466890583142, 0.0425745389885, 0.040883738302]

        self.USE_INSTANCE_SEGMENTATION = True
        self.USE_COORDINATES = False

        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 512

        self.DELTA_VAR = 0.5
        self.DELTA_DIST = 1.5
        self.NORM = 2
