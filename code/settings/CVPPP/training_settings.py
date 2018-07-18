import os
from model_settings import ModelSettings


class TrainingSettings(ModelSettings):

    def __init__(self):
        super(TrainingSettings, self).__init__()

        self.TRAINING_LMDB = os.path.join(
            self.BASE_PATH,
            'data',
            'processed',
            'CVPPP',
            'lmdb',
            'training-lmdb')
        self.VALIDATION_LMDB = os.path.join(
            self.BASE_PATH,
            'data',
            'processed',
            'CVPPP',
            'lmdb',
            'validation-lmdb')

        self.TRAIN_CNN = True

        self.OPTIMIZER = 'Adadelta'
        # optimizer - one of : 'RMSprop', 'Adam', 'Adadelta', 'SGD'
        self.LEARNING_RATE = 1.0
        self.LR_DROP_FACTOR = 0.1
        self.LR_DROP_PATIENCE = 20
        self.WEIGHT_DECAY = 0.001
        # weight decay - use 0 to disable it
        self.CLIP_GRAD_NORM = 10.0
        # max l2 norm of gradient of parameters - use 0 to disable it

        self.HORIZONTAL_FLIPPING = False
        self.VERTICAL_FLIPPING = False
        self.TRANSPOSING = False
        self.ROTATION_90X = False
        self.ROTATION = False
        self.COLOR_JITTERING = False
        self.GRAYSCALING = False
        self.CHANNEL_SWAPPING = False
        self.GAMMA_ADJUSTMENT = False
        self.RESOLUTION_DEGRADING = False

        self.CRITERION = 'Multi'
        # criterion - One of 'CE', 'Dice', 'Multi'
        self.OPTIMIZE_BG = False

        # self.RANDOM_CROPPING = False
        # CROP_SCALE and CROP_AR is used iff self.RANDOM_CROPPING is True
        # self.CROP_SCALE = (1.0, 1.0)
        # Choose it carefully - have a look at
        # lib/preprocess.py -> RandomResizedCrop
        # self.CROP_AR = (1.0, 1.0)
        # Choose it carefully - have a look
        # at lib/preprocess.py -> RandomResizedCrop

        self.SEED = 23
