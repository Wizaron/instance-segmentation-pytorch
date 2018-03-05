import argparse
import random
import os
import getpass
import datetime
import shutil
import numpy as np
import torch
from lib import SegDataset, Model, AlignCollate

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='',
                    help="Filepath of trained model (to continue training) \
                         [Default: '']")
parser.add_argument('--usegpu', action='store_true',
                    help='Enables cuda to train on gpu [Default: False]')
parser.add_argument('--nepochs', type=int, default=600,
                    help='Number of epochs to train for [Default: 600]')
parser.add_argument('--batchsize', type=int,
                    default=2, help='Batch size [Default: 2]')
parser.add_argument('--debug', action='store_true',
                    help='Activates debug mode [Default: False]')
parser.add_argument('--nworkers', type=int,
                    help='Number of workers for data loading \
                        (0 to do it using main process) [Default : 2]',
                    default=2)
parser.add_argument('--dataset', type=str,
                    help='Name of the dataset: "cityscapes" or "CVPPP"',
                    required=True)
opt = parser.parse_args()

assert opt.dataset in ['cityscapes', 'CVPPP']

if opt.dataset == 'cityscapes':
    from settings import CityscapesTrainingSettings
    ts = CityscapesTrainingSettings()
elif opt.dataset == 'CVPPP':
    from settings import CVPPPTrainingSettings
    ts = CVPPPTrainingSettings()


def generate_run_id():

    username = getpass.getuser()

    now = datetime.datetime.now()
    date = map(str, [now.year, now.month, now.day])
    coarse_time = map(str, [now.hour, now.minute])
    fine_time = map(str, [now.second, now.microsecond])

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time),
                       username, '-'.join(fine_time)])
    return run_id


RUN_ID = generate_run_id()
model_save_path = os.path.abspath(os.path.join(os.path.abspath(__file__),
                                               os.path.pardir, os.path.pardir,
                                               'models', opt.dataset, RUN_ID))
os.makedirs(model_save_path)

CODE_BASE_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), os.path.pardir))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'settings'),
                os.path.join(model_save_path, 'settings'))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'lib'),
                os.path.join(model_save_path, 'lib'))

if torch.cuda.is_available() and not opt.usegpu:
    print 'WARNING: You have a CUDA device, so you should probably \
        run with --usegpu'

# Load Seeds
random.seed(ts.SEED)
np.random.seed(ts.SEED)
torch.manual_seed(ts.SEED)

# Define Data Loaders
pin_memory = False
if opt.usegpu:
    pin_memory = True

train_dataset = SegDataset(ts.TRAINING_LMDB)
assert train_dataset

train_align_collate = AlignCollate('training', ts.N_CLASSES, ts.MAX_N_OBJECTS, ts.MEAN,
                                   ts.STD, ts.IMAGE_HEIGHT, ts.IMAGE_WIDTH,
                                   random_hor_flipping=ts.HORIZONTAL_FLIPPING,
                                   random_ver_flipping=ts.VERTICAL_FLIPPING,
                                   random_90x_rotation=ts.ROTATION_90X,
                                   random_rotation=ts.ROTATION,
                                   random_color_jittering=ts.COLOR_JITTERING,
                                   use_coordinates=ts.USE_COORDINATES)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=opt.batchsize,
                                           shuffle=True,
                                           num_workers=opt.nworkers,
                                           pin_memory=pin_memory,
                                           collate_fn=train_align_collate)

test_dataset = SegDataset(ts.VALIDATION_LMDB)
assert test_dataset

test_align_collate = AlignCollate('test', ts.N_CLASSES, ts.MAX_N_OBJECTS, ts.MEAN, ts.STD,
                                  ts.IMAGE_HEIGHT, ts.IMAGE_WIDTH,
                                  random_hor_flipping=ts.HORIZONTAL_FLIPPING,
                                  random_ver_flipping=ts.VERTICAL_FLIPPING,
                                  random_90x_rotation=ts.ROTATION_90X,
                                  random_rotation=ts.ROTATION,
                                  random_color_jittering=ts.COLOR_JITTERING,
                                  use_coordinates=ts.USE_COORDINATES)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=opt.batchsize,
                                          shuffle=False,
                                          num_workers=opt.nworkers,
                                          pin_memory=pin_memory,
                                          collate_fn=test_align_collate)

# Define Model
model = Model(opt.dataset, ts.N_CLASSES, ts.MAX_N_OBJECTS,
              use_instance_segmentation=ts.USE_INSTANCE_SEGMENTATION,
              use_coords=ts.USE_COORDINATES, load_model_path=opt.model,
              usegpu=opt.usegpu)

# Train Model
model.fit(ts.CRITERION, ts.DELTA_VAR, ts.DELTA_DIST, ts.NORM, ts.LEARNING_RATE,
          ts.WEIGHT_DECAY, ts.CLIP_GRAD_NORM, ts.LR_DROP_FACTOR,
          ts.LR_DROP_PATIENCE, ts.OPTIMIZE_BG, ts.OPTIMIZER, ts.TRAIN_CNN,
          opt.nepochs, ts.CLASS_WEIGHTS, train_loader, test_loader,
          model_save_path, opt.debug)
