import os
import glob
import numpy as np
from PIL import Image
from utils import create_dataset

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
                                        os.path.pardir, os.path.pardir))
SEMANTIC_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'cityscapes',
                                'semantic-annotations')
INSTANCE_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'cityscapes',
                                'instance-annotations')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'cityscapes', 'leftImg8bit')
OUT_DIR = os.path.join(DATA_DIR, 'processed', 'cityscapes', 'lmdb')

SUBSETS = ['train', 'val']
SUBSET_NAMES = ['training', 'validation']

try:
    os.makedirs(OUT_DIR)
except BaseException:
    pass

for _i, subset in enumerate(SUBSETS):
    semantic_ann_paths_all = glob.glob(
        os.path.join(SEMANTIC_ANN_DIR, subset, '*.npy'))
    semantic_ann_paths, instance_ann_paths, image_paths = [], [], []

    for f in semantic_ann_paths_all:
        name = os.path.splitext(os.path.basename(f))[0]
        _dir = name.split('_')[0]

        semantic_ann_path = f
        instance_ann_path = os.path.join(
            INSTANCE_ANN_DIR, subset, name + '.npy')
        img_path = os.path.join(
            IMG_DIR, subset, _dir, name + '_leftImg8bit.png')

        # if np.load(instance_ann_path).shape[-1] > 20:
        #    continue

        semantic_ann_paths.append(semantic_ann_path)
        instance_ann_paths.append(instance_ann_path)
        image_paths.append(img_path)

    out_path = os.path.join(OUT_DIR, '{}-lmdb'.format(SUBSET_NAMES[_i]))

    create_dataset(out_path, image_paths, semantic_ann_paths,
                   instance_ann_paths)
