import os, glob
import numpy as np
from PIL import Image
from utils import create_dataset

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
SEMANTIC_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP', 'semantic-annotations')
INSTANCE_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP', 'instance-annotations')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'CVPPP', 'CVPPP2017_LSC_training', 'training', 'A1')
OUT_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP', 'lmdb')

try:
    os.makedirs(OUT_DIR)
except:
    pass

for subset in ['training', 'validation']:
    lst_filepath = os.path.join(DATA_DIR, 'metadata', 'CVPPP', subset + '.lst')
    lst = np.loadtxt(lst_filepath, dtype='str', delimiter=' ')

    img_paths = []; ins_ann_paths = []; semantic_ann_paths = []
    for image_name in lst:
        img_path = os.path.join(IMG_DIR, image_name + '_rgb.png')
        ins_ann_path = os.path.join(INSTANCE_ANN_DIR, image_name + '.npy')
        sem_ann_path = os.path.join(SEMANTIC_ANN_DIR, image_name + '.npy')

        if os.path.isfile(img_path) and os.path.isfile(ins_ann_path) and os.path.isfile(sem_ann_path):
            img_paths.append(img_path)
            ins_ann_paths.append(ins_ann_path)
            semantic_ann_paths.append(sem_ann_path)

    out_path = os.path.join(OUT_DIR, '{}-lmdb'.format(subset))

    create_dataset(out_path, img_paths, semantic_ann_paths, ins_ann_paths)
