import os, glob
import numpy as np
from PIL import Image
from utils import create_dataset

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'CVPPP2017_LSC_training', 'training', 'A1')
OUT_DIR = os.path.join(DATA_DIR, 'processed', 'lmdb')

try:
    os.makedirs(OUT_DIR)
except:
    pass

for subset in ['training', 'validation']:
    lst_filepath = os.path.join(DATA_DIR, 'metadata', subset + '.lst')
    lst = np.loadtxt(lst_filepath, dtype='str', delimiter=' ')

    img_paths = []; ann_paths = []
    for image_name in lst:
        img_path = os.path.join(IMG_DIR, image_name + '_rgb.png')
        ann_path = os.path.join(ANN_DIR, image_name + '.npy')

        if os.path.isfile(img_path) and os.path.isfile(ann_path):
            img_paths.append(img_path)
            ann_paths.append(ann_path)

    out_path = os.path.join(OUT_DIR, '{}-lmdb'.format(subset))

    create_dataset(out_path, img_paths, ann_paths)
