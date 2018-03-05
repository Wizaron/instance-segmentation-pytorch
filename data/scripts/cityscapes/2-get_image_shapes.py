import os
import glob
import numpy as np
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
                                        os.path.pardir, os.path.pardir))
SEMANTIC_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'cityscapes',
                                'semantic-annotations', 'train')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'cityscapes', 'leftImg8bit', 'train')
OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata', 'cityscapes')

image_shapes = []
image_paths = glob.glob(os.path.join(IMG_DIR, '*', '*.png'))
for image_path in image_paths:
    image_name = os.path.splitext(os.path.basename(image_path))[
        0].split('_leftImg8bit')[0]
    semantic_ann_path = os.path.join(SEMANTIC_ANN_DIR, image_name + '.npy')

    if not os.path.isfile(semantic_ann_path):
        continue

    img_size = Image.open(image_path).size
    img_size = (img_size[1], img_size[0])  # height, width

    image_shapes.append([image_name, img_size[0], img_size[1]])

np.savetxt(
    os.path.join(
        OUTPUT_DIR,
        'train-image_shapes.txt'),
    image_shapes,
    fmt='%s',
    delimiter=',')
