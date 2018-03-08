import os
import glob
import numpy as np
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
                                        os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP', 'instance-annotations')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'CVPPP', 'CVPPP2017_LSC_training',
                       'training', 'A1')
OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata', 'CVPPP')

annotation_files = glob.glob(os.path.join(ANN_DIR, '*.npy'))

image_shapes = []
for f in annotation_files:
    image_name = os.path.splitext(os.path.basename(f))[0]
    ann_size = np.load(f).shape[:2]
    image_path = os.path.join(IMG_DIR, image_name + '_rgb.png')
    img_size = Image.open(image_path).size
    img_size = (img_size[1], img_size[0])

    assert ann_size == img_size

    image_shapes.append([image_name, ann_size[0], ann_size[1]])

np.savetxt(os.path.join(OUTPUT_DIR, 'image_shapes.txt'), image_shapes,
           fmt='%s', delimiter=',')
