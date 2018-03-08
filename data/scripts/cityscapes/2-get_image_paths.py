import os
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
                                        os.path.pardir, os.path.pardir))
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'cityscapes', 'leftImg8bit')
METADATA_OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata', 'cityscapes')

SUBSETS = ['train', 'val']
SUBSET_NAMES= ['training', 'validation']

for si, subset in enumerate(SUBSETS):
    lst = os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[si] + '.lst')
    image_names = np.loadtxt(lst, dtype='str', delimiter=',')
    img_dir = os.path.join(IMG_DIR, subset)

    image_paths = []
    for image_name in image_names:
        _dir = image_name.split('_')[0]
        image_path = os.path.join(img_dir, _dir, image_name + '_leftImg8bit.png')
        image_paths.append(image_path)

    np.savetxt(os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[si] + '_image_paths.txt'),
               image_paths, fmt='%s', delimiter=',')
