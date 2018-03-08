import os
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir,
                                        os.path.pardir, os.path.pardir))
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'CVPPP', 'CVPPP2017_LSC_training',
                       'training', 'A1')
METADATA_OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata', 'CVPPP')

SUBSETS = ['train', 'val']
SUBSET_NAMES= ['training', 'validation']

for si, subset in enumerate(SUBSETS):
    lst = os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[si] + '.lst')
    image_names = np.loadtxt(lst, dtype='str', delimiter=',')

    image_paths = []
    for image_name in image_names:
        _dir = image_name.split('_')[0]
        image_path = os.path.join(IMG_DIR, image_name + '_rgb.png')
        image_paths.append(image_path)

    np.savetxt(os.path.join(METADATA_OUTPUT_DIR, SUBSET_NAMES[si] + '_image_paths.txt'),
               image_paths, fmt='%s', delimiter=',')
