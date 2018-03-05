import os, glob
import numpy as np
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'processed', 'CVPPP', 'instance-annotations')
OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata', 'CVPPP')

annotation_files = glob.glob(os.path.join(ANN_DIR, '*.npy'))

number_of_instances = []
for f in annotation_files:
    image_name = os.path.splitext(os.path.basename(f))[0]
    n_instances = np.load(f).shape[-1]

    number_of_instances.append([image_name, n_instances])

np.savetxt(os.path.join(OUTPUT_DIR, 'number_of_instances.txt'), number_of_instances, fmt='%s', delimiter=',')
