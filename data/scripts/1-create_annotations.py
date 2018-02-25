import os, glob, cv2
from PIL import Image
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'raw', 'CVPPP2017_LSC_training', 'training', 'A1')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'CVPPP2017_LSC_training', 'training', 'A1')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')

try:
    os.makedirs(OUTPUT_DIR)
except:
    pass

image_paths = glob.glob(os.path.join(IMG_DIR, '*_rgb.png'))

for image_path in image_paths:
    img = Image.open(image_path)
    img_width, img_height = img.size

    image_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
    annotation_path = os.path.join(ANN_DIR, image_name + '_label.png')

    if not os.path.isfile(annotation_path):
        continue

    annotation = np.array(Image.open(annotation_path))

    assert len(annotation.shape) == 2
    assert np.array(img).shape[:2] == annotation.shape[:2]

    instance_values = set(np.unique(annotation)).difference([0])
    n_instances = len(instance_values)

    if n_instances == 0:
        continue

    mask = np.zeros((img_height, img_width, n_instances), dtype=np.uint8)

    for i, v in enumerate(instance_values):
        _mask = np.zeros((img_height, img_width), dtype=np.uint8)
        _mask[annotation == v] = 1
        mask[:, :, i] = _mask

    np.save(os.path.join(OUTPUT_DIR, image_name + '.npy'), mask)
