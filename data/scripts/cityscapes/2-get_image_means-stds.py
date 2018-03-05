import os, glob
import numpy as np
from PIL import Image

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
SEMANTIC_ANN_DIR = os.path.join(DATA_DIR, 'processed', 'cityscapes', 'semantic-annotations', 'train')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'cityscapes', 'leftImg8bit', 'train')

image_list = glob.glob(os.path.join(IMG_DIR, '*', '*.png'))

reds, greens, blues = [], [], []
for image_path in image_list:
    image_name = os.path.splitext(os.path.basename(image_path))[0].split('_leftImg8bit')[0]
    semantic_ann_path = os.path.join(SEMANTIC_ANN_DIR, image_name + '.npy')

    if not os.path.isfile(semantic_ann_path):
        continue

    img = np.array(Image.open(image_path))
    r, g, b = np.split(img, 3, axis=2)

    r = r.flatten().mean()
    g = g.flatten().mean()
    b = b.flatten().mean()

    reds.append(r)
    greens.append(g)
    blues.append(b)

reds = np.array(reds)
greens = np.array(greens)
blues = np.array(blues)

red_mean = np.mean(reds) / 255.
green_mean = np.mean(greens) / 255.
blue_mean = np.mean(blues) / 255.

red_std = np.std(reds) / 255.
green_std = np.std(greens) / 255.
blue_std = np.std(blues) / 255.

print red_mean, green_mean, blue_mean
print red_std, green_std, blue_std
