import os, sys
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='path of the image')
parser.add_argument('--model', required=True, help='path of the model')
parser.add_argument('--usegpu', action='store_true', help='enables cuda to predict on gpu')
parser.add_argument('--output', required=True, help='path of the output directory')
parser.add_argument('--n_workers', default=1, type=int, help='number of workers for clustering')
opt = parser.parse_args()

image_path = opt.image
model_path = opt.model
output_path = opt.output

try:
    os.makedirs(output_path)
except:
    pass

model_dir = os.path.dirname(model_path)
sys.path.insert(0, model_dir)

from lib import Model, Prediction
from settings import ModelSettings

ms = ModelSettings()

model = Model(ms.MAX_N_OBJECTS, use_instance_segmentation=ms.USE_INSTANCE_SEGMENTATION,
              use_coords=ms.USE_COORDINATES, load_model_path=opt.model, usegpu=opt.usegpu)

prediction = Prediction(ms.IMAGE_HEIGHT, ms.IMAGE_WIDTH, ms.MEAN, ms.STD, ms.USE_COORDINATES, model, opt.n_workers)
image, fg_seg_pred, ins_seg_pred, n_objects_pred = prediction.predict(image_path)

fg_seg_pred = fg_seg_pred * 255

_n_clusters = len(np.unique(ins_seg_pred.flatten())) - 1 #discard bg
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
ins_seg_pred_color = np.zeros((ins_seg_pred.shape[0], ins_seg_pred.shape[1], 3), dtype=np.uint8)
for i in range(_n_clusters):
    ins_seg_pred_color[ins_seg_pred == (i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')

image_name = os.path.splitext(os.path.basename(image_path))[0]

image_pil = Image.fromarray(image)
fg_seg_pred_pil = Image.fromarray(fg_seg_pred)
ins_seg_pred_pil = Image.fromarray(ins_seg_pred)
ins_seg_pred_color_pil = Image.fromarray(ins_seg_pred_color)

image_pil.save(os.path.join(output_path, image_name + '.png'))
fg_seg_pred_pil.save(os.path.join(output_path, image_name + '-fg_mask.png'))
ins_seg_pred_pil.save(os.path.join(output_path, image_name + '-ins_mask.png'))
ins_seg_pred_color_pil.save(os.path.join(output_path, image_name + '-ins_mask_color.png'))
np.save(os.path.join(output_path, image_name + '-n_objects.npy'), n_objects_pred)
