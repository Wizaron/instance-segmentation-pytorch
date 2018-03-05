import os, glob, cv2
from PIL import Image
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
ANN_DIR = os.path.join(DATA_DIR, 'raw', 'cityscapes', 'gtFine')
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'cityscapes', 'leftImg8bit')
SEMANTIC_OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', 'cityscapes', 'semantic-annotations')
INSTANCE_OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', 'cityscapes', 'instance-annotations')
METADATA_OUTPUT_DIR = os.path.join(DATA_DIR, 'metadata', 'cityscapes')

SUBSETS = ['train', 'val']
SUBSET_NAMES = ['training', 'validation']

labels = np.loadtxt(os.path.join(DATA_DIR, 'metadata', 'cityscapes', 'labels.txt'), dtype='str', delimiter=',')
if len(labels.shape) == 1:
    labels = np.expand_dims(labels, axis=0)
db_labels = {int(k) : v for k, v in labels[:, :2]}
network_labels = {int(k) : int(v) for k, v in labels[:, [0, 2]]}

def create_network_label(db_val):

    if db_labels.has_key(db_val):
        return network_labels[db_val]
    else:
        return 0

try:
    os.makedirs(SEMANTIC_OUTPUT_DIR)
except:
    pass

try:
    os.makedirs(INSTANCE_OUTPUT_DIR)
except:
    pass

number_of_instances = []
for subset_idx, subset in enumerate(SUBSETS):
    image_paths = glob.glob(os.path.join(IMG_DIR, subset, '*', '*.png'))

    semantic_out_dir = os.path.join(SEMANTIC_OUTPUT_DIR, subset)
    instance_out_dir = os.path.join(INSTANCE_OUTPUT_DIR, subset)

    try:
        os.makedirs(semantic_out_dir)
    except:
        pass

    try:
        os.makedirs(instance_out_dir)
    except:
        pass

    image_list = []
    for image_path in image_paths:
        img = Image.open(image_path)
        img_width, img_height = img.size

        image_dir = os.path.basename(os.path.dirname(image_path))
        image_name = os.path.splitext(os.path.basename(image_path))[0].split('_leftImg8bit')[0]

        image_list.append(image_name)

        instance_annotation_path = os.path.join(ANN_DIR, subset, image_dir, image_name + '_gtFine_instanceIds.png')

        if not os.path.isfile(instance_annotation_path):
            continue

        # Instance Annotation
        db_instance_annotation = np.array(Image.open(instance_annotation_path))

        assert len(db_instance_annotation.shape) == 2
        assert np.array(img).shape[:2] == db_instance_annotation.shape[:2]

        instance_labels = np.unique(db_instance_annotation.flatten())

        discard = False
        cleaned_db_instance_annotation = []
        for instance_label in instance_labels:
           if instance_label in map(int, list(labels[:, 0])):
               _c_ins = np.zeros((img_height, img_width), dtype=np.uint8)
               _c_ins[db_instance_annotation == instance_label] = 1
               _, contours, hierarchy = cv2.findContours(_c_ins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               n_contours = len(contours)

               if n_contours != 1:
                   discard = True
                   break

               for i in range(n_contours):
                   _ins = np.zeros((img_height, img_width), dtype=np.uint8)
                   _ins = cv2.drawContours(_ins, contours, i, 1, -1)

                   _ins_sum_width = _ins.sum(0)
                   _ins_sum_height = _ins.sum(1)
                   _ins_sum_width = _ins_sum_width[_ins_sum_width != 0]
                   _ins_sum_height = _ins_sum_height[_ins_sum_height != 0]

                   if np.all(_ins_sum_width <= 10) or np.all(_ins_sum_height <= 10):
                       continue

                   _ins = _ins * int(instance_label)

                   cleaned_db_instance_annotation.append(_ins)
               #discard = True
               #break
           elif int(instance_label / 1000) in map(int, list(labels[:, 0])):
               _ins = np.zeros((img_height, img_width), dtype=np.uint8)
               #_ins[db_instance_annotation == instance_label] = int(instance_label / 1000)
               _ins[db_instance_annotation == instance_label] = 1

               _ins_sum_width = _ins.sum(0)
               _ins_sum_height = _ins.sum(1)
               _ins_sum_width = _ins_sum_width[_ins_sum_width != 0]
               _ins_sum_height = _ins_sum_height[_ins_sum_height != 0]

               if np.all(_ins_sum_width <= 10) or np.all(_ins_sum_height <= 10):
                   continue

               _ins = _ins * int(instance_label / 1000)

               cleaned_db_instance_annotation.append(_ins)

        if discard:
            continue

        n_instances = len(cleaned_db_instance_annotation)

        if n_instances == 0:
            continue

        cleaned_db_instance_annotation = np.stack(cleaned_db_instance_annotation, axis=2)

        instance_annotation = cleaned_db_instance_annotation.copy()
        instance_annotation[instance_annotation != 0] = 1

        if not np.all(np.sort(np.unique(instance_annotation.sum(2))) == np.array([0, 1])):
            continue

        semantic_annotation = cleaned_db_instance_annotation.sum(2)
        for _db, _net in network_labels.iteritems():
            semantic_annotation[semantic_annotation == _db] = _net
        semantic_annotation = semantic_annotation.astype(np.uint8)

        number_of_instances.append([image_name, n_instances])

        # Write
        np.save(os.path.join(semantic_out_dir, image_name + '.npy'), semantic_annotation)
        np.save(os.path.join(instance_out_dir, image_name + '.npy'), instance_annotation)

    np.savetxt(os.path.join(METADATA_OUTPUT_DIR, '{}.lst'.format(SUBSET_NAMES[subset_idx])), image_list, fmt='%s', delimiter=',')

np.savetxt(os.path.join(METADATA_OUTPUT_DIR, 'number_of_instances.txt'), number_of_instances, fmt='%s', delimiter=',')
