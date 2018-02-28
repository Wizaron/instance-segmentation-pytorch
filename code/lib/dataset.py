import torch
from torch.utils.data import Dataset
import random

from PIL import Image
import lmdb
import sys
import numpy as np
from StringIO import StringIO

from utils import ImageUtilities as IU


class SegDataset(Dataset):
    """Dataset Reader"""

    def __init__(self, lmdb_path):

        self._lmdb_path = lmdb_path

        self.env = lmdb.open(self._lmdb_path, max_readers=1,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        if not self.env:
            print 'Cannot read lmdb from {}'.format(self._lmdb_path)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get('num-samples'))

    def __load_data(self, index):

        with self.env.begin(write=False) as txn:
            image_key = 'image-{}'.format(index + 1)
            ann_key = 'annotation-{}'.format(index + 1)
            height_key = 'height-{}'.format(index + 1)
            width_key = 'width-{}'.format(index + 1)
            n_objects_key = 'n_objects-{}'.format(index + 1)

            img = txn.get(image_key)
            img = Image.open(StringIO(img))

            height = int(txn.get(height_key))
            width = int(txn.get(width_key))
            n_objects = int(txn.get(n_objects_key))

            annotation = np.fromstring(txn.get(ann_key),
                                       dtype=np.uint8)
            annotation = annotation.reshape(height, width,
                                            n_objects)

        return img, annotation, n_objects

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        image, annotation, n_objects \
            = self.__load_data(index)

        return image, annotation, \
            n_objects

    def __len__(self):
        return self.n_samples


class AlignCollate(object):

    def __init__(self, mode, max_n_objects, mean, std, image_height,
                 image_width, random_hor_flipping=True,
                 random_ver_flipping=True,
                 random_90x_rotation=True, random_rotation=True,
                 random_color_jittering=True, use_coordinates=False):

        self._mode = mode
        self.max_n_objects = max_n_objects

        assert self._mode in ['training', 'test']

        self.mean = mean
        self.std = std
        self.image_height = image_height
        self.image_width = image_width

        self.random_horizontal_flipping = random_hor_flipping
        self.random_vertical_flipping = random_ver_flipping
        self.random_90x_rotation = random_90x_rotation
        self.random_rotation = random_rotation
        self.random_color_jittering = random_color_jittering

        self.use_coordinates = use_coordinates

        if self._mode == 'training':
            if self.random_horizontal_flipping:
                self.horizontal_flipper = IU.image_random_horizontal_flipper()
            if self.random_vertical_flipping:
                self.vertical_flipper = IU.image_random_vertical_flipper()
            if self.random_90x_rotation:
                self.rotator_90x = IU.image_random_90x_rotator()
            if self.random_rotation:
                self.rotator = IU.image_random_rotator(expand=True)
            if self.random_color_jittering:
                self.color_jitter = IU.image_random_color_jitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

            self.img_resizer = IU.image_resizer(self.image_height,
                                                self.image_width)
            self.ann_resizer = IU.image_resizer(self.image_height,
                                                self.image_width,
                                                interpolation=Image.NEAREST)
        else:
            self.img_resizer = IU.image_resizer(self.image_height,
                                                self.image_width)
            self.ann_resizer = IU.image_resizer(self.image_height,
                                                self.image_width,
                                                interpolation=Image.NEAREST)

        self.image_normalizer = IU.image_normalizer(self.mean, self.std)

        if self.use_coordinates:
            self.coordinate_adder = IU.coordinate_adder(
                self.image_height, self.image_width)

    def __preprocess(self, image, annotation):

        # Augmentation
        if self._mode == 'training':
            annotation = list(annotation.transpose(2, 0, 1))
            n_objects = len(annotation)

            if self.random_color_jittering:
                image = self.color_jitter(image)

            if self.random_horizontal_flipping:
                is_flip = random.random() < 0.5
                image = self.horizontal_flipper(image, is_flip)

                for i in range(n_objects):
                    _ann = annotation[i].copy()
                    _ann = self.horizontal_flipper(_ann, is_flip)
                    annotation[i] = _ann

            if self.random_vertical_flipping:
                is_flip = random.random() < 0.5
                image = self.vertical_flipper(image, is_flip)

                for i in range(n_objects):
                    _ann = annotation[i].copy()
                    _ann = self.vertical_flipper(_ann, is_flip)
                    annotation[i] = _ann

            if self.random_90x_rotation:
                n_rot = np.random.choice([0, 1, 2, 3])
                image = self.rotator_90x(image, n_rot)

                for i in range(n_objects):
                    _ann = annotation[i].copy()
                    _ann = self.rotator_90x(_ann, n_rot)
                    annotation[i] = _ann

            if self.random_rotation:
                angle = self.rotator.get_params(10)
                image = self.rotator(image, angle, Image.BILINEAR)

                for i in range(n_objects):
                    _ann = annotation[i].copy()
                    _ann = self.rotator(_ann, angle, Image.NEAREST)
                    annotation[i] = _ann

            annotation = np.array(annotation).transpose(1, 2, 0)

        # Resize Images
        image = self.img_resizer(image)

        # Resize Annotations
        ann_height, ann_width, n_objects = annotation.shape
        annotation_resized = []

        height_ratio = 1.0 * self.image_height / ann_height
        width_ratio = 1.0 * self.image_width / ann_width

        for i in range(n_objects):
            ann_img = Image.fromarray(annotation[:, :, i])
            ann_img = self.ann_resizer(ann_img)
            ann_img = np.array(ann_img)

            annotation_resized.append(ann_img)

        # Fill Annotations with zeros
        for i in range(self.max_n_objects - n_objects):
            zero = np.zeros((ann_height, ann_width),
                            dtype=np.uint8)
            zero = Image.fromarray(zero)
            zero = self.ann_resizer(zero)
            zero = np.array(zero)
            annotation_resized.append(zero.copy())

        annotation_resized = np.stack(annotation_resized, axis=0)
        annotation_resized = annotation_resized.transpose(1, 2, 0)

        # Image Normalization
        image = self.image_normalizer(image)

        if self.use_coordinates:
            image = self.coordinate_adder(image)

        return (image, annotation_resized)

    def __call__(self, batch):
        images, annotations, \
            n_objects = zip(*batch)

        images = list(images)
        annotations = list(annotations)

        # max_n_objects = np.max(n_objects)

        bs = len(images)
        for i in range(bs):
            image, annotation = self.__preprocess(images[i],
                                                  annotations[i])

            images[i] = image
            annotations[i] = annotation

        images = torch.stack(images)

        annotations = np.array(annotations, dtype='int')  # bs, h, w, n_ins

        annotations_for_fg = annotations.sum(3)
        annotations_for_fg[annotations_for_fg != 0] = 1
        fg_annotations = np.eye(2, dtype='int')
        fg_annotations = \
            fg_annotations[annotations_for_fg.flatten()].reshape(annotations.shape[0],
                                                                 annotations.shape[1],
                                                                 annotations.shape[2],
                                                                 2)

        annotations = torch.LongTensor(annotations)
        annotations = annotations.permute(0, 3, 1, 2)

        fg_annotations = torch.LongTensor(fg_annotations)
        fg_annotations = fg_annotations.permute(0, 3, 1, 2)

        n_objects = torch.IntTensor(n_objects)

        return (images, fg_annotations, annotations,
                n_objects)


if __name__ == '__main__':
    ds = SegDataset('../../../data/processed/lmdb/training-lmdb/')
    image, image_rgb, annotation, box_annotation, box_coordinate, \
        n_objects = ds[5]

    print image.size
    print image_rgb.size
    print annotation.shape
    print box_annotation.shape
    print box_coordinate.shape
    print n_objects
    print np.unique(annotation)
    print np.unique(box_annotation)

    ac = AlignCollate([0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                      256, 512, 256, 512)

    loader = torch.utils.data.DataLoader(ds, batch_size=9,
                                         shuffle=True,
                                         num_workers=1,
                                         pin_memory=False,
                                         collate_fn=ac)
    loader = iter(loader)

    images, images_rgb, annotations, box_annotations, \
        box_coordinates, n_objects = loader.next()

    print images.size()
    print images_rgb.size()
    print annotations.size()
    print box_annotations.size()
    print box_coordinates.size()
    print box_coordinates
    print n_objects.size()
    print n_objects
