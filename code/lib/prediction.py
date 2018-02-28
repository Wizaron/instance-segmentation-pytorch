import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import SpectralClustering

from utils import ImageUtilities


class Prediction(object):

    def __init__(self, resize_height, resize_width, mean,
                 std, use_coordinates, model, n_workers):

        self.normalizer = ImageUtilities.image_normalizer(mean, std)
        self.use_coordinates = use_coordinates

        self.resize_height = resize_height
        self.resize_width = resize_width
        self.model = model

        self.n_workers = n_workers

        self.img_resizer = ImageUtilities.image_resizer(
            self.resize_height, self.resize_width)

        if self.use_coordinates:
            self.coordinate_adder = ImageUtilities.coordinate_adder(
                self.resize_height, self.resize_width)

    def get_image(self, image_path):

        img = ImageUtilities.read_image(image_path)
        image_width, image_height = img.size

        img = self.img_resizer(img)
        img = self.normalizer(img)

        return img, image_height, image_width

    def get_annotation(self, annotation_path):

        img = ImageUtilities.read_image(annotation_path)
        return img

    def upsample_prediction(self, prediction, image_height, image_width):

        return cv2.resize(prediction, (image_width, image_height),
                          interpolation=cv2.INTER_NEAREST)

    def cluster(self, fg_seg_prediction, ins_seg_prediction,
                n_objects_prediction):

        seg_height, seg_width = ins_seg_prediction.shape[1:]

        fg_seg_prediction = fg_seg_prediction.cpu().numpy()
        fg_seg_prediction = fg_seg_prediction.argmax(0).astype(np.uint8)

        embeddings = ImageUtilities.coordinate_adder(
            seg_height, seg_width)(ins_seg_prediction)
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings.transpose(1, 2, 0)  # h, w, c

        n_objects_prediction = n_objects_prediction.cpu().numpy()[0]

        embeddings = np.stack([embeddings[:, :, i][fg_seg_prediction == 1]
                               for i in range(embeddings.shape[2])], axis=1)

        clustering = SpectralClustering(n_clusters=n_objects_prediction,
                                        eigen_solver=None, random_state=None,
                                        n_init=10, gamma=1.0, affinity='rbf',
                                        n_neighbors=10, eigen_tol=0.0,
                                        assign_labels='discretize', degree=3,
                                        coef0=1,
                                        kernel_params=None,
                                        n_jobs=self.n_workers).fit(embeddings)

        labels = clustering.labels_

        instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

        fg_coords = np.where(fg_seg_prediction == 1)
        for si in range(len(fg_coords[0])):
            y_coord = fg_coords[0][si]
            x_coord = fg_coords[1][si]
            _label = labels[si] + 1
            instance_mask[y_coord, x_coord] = _label

        return fg_seg_prediction, instance_mask, n_objects_prediction

    def predict(self, image_path):

        image, image_height, image_width = self.get_image(image_path)
        image = image.unsqueeze(0)

        fg_seg_prediction, ins_seg_prediction, n_objects_prediction = \
            self.model.predict(image)

        fg_seg_prediction = fg_seg_prediction.squeeze(0)
        ins_seg_prediction = ins_seg_prediction.squeeze(0)
        n_objects_prediction = n_objects_prediction.squeeze(0)

        fg_seg_prediction, ins_seg_prediction, \
            n_objects_prediction = self.cluster(fg_seg_prediction,
                                                ins_seg_prediction,
                                                n_objects_prediction)

        fg_seg_prediction = self.upsample_prediction(
            fg_seg_prediction, image_height, image_width)
        ins_seg_prediction = self.upsample_prediction(
            ins_seg_prediction, image_height, image_width)

        raw_image_pil = ImageUtilities.read_image(image_path)
        raw_image = np.array(raw_image_pil)

        return raw_image, fg_seg_prediction, ins_seg_prediction, \
            n_objects_prediction
