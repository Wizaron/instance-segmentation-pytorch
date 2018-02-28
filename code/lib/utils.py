from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

from preprocess import RandomResizedCrop, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomTranspose, RandomRotation90x, \
    RandomRotation, AddCoordinates


class ImageUtilities(object):

    @staticmethod
    def read_image(image_path):
        img = Image.open(image_path)
        img_copy = img.copy()
        img.close()
        return img_copy

    @staticmethod
    def image_resizer(height, width, interpolation=Image.BILINEAR):
        return transforms.Resize((height, width), interpolation=interpolation)

    @staticmethod
    def image_random_cropper_and_resizer(
            height, width, interpolation=Image.BILINEAR):
        return RandomResizedCrop(height, width, interpolation=interpolation)

    @staticmethod
    def image_random_horizontal_flipper():
        return RandomHorizontalFlip()

    @staticmethod
    def image_random_vertical_flipper():
        return RandomVerticalFlip()

    @staticmethod
    def image_random_transposer():
        return RandomTranspose()

    @staticmethod
    def image_random_90x_rotator():
        return RandomRotation90x()

    @staticmethod
    def image_random_rotator(expand=False, center=None):
        return RandomRotation(expand=expand, center=center)

    @staticmethod
    def image_random_color_jitter(
            brightness=0, contrast=0, saturation=0, hue=0):
        return transforms.ColorJitter(brightness, contrast, saturation, hue)

    @staticmethod
    def image_random_grayscaler(prob=0.5):
        return transforms.RandomGrayscale(p=prob)

    @staticmethod
    def image_normalizer(mean, std):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    @staticmethod
    def coordinate_adder(height, width):
        return AddCoordinates(height, width)
