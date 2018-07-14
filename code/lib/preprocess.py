import torch
from PIL import Image
import numbers
import numpy as np
try:
    import accimage
except ImportError:
    accimage = None
import random
import math
import collections


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number
            maintaing the aspect ratio. i.e, if height > width, then image will
            be rescaled to (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.
    Notably used in RandomResizedCrop.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics
            as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and
    a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size_height, size_width, interpolation=Image.BILINEAR):
        self.size = (size_height, size_width)
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio
                cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, params):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly cropped and resize image.
        """
        i, j, h, w = params
        return resized_crop(img, i, j, h, w, self.size, self.interpolation)

# ROTATION #


def rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): In degrees degrees counter clockwise order.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC},
            optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to
            PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold
            the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and
            no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """
    is_numpy = isinstance(img, np.ndarray)

    if not _is_pil_image(img):
        if is_numpy:
            img = Image.fromarray(img)
        else:
            raise TypeError(
                'img should be PIL Image or numpy array. \
                Got {}'.format(type(img)))

    img = img.rotate(angle, resample, expand, center)

    if is_numpy:
        img = np.array(img)

    return img


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the
            range of degrees will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC},
            optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to
            PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the
            entire rotated image.
            If false or omitted, make the output image the same size as the
            input image.
            Note that the expand flag assumes rotation around the center and
            no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, expand=False, center=None):
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            degrees = degrees

        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, angle, resample):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        return rotate(img, angle, resample, self.expand, self.center)

    # def __repr__(self):
    #    return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)


# HORIZONTAL FLIPPING #

def hflip(img):
    """Horizontally flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Horizontally flipped image.
    """

    is_numpy = isinstance(img, np.ndarray)

    if not _is_pil_image(img):
        if is_numpy:
            img = Image.fromarray(img)
        else:
            raise TypeError(
                'img should be PIL Image or numpy array. \
                Got {}'.format(type(img)))

    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if is_numpy:
        img = np.array(img)

    return img


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a
    probability of 0.5."""

    def __call__(self, img, flip):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if flip:
            return hflip(img)
        return img

# VERTICAL FLIPPING #


def vflip(img):
    """Vertically flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image: Vertically flipped image.
    """
    is_numpy = isinstance(img, np.ndarray)

    if not _is_pil_image(img):
        if is_numpy:
            img = Image.fromarray(img)
        else:
            raise TypeError(
                'img should be PIL Image or numpy array. \
                Got {}'.format(type(img)))

    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    if is_numpy:
        img = np.array(img)

    return img


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a
    probability of 0.5."""

    def __call__(self, img, flip):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if flip:
            return vflip(img)
        return img

# TRANSPOSE #


def transpose(img):
    """Transpose the given PIL Image.
    Args:
        img (PIL Image): Image to be transposed.
    Returns:
        PIL Image: Transposed image.
    """
    is_numpy = isinstance(img, np.ndarray)

    if not _is_pil_image(img):
        if is_numpy:
            img = Image.fromarray(img)
        else:
            raise TypeError(
                'img should be PIL Image or numpy array. \
                Got {}'.format(type(img)))

    img = img.transpose(Image.TRANSPOSE)

    if is_numpy:
        img = np.array(img)

    return img


class RandomTranspose(object):
    """Transpose the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img, trans):
        """
        Args:
            img (PIL Image): Image to be transposed.
        Returns:
            PIL Image: Randomly transposed image.
        """
        if trans:
            return transpose(img)
        return img

# ROTATE #


def rotate90x(img, n_rot):
    """Rotate the given PIL Image.
    Args:
        img (PIL Image): Image to be rotated.
    Returns:
        PIL Image: Rotated image.
    """
    is_numpy = isinstance(img, np.ndarray)

    if not _is_pil_image(img):
        if is_numpy:
            img = Image.fromarray(img)
        else:
            raise TypeError(
                'img should be PIL Image or numpy array. \
                Got {}'.format(type(img)))

    if n_rot == 1:
        img = img.transpose(Image.ROTATE_90)
    elif n_rot == 2:
        img = img.transpose(Image.ROTATE_180)
    elif n_rot == 3:
        img = img.transpose(Image.ROTATE_270)

    if is_numpy:
        img = np.array(img)

    return img


class RandomRotation90x(object):
    """Rotate the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img, n_rot):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Randomly rotated image.
        """
        return rotate90x(img, n_rot)

#####################


class AddCoordinates(object):

    def __init__(self, image_height, image_width):

        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image):

        y_coords = 2.0 * torch.arange(self.image_height).unsqueeze(
            1).expand(self.image_height, self.image_width) / (self.image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(self.image_width).unsqueeze(
            0).expand(self.image_height, self.image_width) / (self.image_width - 1.0) - 1.0
        coords = torch.stack((y_coords, x_coords), dim=0)

        image = torch.cat((coords, image), dim=0)

        return image
