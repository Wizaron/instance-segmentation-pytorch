import torch
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import random
import math
import numbers
import collections
import numpy as np


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
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
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
        size (sequence or int): Desired output size. Same semantics as ``scale``.
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
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
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
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
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


### HORIZONTAL FLIPPING ###

def hflip(img):
    """Horizontally flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image: Horizontally flipped image.
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
    """Horizontally flip the given PIL Image."""

    def __call__(self, img, flip):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Flipped image.
        """
        if flip:
            return hflip(img)
        return img


### VERTICAL FLIPPING ###

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
    """Vertically flip the given PIL Image."""

    def __call__(self, img, flip):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Flipped image.
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
    """Transpose the given PIL Image."""

    def __call__(self, img, trans):
        """
        Args:
            img (PIL Image): Image to be transposed.
        Returns:
            PIL Image: Transposed image.
        """
        if trans:
            return transpose(img)
        return img


### RANDOM ROTATION ###

def rotate(img, angle, resample=Image.BILINEAR, expand=True):

    is_numpy = isinstance(img, np.ndarray)

    if not _is_pil_image(img):
        if is_numpy:
            img = Image.fromarray(img)
        else:
            raise TypeError(
                'img should be PIL Image or numpy array. \
                Got {}'.format(type(img)))

    img = img.rotate(angle, resample=resample, expand=expand)

    if is_numpy:
        img = np.array(img)

    return img

def rotate_with_random_bg(img, angle, resample=Image.BILINEAR, expand=True):

    is_numpy = isinstance(img, np.ndarray)

    if not _is_pil_image(img):
        if is_numpy:
            img = Image.fromarray(img)
        else:
            raise TypeError(
                'img should be PIL Image or numpy array. \
                Got {}'.format(type(img)))

    img_np = np.array(img)

    img = img.convert('RGBA')
    img = rotate(img, angle, resample=resample, expand=expand)

    key = np.random.choice([0, 1, 2, 3])
    if key == 0:
        bg = Image.new('RGBA', img.size, (255, ) * 4)    # White image
    elif key == 1:
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255)) # Black image
    elif key == 2:
        mean_color = map(int, img_np.mean((0, 1)))
        bg = Image.new('RGBA', img.size, (mean_color[0], mean_color[1], mean_color[2], 255)) # Mean
    elif key == 3:
        median_color = map(int, np.median(img_np, (0, 1)))
        bg = Image.new('RGBA', img.size, (median_color[0], median_color[1], median_color[2], 255)) # Median

    img = Image.composite(img, bg, img)
    img = img.convert('RGB')

    if is_numpy:
        img = np.array(img)

    return img

class RandomRotate(object):

    def __init__(self, interpolation=Image.BILINEAR, random_bg=True):
        self.interpolation = interpolation
        self.random_bg = random_bg

    def __call__(self, img, angle, expand):
        if self.random_bg:
            return rotate_with_random_bg(img, angle, resample=self.interpolation, expand=expand)
        else:
            return rotate(img, angle, resample=self.interpolation, expand=expand)

### RANDOM CHANNEL SWAPING ###

def swap_channels(img):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img_np = np.array(img)

    channel_idxes = np.random.choice([0, 1, 2], 3, True)

    return Image.fromarray(img_np[:, :, channel_idxes])

class RandomChannelSwap(object):

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() >= self.prob:
            return img

        return swap_channels(img)

### GAMMA CORRECTION ###

def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
    See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    return img

class RandomGamma(object):

    def __init__(self, gamma_range, gain=1):
        self.min_gamma = gamma_range[0]
        self.max_gamma = gamma_range[1]

        self.gain = gain

    def __call__(self, img):
        gamma = np.random.rand() * (self.max_gamma - self.min_gamma) + self.min_gamma
        return adjust_gamma(img, gamma=gamma, gain=self.gain)

### RESOLUTION ###

def random_resolution(img, ratio):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img_size = np.array(img.size)
    new_size = (img_size * ratio).astype('int')

    img = img.resize(new_size, Image.ANTIALIAS)
    img = img.resize(img_size, Image.ANTIALIAS)

    return img

class RandomResolution(object):

    def __init__(self, ratio_range):
        self.ratio_range = np.arange(ratio_range[0], ratio_range[1], 0.05)

    def __call__(self, img):
        _range = np.random.choice(self.ratio_range)
        return random_resolution(img, _range)
