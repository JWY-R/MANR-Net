from __future__ import division
import torch
import math
import random

from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None

import numpy as np
import numbers
import types
import collections
import warnings

import scipy.ndimage.interpolation as itpl
import skimage.transform


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def adjust_brightness(img, brightness_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(
            'hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255)**gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return img


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    def __call__(self, img):
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if isinstance(img, np.ndarray):
            # handle numpy array
            if img.ndim == 3:
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
            elif img.ndim == 2:
                img = torch.from_numpy(img.copy())
            else:
                raise RuntimeError(
                    'img should be ndarray with 2 or 3 dimensions. Got {}'.
                    format(img.ndim))

            return img


class NormalizeNumpyArray(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        # TODO: make efficient
        print(img.shape)
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img


class NormalizeTensor(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        if not _is_tensor_image(tensor):
            raise TypeError('tensor is not a torch image.')
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):

        return skimage.transform.rotate(img, self.angle, resize=False, order=0)


class Resize(object):

    def __init__(self, size, interpolation='nearest'):
        assert isinstance(size, float)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        if img.ndim == 3:
            return skimage.transform.rescale(img, self.size, order=0)
        elif img.ndim == 2:
            return skimage.transform.rescale(img, self.size, order=0)
        else:
            RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))


        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))


class BottomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = h - th
        j = int(round((w - tw) / 2.))


        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = np.random.randint(0, h-th+1)
        j = np.random.randint(0, w-tw+1)

        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)
        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))


class Crop(object):

    def __init__(self, crop):
        self.crop = crop

    @staticmethod
    def get_params(img, crop):

        x_l, x_r, y_b, y_t = crop
        h = img.shape[0]
        w = img.shape[1]
        assert x_l >= 0 and x_l < w
        assert x_r >= 0 and x_r < w
        assert y_b >= 0 and y_b < h
        assert y_t >= 0 and y_t < h
        assert x_l < x_r and y_b < y_t

        return x_l, x_r, y_b, y_t

    def __call__(self, img):

        x_l, x_r, y_b, y_t = self.get_params(img, self.crop)

        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[y_b:y_t, x_l:x_r, :]
        elif img.ndim == 2:
            return img[y_b:y_t, x_l:x_r]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))


class Lambda(object):

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class HorizontalFlip(object):

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):

        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if self.do_flip:
            return np.fliplr(img)
        else:
            return img


class ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        transforms = []
        transforms.append(
            Lambda(lambda img: adjust_brightness(img, brightness)))
        transforms.append(Lambda(lambda img: adjust_contrast(img, contrast)))
        transforms.append(
            Lambda(lambda img: adjust_saturation(img, saturation)))
        transforms.append(Lambda(lambda img: adjust_hue(img, hue)))
        np.random.shuffle(transforms)
        self.transform = Compose(transforms)

    def __call__(self, img):

        if not (_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        pil = Image.fromarray(img)
        return np.array(self.transform(pil))
