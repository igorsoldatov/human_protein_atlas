from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)
from random import randrange
from random import randint
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.misc


def load_image(image_id='00383b44-bbbb-11e8-b2ba-ac1f6b6435d0'):
    path = '../DATASET/human_protein_atlas/all/train/'
    image = np.zeros(shape=(512, 512, 4), dtype=np.uint8)
    image[:, :, 0] = imread(path + image_id + "_green" + ".png")
    image[:, :, 1] = imread(path + image_id + "_blue" + ".png")
    image[:, :, 2] = imread(path + image_id + "_red" + ".png")
    image[:, :, 3] = imread(path + image_id + "_yellow" + ".png")
    return image


def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


image = load_image()
scipy.misc.imsave(f'./augs/origing.jpg', image[:, :, :3])

whatever_data = "my name"
augmentation = strong_aug(p=0.9)
data = {"image": image}

for i in range(100):
    # augmented = augmentation(**data)
    # image = augmented["image"]
    image = augmentation(**data)["image"]
    scipy.misc.imsave(f'./augs/{randint(1, 100000000)}.jpg', image[:, :, :3])

# plt.imshow(image[:, :, :3])
# plt.show()