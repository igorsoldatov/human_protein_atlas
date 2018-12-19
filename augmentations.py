from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ElasticTransform
)
from random import randrange
from random import randint
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.misc

from tqdm import tqdm_notebook, tqdm


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
        ElasticTransform(p=1.0),
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
            # CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        # HueSaturationValue(p=0.3),
    ], p=p)


# image = load_image()
# scipy.misc.imsave(f'./augs/origin.jpg', image[:, :, :3])
#
# whatever_data = "my name"
# augmentation = strong_aug(p=1.0)
# data = {"image": image}
#
# for n in range(1000):
#     # augmented = augmentation(**data)
#     # image = augmented["image"]
#     image = augmentation(**data)["image"]
#     # print(f'shape={image.shape}')
#     if image.shape[2] == 3:
#         print(f'shape={image.shape}')
#     scipy.misc.imsave(f'./augs/{randint(1, 100000000)}.jpg', image[:, :, :3])

# plt.imshow(image[:, :, :3])
# plt.show()

def tensor_to_image(tensor):
    img = np.zeros((512, 512, 4))
    for i in range(4):
        img[:, :, i] = tensor[0, i, :, :]
    return img


def image_to_tensor(img):
    tensor = np.zeros((1, 4, 512, 512))
    for i in range(4):
        tensor[0, i, :, :] = img[:, :, i]
    return tensor


image = load_image()
tensor = image_to_tensor(image)
image = tensor_to_image(tensor)
scipy.misc.imsave(f'./classes_example/tta/origin.jpg', image[:, :, :3])
scipy.misc.imsave(f'./classes_example/tta/fliplr.jpg', np.fliplr(image)[:, :, :3])
scipy.misc.imsave(f'./classes_example/tta/flipud.jpg', np.flipud(image)[:, :, :3])
scipy.misc.imsave(f'./classes_example/tta/rot90_1.jpg', np.rot90(image, 1)[:, :, :3])
scipy.misc.imsave(f'./classes_example/tta/rot90_2.jpg', np.rot90(image, 2)[:, :, :3])
scipy.misc.imsave(f'./classes_example/tta/rot90_3.jpg', np.rot90(image, 3)[:, :, :3])
