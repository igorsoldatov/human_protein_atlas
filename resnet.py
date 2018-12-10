import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
import warnings
import argparse
import scipy.misc

from random import randrange, randint
import random
from PIL import Image
from scipy.misc import imread
from skimage.transform import resize
from sklearn.metrics import f1_score
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, ZeroPadding2D, BatchNormalization, \
    GlobalAveragePooling2D
from keras.engine.input_layer import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model

from classification_models import ResNet18
from classification_models import ResNet34
# from classification_models import ResNet50
from keras.applications.resnet50 import ResNet50
from classification_models import ResNet101
from classification_models import ResNet152
from classification_models import ResNeXt50
from classification_models import ResNeXt101

from tqdm import tqdm_notebook, tqdm

from multiprocessing import Process

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ElasticTransform
)

zoo = {'resnet18': ResNet18,
       'resnet34': ResNet34,
       'resnet50': ResNet50,
       'resnet101': ResNet101,
       'resnet152': ResNet152,
       'resnext50': ResNeXt50,
       'resnext101': ResNeXt101}

import warnings

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--gpu', default='0', type=str, help='GPU')
# parser.add_argument('--lr', default=0.0001, type=float, help='leaning rate')
# parser.add_argument('--fold', default=1, type=int, help='training fold')
# parser.add_argument('--batch', default=10, type=int, help='batch size')
# parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
# parser.add_argument('--fcl1', default=1024, type=int, help='number of units of FCL1')
# parser.add_argument('--fcl2', default=0, type=int, help='number of units of FCL2')
# parser.add_argument('--tune', default='', type=str, help='hyperparameter for tuning')
# parser.add_argument('--arch', default='resnet50', type=str, help='architecture')
# parser.add_argument('--use_memory', default=False, type=bool, help='use_memory')
# global args
# args = parser.parse_args()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

random.seed(2018)
np.random.seed(2018)

label_names = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}

reverse_train_labels = dict((v, k) for k, v in label_names.items())


def get_labels(path):
    labels = pd.read_csv(path)
    labels["number_of_targets"] = labels.drop(["Id", "Target"], axis=1).sum(axis=1)
    for key in label_names.keys():
        labels[label_names[key]] = 0
    labels = labels.apply(fill_targets, axis=1)
    return labels


def load_image(path, image_id):
    image = np.zeros(shape=(512, 512, 4), dtype=np.uint8)
    image[:, :, 0] = imread(path + image_id + "_green" + ".png")
    image[:, :, 1] = imread(path + image_id + "_blue" + ".png")
    image[:, :, 2] = imread(path + image_id + "_red" + ".png")
    image[:, :, 3] = imread(path + image_id + "_yellow" + ".png")
    return image


def resize_img(image):
    image_ = resize(image, (256, 256), preserve_range=True)
    return image_.astype(np.uint8)


def train_gpu(gpu, fold, train_path, labels, parameter):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    os.makedirs(f'models/{parameter.tune}/', exist_ok=True)

    train_labels = pd.read_csv(f'./folds/train_{fold}.csv')
    valid_labels = pd.read_csv(f'./folds/valid_{fold}.csv')
    train_ids = train_labels.Id.tolist()
    valid_ids = valid_labels.Id.tolist()

    preprocessor = ImagePreprocessor(parameter)

    training_generator = DataGeneratorTrainDist(train_ids, labels, parameter, preprocessor)
    validation_generator = DataGenerator(valid_ids, labels, parameter, preprocessor, validation=True)
    predict_generator = PredictGenerator(valid_ids, preprocessor, train_path)

    model = BaseLineModel(parameter)
    model.build_model()
    model.compile_model()
    # name = '24_resnet18-batch_size-170-lr-0.45417-87ep'
    # model.load(f'./models/batch_size/{name}.h5', custom_objects={'f1_loss': f1_loss, 'f1': f1})
    model.set_generators(training_generator, validation_generator)
    history = model.learn()

    # os.makedirs(f'models/{parameter.tune}/', exist_ok=True)
    # model.save(parameter.model_name)


def crop4(image):
    h = 256
    w = 256
    crops = []
    crops.append(image[:h, :w])
    crops.append(image[:h, w:])
    crops.append(image[h:, :w])
    crops.append(image[h:, w:])
    return crops


def crop9(image):
    h = 256
    w = 256
    crops = []
    crops.append(image[:h, :w])
    crops.append(image[:h, int(w / 2):w + int(w / 2)])
    crops.append( image[:h, w:])
    crops.append(image[int(h / 2):w + int(h / 2):, :w])
    crops.append(image[int(h / 2):w + int(h / 2):, int(w / 2):w + int(w / 2)])
    crops.append(image[int(h / 2):w + int(h / 2):, w:])
    crops.append(image[h:, :w])
    crops.append(image[h:, int(w / 2):w + int(w / 2)])
    crops.append(image[h:, w:])
    return crops


def predict_crop4(model, image):
    crops = crop4(image)
    score_predict = np.zeros((4, 28))
    for n, crop in enumerate(crops):
        crop = crop.astype(dtype=np.float16)
        crop /= 255
        crop = crop.reshape((1, *crop.shape))
        score_predict[n, :] = model.model.predict(crop)
    score_predict = score_predict.sum(axis=0)
    return score_predict.astype(np.float16)


def predict_crop9(model, image):
    crops = crop9(image)
    score_predict = np.zeros((9, 28))
    for n, crop in enumerate(crops):
        crop = crop.astype(dtype=np.float16)
        crop /= 255
        crop = crop.reshape((1, *crop.shape))
        score_predict[n, :] = model.model.predict(crop)
    score_predict = score_predict.sum(axis=0)
    return score_predict.astype(np.float16)


def predict_submission(name, TTA=False):
    test_path = '../DATASET/human_protein_atlas/all/test/'

    labels = pd.read_csv('../DATASET/human_protein_atlas/all/sample_submission.csv')

    parameter = ModelParameter(test_path,
                            lr=0.00003,
                            fcl=[1024, 1024, 1024],
                            batch_size=175,
                            n_epochs=300,
                            tune='aug',
                            arch='resnet18',
                            dataset=None,
                            aug='strong_aug')
    preprocessor = ImagePreprocessor(parameter)

    model = BaseLineModel(parameter)
    # model.build_model()
    # model.compile_model()

    model.load(f'./models/new/{name}.h5', custom_objects={'f1_loss': f1_loss, 'f1': f1})

    if TTA:
        predicted = []
        for n in range(9):
            predicted.append([])
        for idx in tqdm(labels['Id'], total=len(labels)):
            image = preprocessor.load_image(idx)
            score_predict = predict_crop9(model, image)
            for n in range(9):
                score = (score_predict >= n+1)
                label_predict = np.arange(28)[score >= 0.5]
                str_predict_label = ' '.join(str(l) for l in label_predict)
                predicted[n].append(str_predict_label)

        for n in range(9):
            labels['Predicted'] = predicted[n]
            labels.to_csv(f'./submissions/{name}-TTA9(th-{n+1}).csv', header=True, index=False)
    else:
        predicted = []
        for idx in tqdm(labels['Id'], total=len(labels)):
            image = preprocessor.load_image(idx)
            image = preprocessor.preprocess(image, True)
            image = image.reshape((1, *image.shape))
            score_predict = model.model.predict(image)[0]
            label_predict = np.arange(28)[score_predict >= 0.15]
            str_predict_label = ' '.join(str(l) for l in label_predict)
            predicted.append(str_predict_label)
        labels['Predicted'] = predicted
        labels.to_csv(f'./submissions/{name}-th-0.15.csv', header=True, index=False)


def one_hot_target(target):
    one_hot = np.zeros(28)
    if target != '':
        targets = np.array(target.split(' ')).astype(np.int).tolist()
        for t in targets:
            one_hot[t] = 1
    return one_hot


def error_statistic(name):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_path = '../DATASET/human_protein_atlas/all/train/'

    labels = pd.read_csv('./folds/valid_1.csv')

    parameter = ModelParameter(train_path,
                            lr=0.00003,
                            fcl=[1024, 1024, 1024],
                            batch_size=175,
                            n_epochs=300,
                            tune='aug',
                            arch='resnet18',
                            dataset=None,
                            aug='strong_aug')
    preprocessor = ImagePreprocessor(parameter)

    model = BaseLineModel(parameter)
    model.load(f'./models/new/{name}.h5', custom_objects={'f1_loss': f1_loss, 'f1': f1})

    predicted = {}
    class_count = {}
    for idx, target in tqdm(zip(labels['Id'], labels['Target']), total=len(labels)):
        image = preprocessor.load_image(idx)
        image = preprocessor.preprocess(image, True)
        image = image.reshape((1, *image.shape))
        score_predict = model.model.predict(image)[0]
        label_predict = np.arange(28)[score_predict >= 0.5]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        score = f1_score(one_hot_target(target), one_hot_target(str_predict_label), average='macro')
        if target in predicted:
            predicted[target] += score
            class_count[target] += 1
        else:
            predicted[target] = score
            class_count[target] = 1

    for k in predicted:
        print(f'\'{k}\' : {class_count[k]} : {predicted[k]/class_count[k]}')


def dataset_statistic():
    labels = pd.read_csv('../DATASET/human_protein_atlas/all/train_ord.csv')
    plain_targets = {}
    complex_targets = {}
    for idx, target in tqdm(zip(labels['Id'], labels['Target']), total=len(labels)):
        if target in complex_targets:
            complex_targets[target] += 1
        else:
            complex_targets[target] = 1

        for t in target.split(' '):
            if t in plain_targets:
                plain_targets[t] += 1
            else:
                plain_targets[t] = 1

    print('Plain targets:')
    for k in plain_targets:
        print(f'{k} : {plain_targets[k]}')
    print('\n\n\n\nComplex targets:')
    for k in complex_targets:
        print(f'\'{k}\' : {complex_targets[k]}')


def train_distribution():
    p = np.array([0.002,0.042,0.01,0.041,0.038,0.032,0.034,0.038,0.041,0.034,0.038,0.042,0.042,0.042,0.042,0.042,0.034,0.042,0.038,0.042,0.042,0.032,0.042,0.038,0.042,0.005,0.041,0.042])
    c = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], dtype=np.uint8)
    spec = ['0', '2', '25', '21', '5', '16']

    labels = pd.read_csv('../DATASET/human_protein_atlas/all/train_ord.csv')
    dist = {}
    stat = {}
    for i in range(28):
        dist[str(i)] = []
        stat[str(i)] = 0
    for idx, target in tqdm(zip(labels['Id'], labels['Target']), total=len(labels)):
        if target in spec:
            dist[target].append(target)
        else:
            for t in target.split(' '):
                if t not in spec:
                    dist[t].append(target)

    for n in tqdm(range(23000), total=23000):
        k = np.random.choice(c, p=p)
        target = random.choice(dist[str(k)])
        for t in target.split(' '):
            if t in stat:
                stat[t] += 1

    for k in stat:
        print(f'{k} : {stat[k]}')


def score_valid(gpu, fold, train_path, labels, parameter):
    def one_hot_target(tr):
        one_hot = np.zeros(28)
        if tr != '':
            targets = np.array(tr.split(" ")).astype(np.int).tolist()
            one_hot = np.zeros(28)
            for t in targets:
                one_hot[t] = 1
        return one_hot

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    train_labels = pd.read_csv(f'./folds/train_{fold}.csv')
    valid_labels = pd.read_csv(f'./folds/valid_{fold}.csv')
    train_ids = train_labels.Id.tolist()
    valid_ids = valid_labels.Id.tolist()

    preprocessor = ImagePreprocessor(parameter)

    training_generator = DataGenerator(train_ids, labels, parameter, preprocessor)
    validation_generator = DataGenerator(valid_ids, labels, parameter, preprocessor, validation=True)
    predict_generator = PredictGenerator(valid_ids, preprocessor, train_path)

    model = BaseLineModel(parameter)
    # model.build_model()
    # model.compile_model()
    name = '24_resnet18-batch_size-170-lr-0.45417-87ep'
    model.load(f'./models/new/{name}.h5', custom_objects={'f1_loss': f1_loss, 'f1': f1})
    model.set_generators(training_generator, validation_generator)
    y_true = np.zeros((len(valid_labels), 28), dtype=np.uint8)
    y_pred = np.zeros((len(valid_labels), 28), dtype=np.uint8)
    n = -1
    for idx, target in tqdm(zip(valid_labels['Id'], valid_labels['Target']), total=len(valid_labels)):
        n += 1
        image = preprocessor.load_image(idx)
        image = preprocessor.preprocess(image, True)
        image = image.reshape((1, *image.shape))
        score_predict = model.model.predict(image)[0]
        label_predict = np.arange(28)[score_predict >= 0.5]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        # print(f'\'{str_predict_label}\'')
        y_true[n] = one_hot_target(target)
        y_pred[n] = one_hot_target(str_predict_label)

    score = f1_score(y_true, y_pred, average='macro')
    # score = model.score()
    print(f'score: {score}')


def mean_weights_layer(layer):
    list_weights = layer.get_weights()
    val = 0
    for w in list_weights:
        val += np.mean(w)
    return val


def convert_weights(input_shape_src, input_shape_trg, resnet):
    def get_new_weights_conv(weights):
        w = np.zeros((7, 7, 4, 64))
        for i in range(64):
            w[:, :, :3, i] = weights[0][:, :, :, i]
            w[:, :, 3, i] = weights[0][:, :, 2, i]
        weights[0] = w
        return weights

    def get_new_weights_bn(weights):
        for i in range(3):
            w = np.zeros((4,))
            w[:3] = weights[i]
            w[3] = weights[i][2]
            weights[i] = w
        return weights

    print('get pretrained model....')
    pretrain_model = zoo[resnet](include_top=False, weights='imagenet', input_shape=input_shape_src)
    print('create new model....')
    pretrain_model_new = zoo[resnet](include_top=False, weights=None, input_shape=input_shape_trg)

    for l in range(len(pretrain_model.layers)):
        print(f'convert layer #{l}....')
        if l == 3:
            pretrain_model_new.layers[l].set_weights(get_new_weights_conv(pretrain_model.layers[l].get_weights()))
        elif l == 1:
            pretrain_model_new.layers[l].set_weights(get_new_weights_bn(pretrain_model.layers[l].get_weights()))
        else:
            pretrain_model_new.layers[l].set_weights(pretrain_model.layers[l].get_weights())

    for l in range(len(pretrain_model.layers)):
        name = pretrain_model.layers[l].name
        val_1 = mean_weights_layer(pretrain_model.layers[l])
        val_2 = mean_weights_layer(pretrain_model_new.layers[l])
        print(f'{name} = {val_1 == val_2}')

    print('save new model....')
    pretrain_model_new.save_weights(f'./pretrained_weights/{resnet}_4ch.h5')

    # model = Sequential()
    # model.add(pretrain_model_new)
    # model.add(GlobalAveragePooling2D(name='avg_pool'))
    # model.add(Dense(num_classes, activation='sigmoid', name='fc28'))
    # model.summary()
    # model.save_weights('./weights_resnet50_4ch.h5')

    # model.add(model_resnet50)
    # model.add(Flatten())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1024))
    # model.add(Dropout(0.5))
    # model.add(Dense(self.num_classes))
    # model.add(Activation('sigmoid'))


def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row


def focal_loss(target, input):
    gamma = 2.
    input = tf.cast(input, tf.float32)
    # max_val = K.clip(-input, 0, 1)
    max_val = K.relu(-input)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss

    return K.mean(K.sum(loss, axis=1))


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


class ModelParameter:
    def __init__(self, basepath,
                 lr=0.0001,
                 num_classes=28,
                 image_rows=512,
                 image_cols=512,
                 batch_size=10,
                 n_channels=4,
                 row_scale_factor=2,
                 col_scale_factor=2,
                 shuffle=True,
                 n_epochs=100,
                 fcl=[],
                 tune='test',
                 arch='resnet18',
                 dataset=None,
                 aug='',
                 number=100):
        self.basepath = basepath
        self.arch = arch
        self.lr = lr
        self.num_classes = num_classes
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.row_scale_factor = row_scale_factor
        self.col_scale_factor = col_scale_factor
        self.scaled_row_dim = np.int(self.image_rows / self.row_scale_factor)
        self.scaled_col_dim = np.int(self.image_cols / self.col_scale_factor)
        self.n_epochs = n_epochs
        self.fcl = fcl
        self.aug = aug
        self.tune = tune
        if tune == 'lr':
            self.log_dir = 'logs/{}/{}_{}-{}-{:.8f}'.format(tune, number, arch, tune, self.__getattribute__(tune))
            self.model_name = './models/{}/{}'.format(tune, number)
            self.model_name = self.model_name + '-{epoch:02d}ep-{val_f1:.4f}.h5'
        elif tune == 'fcl':
            self.log_dir = 'logs/{}/{}_{}-{}-({})'.format(tune, number, arch, tune, self.fcl)
            self.model_name = './models/{}/{}'.format(tune, number)
            self.model_name = self.model_name + '-{epoch:02d}ep-{val_f1:.4f}.h5'
        else:
            self.log_dir = 'logs/{}/{}_{}-{}-{}'.format(tune, number, arch, tune, self.__getattribute__(tune))
            self.model_name = './models/{}/{}'.format(tune, number)
            self.model_name = self.model_name + '-{epoch:02d}ep-{val_f1:.4f}.h5'
        self.tensorbord = TensorBoard(log_dir=self.log_dir)
        self.dataset = dataset


class ImagePreprocessor:

    def __init__(self, modelparameter):
        self.parameter = modelparameter
        self.basepath = self.parameter.basepath
        self.scaled_row_dim = self.parameter.scaled_row_dim
        self.scaled_col_dim = self.parameter.scaled_col_dim
        self.n_channels = self.parameter.n_channels
        self.aug = self.parameter.aug
        self.augmentation = self.strong_aug_01()

    def preprocess(self, image, validation=False):
        # image = self.resize(image)
        # image = self.reshape(image)
        if not validation:
            if self.aug == 'crop_random':
                print(f'aug == {self.aug}')
                image = self.crop_random(image)
            elif self.aug == 'crop4':
                print(f'aug == {self.aug}')
                image = self.crop4(image)
            elif self.aug == 'crop9':
                print(f'aug == {self.aug}')
                image = self.crop9(image)
            elif self.aug == 'strong_aug':
                image = self.crop_random(image)
                data = {"image": image}
                image = self.augmentation(**data)["image"]
        else:
            image = self.crop_random(image)
        image = self.normalize(image)
        return image

    def crop_random(self, image):
        shape = image.shape
        h = self.scaled_row_dim
        w = self.scaled_col_dim
        y = randrange(0, shape[0] - h)
        x = randrange(0, shape[1] - w)
        return image[y:y+h, x:x+w, :]

    def crop4(self, image):
        h = self.scaled_row_dim
        w = self.scaled_col_dim
        r = randrange(0, 4)
        if r == 0:
            return image[:h, :w]
        elif r == 1:
            return image[:h, w:]
        elif r == 2:
            return image[h:, :w]
        elif r == 3:
            return image[h:, w:]

    def crop9(self, image):
        h = self.scaled_row_dim
        w = self.scaled_col_dim
        r = randrange(0, 9)
        if r == 0:
            return image[:h, :w]
        elif r == 1:
            return image[:h, int(w/2):w+int(w/2)]
        elif r == 2:
            return image[:h, w:]
        elif r == 3:
            return image[int(h/2):w+int(h/2):, :w]
        elif r == 4:
            return image[int(h/2):w+int(h/2):, int(w/2):w+int(w/2)]
        elif r == 5:
            return image[int(h/2):w+int(h/2):, w:]
        elif r == 6:
            return image[h:, :w]
        elif r == 7:
            return image[h:, int(w/2):w+int(w/2)]
        elif r == 8:
            return image[h:, w:]

    def resize(self, image):
        if image.shape[0] != self.scaled_row_dim or image.shape[1] != self.scaled_col_dim:
            image = image.astype(np.uint16)
            image = resize(image, (self.scaled_row_dim, self.scaled_col_dim), preserve_range=True)
            image = image.astype(np.float16)
        return image

    def strong_aug_00(self, p=0.5):
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            ElasticTransform(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                # MedianBlur(blur_limit=3, p=0.1),
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

    def strong_aug_01(self, p=0.5):
        return Compose([
            OneOf([
                RandomRotate90(),
                Flip(),
                Transpose(),
                ElasticTransform(),
            ], p=1.0)
        ], p=p)

    def strong_aug_02(self, p=0.5):
        return Compose([
            OneOf([
                RandomRotate90(),
                Flip(),
                Transpose(),
                ElasticTransform(),
            ], p=1.0)
        ], p=p)

    def reshape(self, image):
        image = np.reshape(image, (image.shape[0], image.shape[1], self.n_channels))
        return image

    def normalize(self, image):
        image = image.astype(dtype=np.float16)
        image /= 255
        return image

    def load_image(self, image_id):
        image = np.zeros(shape=(512, 512, 4), dtype=np.uint8)
        image[:, :, 0] = imread(self.basepath + image_id + "_green" + ".png")
        image[:, :, 1] = imread(self.basepath + image_id + "_blue" + ".png")
        image[:, :, 2] = imread(self.basepath + image_id + "_red" + ".png")
        image[:, :, 3] = imread(self.basepath + image_id + "_yellow" + ".png")
        return image[:, :, 0:self.parameter.n_channels]


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, modelparameter, imagepreprocessor, validation=False):
        self.params = modelparameter
        self.labels = labels
        self.list_IDs = list_IDs
        self.dim = (self.params.scaled_row_dim, self.params.scaled_col_dim)
        self.batch_size = self.params.batch_size
        self.n_channels = self.params.n_channels
        self.num_classes = self.params.num_classes
        self.shuffle = self.params.shuffle
        self.preprocessor = imagepreprocessor
        self.on_epoch_end()
        self.dataset = self.params.dataset
        self.use_memory = (self.dataset is not None)
        self.validation = validation
        if self.use_memory:
            self.nlabels = {}
            for n, idx in tqdm(enumerate(labels['Id'].tolist()), total=len(labels)):
                self.nlabels[idx] = n

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id == identifier].drop(['Id', 'Target', 'number_of_targets'], axis=1).values

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        # Generate data
        for i, identifier in enumerate(list_IDs_temp):
            # Store sample
            if self.use_memory:
                image = self.dataset[self.nlabels[identifier]]
            else:
                image = self.preprocessor.load_image(identifier)
            image = self.preprocessor.preprocess(image, self.validation)
            X[i] = image
            # Store class
            y[i] = self.get_targets_per_image(identifier)
        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y


class DataGeneratorTrainDist(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, modelparameter, imagepreprocessor, validation=False):
        self.params = modelparameter
        self.labels = labels
        self.list_IDs = list_IDs
        self.dim = (self.params.scaled_row_dim, self.params.scaled_col_dim)
        self.batch_size = self.params.batch_size
        self.n_channels = self.params.n_channels
        self.num_classes = self.params.num_classes
        self.shuffle = self.params.shuffle
        self.preprocessor = imagepreprocessor
        self.on_epoch_end()
        self.dataset = self.params.dataset
        self.use_memory = (self.dataset is not None)
        self.validation = validation
        if self.use_memory:
            self.nlabels = {}
            for n, idx in tqdm(enumerate(labels['Id'].tolist()), total=len(labels)):
                self.nlabels[idx] = n
        self.dist_idxs = self.__make_dist_dataset()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id == identifier].drop(['Id', 'Target', 'number_of_targets'], axis=1).values

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        # Generate data
        for i, identifier in enumerate(list_IDs_temp):
            # Store sample
            if self.use_memory:
                image = self.dataset[self.nlabels[identifier]]
            else:
                image = self.preprocessor.load_image(identifier)
            image = self.preprocessor.preprocess(image, self.validation)
            X[i] = image
            # Store class
            y[i] = self.get_targets_per_image(identifier)
        return X, y

    def __make_dist_dataset(self):
        spec = ['0', '2', '25', '21', '5', '16']
        labels = pd.read_csv('../DATASET/human_protein_atlas/all/train_ord.csv')
        dist = {}
        for i in range(28):
            dist[str(i)] = []
        for idx, target in tqdm(zip(labels['Id'], labels['Target']), total=len(labels)):
            if target in spec:
                dist[target].append(idx)
            else:
                for t in target.split(' '):
                    if t not in spec:
                        dist[t].append(idx)
        return dist

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(23000 / self.batch_size)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
        y = np.empty((self.batch_size, self.num_classes), dtype=int)

        c = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            dtype=np.uint8)
        p = np.array(
            [0.002, 0.042, 0.01, 0.041, 0.038, 0.032, 0.034, 0.038, 0.041, 0.034, 0.038, 0.042, 0.042, 0.042, 0.042,
             0.042, 0.034, 0.042, 0.038, 0.042, 0.042, 0.032, 0.042, 0.038, 0.042, 0.005, 0.041, 0.042])

        for i in range(self.batch_size):
            k = np.random.choice(c, p=p)
            idx = random.choice(self.dist_idxs[str(k)])
            if self.use_memory:
                image = self.dataset[self.nlabels[idx]]
            else:
                image = self.preprocessor.load_image(idx)
            image = self.preprocessor.preprocess(image, self.validation)
            X[i] = image
            # Store class
            y[i] = self.get_targets_per_image(idx)

        return X, y


class PredictGenerator:

    def __init__(self, predict_Ids, imagepreprocessor, predict_path):
        self.preprocessor = imagepreprocessor
        self.preprocessor.basepath = predict_path
        self.identifiers = predict_Ids

    def predict(self, model):
        y = np.empty(shape=(len(self.identifiers), self.preprocessor.parameter.num_classes))
        for n in range(len(self.identifiers)):
            image = self.preprocessor.load_image(self.identifiers[n])
            image = self.preprocessor.preprocess(image)
            image = image.reshape((1, *image.shape))
            y[n] = model.predict(image)
        return y


class BaseLineModel:

    def __init__(self, modelparameter):
        self.params = modelparameter
        self.arch = self.params.arch
        self.num_classes = self.params.num_classes
        self.fcl = self.params.fcl
        self.img_rows = self.params.scaled_row_dim
        self.img_cols = self.params.scaled_col_dim
        self.n_channels = self.params.n_channels
        self.input_shape = (self.img_rows, self.img_cols, self.n_channels)
        self.my_metrics = [f1]
        self.model = Sequential()

    def build_model(self):
        if self.arch == 'resnet18':
            model_resnet = ResNet18(include_top=False, weights=None, input_shape=self.input_shape)
            model_resnet.load_weights('./pretrained_weights/resnet18_4ch.h5')
        elif self.arch == 'resnet34':
            model_resnet = ResNet34(include_top=False, weights=None, input_shape=self.input_shape)
            model_resnet.load_weights('./pretrained_weights/resnet34_4ch.h5')
        elif self.arch == 'resnet50':
            model_resnet = ResNet50(include_top=False, weights=None, input_shape=self.input_shape)
            model_resnet.load_weights('./pretrained_weights/resnet50_4ch.h5')

        self.model.add(model_resnet)
        self.model.add(GlobalAveragePooling2D(name='avg_pool'))
        for n, fcl in enumerate(self.fcl):
            self.model.add(Dense(fcl, activation='relu', name=f'fcl{n}'))
            self.model.add(Dropout(1))
        self.model.add(Dense(self.num_classes, activation='sigmoid', name='fc28')) # f1_loss
        # self.model.add(Dense(self.num_classes, name='fc28')) # focal_loss
        self.model.summary()

    def compile_model(self):
        self.model.compile(loss=f1_loss,
                           optimizer=keras.optimizers.Adam(lr=self.params.lr),
                           metrics=self.my_metrics)

    def set_generators(self, train_generator, validation_generator):
        self.training_generator = train_generator
        self.validation_generator = validation_generator

    def learn(self):
        model_checkpoint = ModelCheckpoint(self.params.model_name, monitor='val_f1',
                                           mode='max', save_best_only=True, verbose=1)
        return self.model.fit_generator(generator=self.training_generator,
                                        validation_data=self.validation_generator,
                                        epochs=self.params.n_epochs,
                                        use_multiprocessing=True,
                                        workers=16,
                                        callbacks=[model_checkpoint, self.params.tensorbord])

    def score(self):
        return self.model.evaluate_generator(generator=self.validation_generator,
                                             use_multiprocessing=True,
                                             workers=16,
                                             verbose=1)

    def predict(self, predict_generator):
        y = predict_generator.predict(self.model)
        return y

    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)

    def load(self, modelinputpath, custom_objects={}):
        self.model = load_model(modelinputpath, custom_objects=custom_objects)


def main():
    train_path = '../DATASET/human_protein_atlas/all/train/'
    labels = get_labels('../DATASET/human_protein_atlas/all/train.csv')

    # dataset = None
    # dataset = np.zeros((len(labels), 512, 512, 4), dtype=np.uint8)
    # for n, idx in tqdm(enumerate(labels['Id'].tolist()), total=len(labels)):
    #     image = load_image(train_path, idx)
    #     dataset[n, :, :, :] = image
    # np.save('./train_512x512x4.npy', dataset)
    dataset = np.load('./train_512x512x4.npy')

    param1 = ModelParameter(train_path,
                            lr=0.00003,
                            fcl=[1024, 1024, 1024],
                            batch_size=170,
                            n_epochs=100,
                            tune='batch_size',
                            arch='resnet18',
                            dataset=dataset,
                            aug='strong_aug',
                            number=28,
                            shuffle=True)

    train_gpu('0', 1, train_path, labels, param1)

    # p1 = Process(target=train_gpu, args=('0', 1, train_path, labels, param1))
    #
    # param2 = ModelParameter(train_path,
    #                         lr=0.00002738,
    #                         fcl1=1024,
    #                         fcl2=1024,
    #                         batch_size=64,
    #                         n_epochs=200,
    #                         tune='aug',
    #                         arch='resnet18',
    #                         dataset=dataset,
    #                         aug='')
    #
    # p2 = Process(target=train_gpu, args=('1', 1, train_path, labels, param2))
    #
    # p1.start()
    # p2.start()
    #
    # p1.join()
    # p2.join()


if __name__ == '__main__':
    main()
    # error_statistic('22_rn18-strong_aug-valf1-0.44200-119ep')
