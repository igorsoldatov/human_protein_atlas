import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
import warnings
import argparse
import scipy.misc

from random import randrange, randint
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

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

zoo = {'resnet18': ResNet18,
       'resnet34': ResNet34,
       'resnet50': ResNet50,
       'resnet101': ResNet101,
       'resnet152': ResNet152,
       'resnext50': ResNeXt50,
       'resnext101': ResNeXt101}

import warnings

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu', default='0', type=str, help='GPU')
parser.add_argument('--lr', default=0.0001, type=float, help='leaning rate')
parser.add_argument('--fold', default=1, type=int, help='training fold')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--fcl1', default=1024, type=int, help='number of units of FCL1')
parser.add_argument('--fcl2', default=0, type=int, help='number of units of FCL2')
parser.add_argument('--tune', default='', type=str, help='hyperparameter for tuning')
parser.add_argument('--arch', default='resnet50', type=str, help='architecture')
parser.add_argument('--use_memory', default=False, type=bool, help='use_memory')

global args
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# 1060 - 6 batch size
# 1070 - 10 batch size
# adam, lr = 0.0001 - default

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
                 shuffle=False,
                 n_epochs=100,
                 fcl1=0,
                 fcl2=0,
                 tune='test',
                 arch='resnet18'):
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
        self.fcl1 = fcl1
        self.fcl2 = fcl2
        if tune == 'lr':
            self.log_dir = 'logs/{}/{}-{}-{:.8f}'.format(tune, arch, tune, self.__getattribute__(tune))
            self.model_name = 'models/{}/{}-{}-{:.8f}.h5'.format(tune, arch, tune, self.__getattribute__(tune))
        elif tune == 'fcl':
            self.log_dir = 'logs/{}/{}-{}-({:.0f}-{:.0f})'.format(tune, arch, tune, self.fcl1, self.fcl2)
            self.model_name = 'models/{}/{}-{}-({:.0f}-{:.0f}).h5'.format(tune, arch, tune, self.fcl1, self.fcl2)
        else:
            self.log_dir = 'logs/{}/{}-{}-{}'.format(tune, arch, tune, self.__getattribute__(tune))
            self.model_name = 'models/{}/{}-{}-{}.h5'.format(tune, arch, tune, self.__getattribute__(tune))
        self.tensorbord = TensorBoard(log_dir=self.log_dir)


class ImagePreprocessor:

    def __init__(self, modelparameter):
        self.parameter = modelparameter
        self.basepath = self.parameter.basepath
        self.scaled_row_dim = self.parameter.scaled_row_dim
        self.scaled_col_dim = self.parameter.scaled_col_dim
        self.n_channels = self.parameter.n_channels
        self.augmentation = self.strong_aug(p=0.9)

    def preprocess(self, image):
        # image = self.resize(image)
        image = self.crop_random(image)
        # image = self.reshape(image)
        # data = {"image": image}
        # image = self.augmentation(**data)["image"]
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
        image = resize(image, (self.scaled_row_dim, self.scaled_col_dim))
        return image

    def strong_aug(self, p=0.5):
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

    def __init__(self, list_IDs, labels, modelparameter, imagepreprocessor, dataset=None):
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
        self.dataset = dataset
        self.use_memory = (dataset is not None)
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
            image = self.preprocessor.preprocess(image)
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
        self.fcl1 = self.params.fcl1
        self.fcl2 = self.params.fcl2
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
        if self.fcl1 != 0:
            self.model.add(Dense(self.fcl1, activation='relu', name='fcl1'))
        if self.fcl1 != 0 and self.fcl2 != 0:
            self.model.add(Dense(self.fcl2, activation='relu', name='fcl2'))
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
                                        callbacks=[self.params.tensorbord])

    def score(self):
        return self.model.evaluate_generator(generator=self.validation_generator,
                                             use_multiprocessing=True,
                                             workers=16)

    def predict(self, predict_generator):
        y = predict_generator.predict(self.model)
        return y

    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)

    def load(self, modelinputpath, custom_objects={}):
        self.model = load_model(modelinputpath, custom_objects=custom_objects)


def main():
    train_path = '../DATASET/human_protein_atlas/all/train/'

    fold = args.fold
    labels = pd.read_csv('../DATASET/human_protein_atlas/all/train.csv')
    labels["number_of_targets"] = labels.drop(["Id", "Target"], axis=1).sum(axis=1)
    for key in label_names.keys():
        labels[label_names[key]] = 0
    labels = labels.apply(fill_targets, axis=1)

    train_labels = pd.read_csv(f'./folds/train_{fold}.csv')
    valid_labels = pd.read_csv(f'./folds/valid_{fold}.csv')
    train_ids = train_labels.Id.tolist()
    valid_ids = valid_labels.Id.tolist()

    parameter = ModelParameter(train_path,
                               lr=args.lr,
                               fcl1=args.fcl1,
                               fcl2=args.fcl2,
                               batch_size=args.batch,
                               n_epochs=args.epochs,
                               tune=args.tune,
                               arch=args.arch)
    preprocessor = ImagePreprocessor(parameter)

    dataset = None
    if args.use_memory:
        print(f'use_memory={args.use_memory}')
        dataset = np.zeros((len(labels), 512, 512, 4), dtype=np.uint8)
        for n, idx in tqdm(enumerate(labels['Id'].tolist()), total=len(labels)):
            dataset[n, :, :, :] = preprocessor.load_image(idx)

    training_generator = DataGenerator(train_ids, labels, parameter, preprocessor, dataset)
    validation_generator = DataGenerator(valid_ids, labels, parameter, preprocessor, dataset)
    predict_generator = PredictGenerator(valid_ids, preprocessor, train_path)

    model = BaseLineModel(parameter)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    history = model.learn()

    # os.mkdir(f'models/{args.tune}/')
    os.makedirs(f'models/{args.tune}/', exist_ok=True)
    model.save(parameter.model_name)
    # proba_predictions = model.predict(predict_generator)
    # baseline_proba_predictions = pd.DataFrame(proba_predictions, columns=labels.drop(
    #     ["Target", "number_of_targets", "Id"], axis=1).columns)
    # baseline_proba_predictions.to_csv("baseline_predictions.csv")


if __name__ == '__main__':
    main()
