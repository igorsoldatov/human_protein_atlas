import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
import warnings

from PIL import Image
from scipy.misc import imread
from skimage.transform import resize
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, ZeroPadding2D, BatchNormalization, \
    GlobalAveragePooling2D
from keras.engine.input_layer import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K

import warnings

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


class ModelParameter:

    def __init__(self, basepath,
                 num_classes=28,
                 image_rows=512,
                 image_cols=512,
                 batch_size=8,
                 n_channels=4,
                 row_scale_factor=2,
                 col_scale_factor=2,
                 shuffle=False,
                 n_epochs=100):
        self.basepath = basepath
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
        self.tensorbord = TensorBoard(log_dir='logs/adam_lr_0.001_focal_loss_{}'.format(time()))


class ImagePreprocessor:

    def __init__(self, modelparameter):
        self.parameter = modelparameter
        self.basepath = self.parameter.basepath
        self.scaled_row_dim = self.parameter.scaled_row_dim
        self.scaled_col_dim = self.parameter.scaled_col_dim
        self.n_channels = self.parameter.n_channels

    def preprocess(self, image):
        image = self.resize(image)
        image = self.reshape(image)
        image = self.normalize(image)
        return image

    def resize(self, image):
        image = resize(image, (self.scaled_row_dim, self.scaled_col_dim))
        return image

    def reshape(self, image):
        image = np.reshape(image, (image.shape[0], image.shape[1], self.n_channels))
        return image

    def normalize(self, image):
        image /= 255
        return image

    def load_image(self, image_id):
        image = np.zeros(shape=(512, 512, 4))
        image[:, :, 0] = imread(self.basepath + image_id + "_green" + ".png")
        image[:, :, 1] = imread(self.basepath + image_id + "_blue" + ".png")
        image[:, :, 2] = imread(self.basepath + image_id + "_red" + ".png")
        image[:, :, 3] = imread(self.basepath + image_id + "_yellow" + ".png")
        return image[:, :, 0:self.parameter.n_channels]


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, modelparameter, imagepreprocessor):
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

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id == identifier].drop(['Id', 'Target', 'number_of_targets'], axis=1).values

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        # Generate data
        for i, identifier in enumerate(list_IDs_temp):
            # Store sample
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


def create_model(input_shape, n_out):
    pretrain_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    pretrain_model.summary()

    model = Sequential()
    model.add(pretrain_model)
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_out))
    model.add(Activation('sigmoid'))
    return model


def mean_weights_layer(layer):
    list_weights = layer.get_weights()
    val = 0
    for w in list_weights:
        val += np.mean(w)
    return val


def convert_weights(input_shape_src, input_shape_trg, num_classes):

    def get_new_weights(weights):
        w = np.zeros((7, 7, 4, 64))
        for i in range(64):
            w[:, :, :3, i] = weights[0][:, :, :, i]
            w[:, :, 3, i] = weights[0][:, :, 2, i]
        weights[0] = w
        return weights

    pretrain_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape_src)
    pretrain_model_new = ResNet50(include_top=False, weights=None, input_shape=input_shape_trg)

    for l in range(len(pretrain_model.layers)):
        if l == 2:
            pretrain_model_new.layers[l].set_weights(get_new_weights(pretrain_model.layers[l].get_weights()))
        else:
            pretrain_model_new.layers[l].set_weights(pretrain_model.layers[l].get_weights())

    for l in range(len(pretrain_model.layers)):
        name = pretrain_model.layers[l].name
        val_1 = mean_weights_layer(pretrain_model.layers[l])
        val_2 = mean_weights_layer(pretrain_model_new.layers[l])
        print(f'{name} = {val_1 == val_2}')

    pretrain_model_new.save_weights('./weights_resnet50_4ch.h5')

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


class BaseLineModel:

    def __init__(self, modelparameter):
        self.params = modelparameter
        self.num_classes = self.params.num_classes
        self.img_rows = self.params.scaled_row_dim
        self.img_cols = self.params.scaled_col_dim
        self.n_channels = self.params.n_channels
        self.input_shape = (self.img_rows, self.img_cols, self.n_channels)
        self.my_metrics = ['accuracy']
        self.model = Sequential()

    def build_model(self):
        model_resnet50 = ResNet50(include_top=False, weights=None, input_shape=self.input_shape)
        model_resnet50.load_weights('./weights_resnet50_4ch.h5')

        self.model.add(model_resnet50)
        self.model.add(GlobalAveragePooling2D(name='avg_pool'))
        # self.model.add(Dense(self.num_classes, activation='sigmoid', name='fc28'))
        self.model.add(Dense(self.num_classes, name='fc28'))

    def compile_model(self):
        self.model.compile(loss=focal_loss,
                           optimizer=keras.optimizers.Adam(lr=0.001),
                           metrics=self.my_metrics)

    def set_generators(self, train_generator, validation_generator):
        self.training_generator = train_generator
        self.validation_generator = validation_generator

    def learn(self):
        return self.model.fit_generator(generator=self.training_generator,
                                        validation_data=self.validation_generator,
                                        epochs=self.params.n_epochs,
                                        use_multiprocessing=True,
                                        workers=8,
                                        callbacks=[self.params.tensorbord])

    def score(self):
        return self.model.evaluate_generator(generator=self.validation_generator,
                                             use_multiprocessing=True,
                                             workers=8)

    def predict(self, predict_generator):
        y = predict_generator.predict(self.model)
        return y

    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)

    def load(self, modelinputpath):
        self.model = load_model(modelinputpath)


# model = create_model((224, 224, 4), 28)
# model = create_model((512, 512, 4), 28)
# model.summary()

def main():
    train_path = '../DATASET/human_protein_atlas/all/train/'

    fold = 1
    labels = pd.read_csv('../DATASET/human_protein_atlas/all/train.csv')
    labels["number_of_targets"] = labels.drop(["Id", "Target"], axis=1).sum(axis=1)
    for key in label_names.keys():
        labels[label_names[key]] = 0
    labels = labels.apply(fill_targets, axis=1)

    train_labels = pd.read_csv(f'./folds/train_{fold}.csv')
    valid_labels = pd.read_csv(f'./folds/valid_{fold}.csv')
    train_ids = train_labels.Id.tolist()
    valid_ids = valid_labels.Id.tolist()

    parameter = ModelParameter(train_path)
    preprocessor = ImagePreprocessor(parameter)

    if 0:
        n_channels = parameter.n_channels
        input_shape_src = (parameter.scaled_row_dim, parameter.scaled_col_dim, n_channels - 1)
        input_shape_trg = (parameter.scaled_row_dim, parameter.scaled_col_dim, n_channels)
        convert_weights(input_shape_src, input_shape_trg, parameter.num_classes)

    training_generator = DataGenerator(train_ids, labels, parameter, preprocessor)
    validation_generator = DataGenerator(valid_ids, labels, parameter, preprocessor)
    predict_generator = PredictGenerator(valid_ids, preprocessor, train_path)

    model = BaseLineModel(parameter)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    history = model.learn()
    # model.save("baseline_model.h5")
    proba_predictions = model.predict(predict_generator)
    baseline_proba_predictions = pd.DataFrame(proba_predictions, columns=labels.drop(
        ["Target", "number_of_targets", "Id"], axis=1).columns)
    baseline_proba_predictions.to_csv("baseline_predictions.csv")


def test():
    model = Sequential()


if __name__ == '__main__':
    main()
