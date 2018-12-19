PYTHONHASHSEED=0
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random as rn
np.random.seed(2018)
rn.seed(2018)
import keras
import warnings
import argparse
import scipy.misc


from random import randrange, randint
from PIL import Image
from scipy.misc import imread, imsave
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
tf.set_random_seed(2018)
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


def predict_submission(name, number, TTA=False):
    def make_subs(labels_, predict_, file_name, class_num=28):
        predicted_ = []
        for i in range(predict_.shape[0]):
            label = np.arange(class_num)[predict_[i]]
            str_label = ' '.join(str(l) for l in label)
            if class_num != 28:
                predicted_.append(retarget_15(str_label))
            else:
                predicted_.append(str_label)
        labels_['Predicted'] = predicted_
        labels_.to_csv(file_name, header=True, index=False)

    def make_subs_by_classes(labels_, predict_, file_name, class_num=28):
        predicted_ = []
        for i in range(28):
            predicted_.append([])

        for i in range(predict_.shape[0]):
            label = np.arange(class_num)[predict_[i]]
            label = label.tolist()
            if class_num == 28:
                label = [str(l_) for l_ in label]
            else:
                label = [retarget_15(str(l_)) for l_ in label]
            for l in range(28):
                if str(l) in label:
                    predicted_[l].append(str(l))
                else:
                    predicted_[l].append('')

        for i in range(28):
            labels_['Predicted'] = predicted_[i]
            labels_.to_csv(file_name + f'_{str(i).zfill(2)}.csv', header=True, index=False)

    def predict_and_save_row_scores(model_, labels_, name_, number_):
        score_predict_ = np.zeros((len(labels_), 28))
        for n_, idx_ in tqdm(enumerate(labels_['Id']), total=len(labels_)):
            image_ = preprocessor.load_image(idx_)
            image_ = preprocessor.preprocess(image_, True)
            image_ = image_.reshape((1, *image_.shape))
            score_predict_[n_] = model_.model.predict(image_)[0]
        os.makedirs(f'./score_predict/{number_}/', exist_ok=True)
        np.save(f'./score_predict/{number_}/{name_}.npy', score_predict_)
        return score_predict_

    def get_norm(classes_sum):
        classes_sum_target = np.array([4852, 472, 1364, 588, 700, 946, 380, 1063, 20, 17, 11, 412, 259, 202,
                                       401, 8, 200, 79, 340, 558, 65, 1422, 302, 1117, 121, 3099, 124, 4])
        coef = np.array([1, 10, 4, 8, 7, 5, 13, 5, 243, 285, 441, 12, 19, 24,
                         12, 607, 24, 61, 14, 9, 75, 3, 16, 4, 40, 2, 39, 1213])
        norm = np.absolute(classes_sum_target - classes_sum)
        norm = norm * coef
        sum_norm = norm.sum()
        return sum_norm

    def retarget_15(target_):
        maping = {'0': '',
                  '1': '8',
                  '2': '9',
                  '3': '10',
                  '4': '11',
                  '5': '12',
                  '6': '13',
                  '7': '15',
                  '8': '16',
                  '9': '17',
                  '10': '18',
                  '11': '20',
                  '12': '24',
                  '13': '26',
                  '14': '27'}
        new_targets_ = []
        nt = ''
        if target_ != '':
            targets_ = np.array(target_.split(' ')).astype(np.uint8).tolist()
            for tr in targets_:
                new_targets_.append(maping[str(tr)])
            nt = ' '.join(str(l) for l in new_targets_)
        return nt


    os.makedirs(f'./submissions/{number}/', exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    test_path = '../DATASET/human_protein_atlas/all/test/'
    labels = pd.read_csv('../DATASET/human_protein_atlas/all/sample_submission.csv')

    if 0:
        parameter = ModelParameter(test_path,
                                lr=0.00003,
                                fcl=[1024, 1024, 1024],
                                batch_size=150,
                                n_epochs=100,
                                tune='aug',
                                arch='resnet18',
                                dataset=None,
                                aug='strong_aug')
        preprocessor = ImagePreprocessor(parameter)

        model = BaseLineModel(parameter)
        # model.build_model()
        # model.compile_model()

        model.load(f'./models/{number}/{name}.h5', custom_objects={'f1_loss': f1_loss, 'f1': f1})

        # score_predict = predict_and_save_row_scores(model, labels, name, number)

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
    elif 0:
        predict = (score_predict >= 0.5)
        make_subs(labels, predict, f'./submissions/{number}/{name}-baseline.csv')
    elif 0:
        score_predict = np.load(f'./score_predict/{number}/{name}.npy')

        train_labels = pd.read_csv('../DATASET/human_protein_atlas/all/train_ord.csv')
        target_dist = get_distribution(train_labels)
        target_dist = target_dist / target_dist.sum()

        sigma = 0.005
        singl = [-1, 1]
        thresholds = np.ones(28)
        thresholds *= 0.5

        label_predict = (score_predict >= thresholds)
        make_subs(labels, label_predict, f'./submissions/{name}_baseline.csv')
        classes_sum = label_predict.sum(axis=0)
        print('Baseline')
        print('Number of classes: ')
        print(classes_sum)
        dist = classes_sum / classes_sum.sum()
        best_norm = np.linalg.norm(target_dist - dist)
        print('\nBaseline distribution: ')
        print(dist)
        print('\nTarget distribution:')
        print(target_dist)
        print(f'\nThe best norm: {best_norm}\n\n\n')
        for i in tqdm(range(100000), total=100000):
            r = randrange(0, 28)
            s = rn.choices(singl)[0]
            new_thresholds = thresholds.copy()
            new_thresholds[r] += s * sigma
            label_predict = (score_predict >= new_thresholds)
            classes_sum = label_predict.sum(axis=0)
            dist = classes_sum / classes_sum.sum()
            norm = np.linalg.norm(target_dist - dist)
            if norm <= best_norm:
                best_norm = norm
                thresholds = new_thresholds
                # print(f'best_norm: {best_norm}')
        print('The best')
        print('The best number of classes: ')
        print(classes_sum)
        print(f'\nThe best thresholds: ')
        print(thresholds)
        print(f'\nThe best distribution: ')
        print(dist)
        print(f'\nThe best norm: {best_norm}\n\n\n')
        best_label_predict = (score_predict >= thresholds)
        make_subs(labels, best_label_predict, f'./submissions/{name}_best_thresholds_{best_norm}.csv')
    elif 0:
        score_predict = np.load(f'./score_predict/{number}/{name}.npy')

        sigma = 0.005
        singl = [-1, 1]
        thresholds = np.ones(28)
        thresholds *= 0.15

        label_predict = (score_predict >= thresholds)
        make_subs(labels, label_predict, f'./submissions/{number}/{name}_baseline.csv')
        make_subs_by_classes(labels, label_predict, f'./submissions/{number}/{name}_baseline')
        if 0:
            classes_sum = label_predict.sum(axis=0)
            print('Baseline')
            print('Number of classes: ')
            print(classes_sum)
            best_norm = get_norm(classes_sum)
            print(f'\nThe best norm: {best_norm}\n\n\n')
            for i in tqdm(range(100000), total=100000):
                r = randrange(0, 28)
                s = rn.choices(singl)[0]
                new_thresholds = thresholds.copy()
                new_thresholds[r] += s * sigma
                label_predict = (score_predict >= new_thresholds)
                classes_sum = label_predict.sum(axis=0)
                # dist = classes_sum / classes_sum.sum()
                # norm = np.linalg.norm(target_dist - dist)
                norm = get_norm(classes_sum)
                if norm <= best_norm:
                    best_norm = norm
                    thresholds = new_thresholds
                    # print(f'best_norm: {best_norm}')
            print('The best')
            print('The best number of classes: ')
            print(classes_sum)
            print(f'\nThe best thresholds: ')
            print(thresholds)
            print(f'\nThe best norm: {best_norm}\n\n\n')
            best_label_predict = (score_predict >= thresholds)
            make_subs(labels, best_label_predict, f'./submissions/{number}/{name}_best_thresholds_{best_norm}')
            # make_subs_by_classes(labels, best_label_predict, f'./submissions/{number}/{name}_best_thresholds_{best_norm}')
    else:
        score_predict = np.load(f'./score_predict/{number}/{name}.npy')

        class_num = 28
        sigma = 0.005
        singl = [-1, 1]
        thresholds = np.ones(class_num)
        thresholds *= 0.15

        label_predict = (score_predict >= thresholds)
        make_subs(labels, label_predict, f'./submissions/{number}/{name}_baseline.csv', class_num)
        make_subs_by_classes(labels, label_predict, f'./submissions/{number}/{name}_baseline', class_num)


def one_hot_target(target):
    one_hot = np.zeros(28, dtype=np.uint8)
    if target != '':
        targets = np.array(target.split(' ')).astype(np.int).tolist()
        for t in targets:
            one_hot[t] = 1
    return one_hot


def target_from_one_hot(one_hot):
    labels = np.arange(28)[one_hot.astype(np.bool)]
    label = ' '.join(str(l) for l in labels)
    return label


def ordered_labels(file_in, file_out):
    labels = pd.read_csv(file_in)
    for index, row in labels.iterrows():
        target_ord = one_hot_target(row['Target'])
        target_ord = target_from_one_hot(target_ord)
        if row['Target'] != target_ord:
            row['Target'] = target_ord
    labels.to_csv(file_out, header=True, index=False)


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


def dataset_statistic(file='../DATASET/human_protein_atlas/all/train_ord.csv'):
    sort_complex_target = False
    labels = pd.read_csv(file)
    plain_targets = {}
    for c in range(28):
        plain_targets[str(c)] = 0
    complex_targets = {}

    # Статистика простых и составных классов по всему файлу
    if 1:
        for idx, target in tqdm(zip(labels['Id'], labels['Target']), total=len(labels)):
            if sort_complex_target:
                os.makedirs(f'./classes_example/complex/{target}/', exist_ok=True)
                image = load_image('../DATASET/human_protein_atlas/all/train/', idx)
                imsave(f'./classes_example/complex/{target}/{idx}.jpg', image[:, :, :3])
            if target in complex_targets:
                complex_targets[target] += 1
            else:
                complex_targets[target] = 1

            for t in target.split(' '):
                if t in plain_targets:
                    plain_targets[t] += 1
                else:
                    plain_targets[t] = 1

    # Статистика простых и составных классов по избранным классам
    if 0:
        def retarget(targets_):
            maping = {'0': '0',
                      '8': '1',
                      '9': '2',
                      '10': '3',
                      '11': '4',
                      '12': '5',
                      '13': '6',
                      '15': '7',
                      '16': '8',
                      '17': '9',
                      '18': '10',
                      '20': '11',
                      '24': '12',
                      '26': '13',
                      '27': '14'}
            new_targets_ = []
            for n_, tr in enumerate(targets_):
                new_targets_.append(maping[str(tr)])
            nt = ' '.join(str(l) for l in new_targets_)
            return nt

        new_idxs = []
        new_targets = []

        remove_ = [0, 1, 2, 3, 4, 5, 6, 7, 14, 19, 21, 22, 23, 25]
        remove = {}
        for r in remove_:
            remove[str(r)] = 200
            plain_targets[str(r)] = 0
        spec = [8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 24, 26, 27]
        for idx, target in tqdm(zip(labels['Id'], labels['Target']), total=len(labels)):
            # add 200 removed classes
            if target in remove:
                if remove[target] > 0:
                    plain_targets[target] += 1
                    remove[target] -= 1
                    new_idxs.append(idx)
                    new_targets.append('0')

            # make zero all not spec classes
            targets = np.array(target.split(' ')).astype(np.uint8).tolist()
            for n, t in enumerate(targets):
                if t not in spec:
                    targets[n] = 0
            targets = list(set(targets))

            # calculate statistic
            for t_ in targets:
                if t_ in spec:
                    target = ' '.join(str(l) for l in targets)

                    new_idxs.append(idx)
                    new_targets.append(retarget(targets))

                    if target in complex_targets:
                        complex_targets[target] += 1
                    else:
                        complex_targets[target] = 1
                    for t in targets:
                        t = str(t)
                        if t in plain_targets:
                            plain_targets[t] += 1
                        else:
                            plain_targets[t] = 1
                    break
        new_labels = pd.DataFrame()
        new_labels['Id'] = new_idxs
        new_labels['Target'] = new_targets
        new_labels.to_csv('./train_smallest_15.csv', header=True, index=False)

    if 0:
        hier_targets = []
        for i in range(28):
            hier_targets.append({})

        for k in plain_targets:
            plain_targets[k] = 0

        spec = ['9', '8', '15', '27', '20']

        for k in complex_targets:
            count = complex_targets[k]
            # if count == 1:
            #     count = 8
            # elif count == 2:
            #     count = 32
            # elif count == 3:
            #     count = 72
            # elif count == 4:
            #     count = 128
            #
            # complex_targets[k] = count

            for t in k.split(' '):
                if t in spec and k not in spec:
                    count = count * count * 8
                    complex_targets[k] = count
                plain_targets[t] += count
                if k in hier_targets[int(t)]:
                    hier_targets[int(t)][k] += count
                else:
                    hier_targets[int(t)][k] = count

    print('Plain targets:')
    for k in plain_targets:
        print(f'{k} : {plain_targets[k]}')
    print('\n\n\n\nComplex targets:')
    for k in complex_targets:
        print(f'\'{k}\' : {complex_targets[k]}')

    if 0:
        print('\n\n\n\nHierarcy targets:')
        for k in range(len(hier_targets)):
            print(f'{k}')
            for i in hier_targets[k]:
                print(f'    \'{i}\' : {hier_targets[k][i]}')


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
        target = rn.choice(dist[str(k)])
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
    labels_idx = labels.Id.tolist()

    preprocessor = ImagePreprocessor(parameter)

    training_generator = DataGenerator(train_ids, labels, parameter, preprocessor)
    validation_generator = DataGenerator(valid_ids, labels, parameter, preprocessor, validation=True)
    predict_generator = PredictGenerator(valid_ids, preprocessor, train_path)

    model = BaseLineModel(parameter)
    # model.build_model()
    # model.compile_model()
    name = '29-56ep-0.5247'
    model.load(f'./models/29/{name}.h5', custom_objects={'f1_loss': f1_loss, 'f1': f1})
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

    score_macro = f1_score(y_true, y_pred, average='macro')
    print(f'score_macro: {score_macro}')
    score_micro = f1_score(y_true, y_pred, average='micro')
    print(f'score_micro: {score_micro}')
    score_samples = f1_score(y_true, y_pred, average='samples')
    print(f'score_samples: {score_samples}')


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


def get_distribution(labels):
    dist = np.zeros(28)
    for idx, target in zip(labels['Id'], labels['Target']):
        classes = target.split(' ')
        for cls in classes:
            dist[int(cls)] += 1
    return dist


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
        self.number = number
        if tune == 'lr':
            self.log_dir = 'logs/{}/{}_{}-{}-{:.8f}'.format(number, number, arch, tune, self.__getattribute__(tune))
            self.model_name = f'./models/{number}/{number}'
            self.model_name = self.model_name + '-{epoch:02d}ep-{val_f1:.4f}.h5'
        elif tune == 'fcl':
            self.log_dir = 'logs/{}/{}_{}-{}-({})'.format(number, number, arch, tune, self.fcl)
            self.model_name = f'./models/{number}/{number}'
            self.model_name = self.model_name + '-{epoch:02d}ep-{val_f1:.4f}.h5'
        else:
            self.log_dir = 'logs/{}/{}_{}-{}-{}'.format(number, number, arch, tune, self.__getattribute__(tune))
            self.model_name = f'./models/{number}/{number}'
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
        self.augmentation = self.strong_aug_02()

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
                if (self.parameter.row_scale_factor + self.parameter.col_scale_factor) > 2:
                    image = self.crop_random(image)
                data = {"image": image}
                image = self.augmentation(**data)["image"]
        else:
            if (self.parameter.row_scale_factor + self.parameter.col_scale_factor) > 2:
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

    def strong_aug_02(self, p=0.9):
        return Compose([
            OneOf([
                RandomRotate90(),
                Flip(),
                Transpose(),
                # ElasticTransform(),
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


class DataGeneratorTrainBalance(keras.utils.Sequence):

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
            idx = rn.choice(self.dist_idxs[str(k)])
            if self.use_memory:
                image = self.dataset[self.nlabels[idx]]
            else:
                image = self.preprocessor.load_image(idx)
            image = self.preprocessor.preprocess(image, self.validation)
            X[i] = image
            # Store class
            y[i] = self.get_targets_per_image(idx)

        return X, y


class DataGeneratorTrainBalanceNew(keras.utils.Sequence):

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
        self.dist_base_classes, self.dist_classes, self.idx_class, self.max_len_base_class = self.__make_dist_dataset()

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
        labels = pd.read_csv('./folds/train_1.csv')
        dist_base_classes = {}
        dist_classes = {}
        idx_class = {}

        for i in range(28):
            dist_base_classes[str(i)] = []

        for idx, target in tqdm(zip(labels['Id'], labels['Target']), total=len(labels)):
            idx_class[idx] = target

            if target in dist_classes:
                dist_classes[target].append(idx)
            else:
                dist_classes[target] = []
                dist_classes[target].append(idx)

            for t in target.split(' '):
                dist_base_classes[t].append(idx)

        max_len_base_class = 0
        for k in dist_base_classes:
            if len(dist_base_classes[k]) > max_len_base_class:
                max_len_base_class = len(dist_base_classes[k])

        return dist_base_classes, dist_classes, idx_class, max_len_base_class

    def __make_new_train_example(self, class_):
        lr = np.random.choice([True, False])
        class_idxs = self.dist_classes[class_]
        idx1 = class_idxs[randrange(0, len(class_idxs))]
        idx2 = class_idxs[randrange(0, len(class_idxs))]
        if self.use_memory:
            img1 = self.dataset[self.nlabels[idx1]]
            img2 = self.dataset[self.nlabels[idx2]]
        else:
            img1 = self.preprocessor.load_image(idx1)
            img2 = self.preprocessor.load_image(idx2)

        img1 = self.preprocessor.preprocess(img1, False)
        img2 = self.preprocessor.preprocess(img2, False)

        image = np.zeros((512, 512, 4), dtype=np.float16)
        if lr:
            image[:, :256, :] = img1[:, :256, :]
            image[:, 256:, :] = img2[:, 256:, :]
        else:
            image[:256, :, :] = img1[:256, :, :]
            image[256:, :, :] = img2[256:, :, :]
        return image, idx1

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(23000 / self.batch_size)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
        y = np.empty((self.batch_size, self.num_classes), dtype=np.uint8)

        for i in range(self.batch_size):
            base_class = str(randrange(0, 28))
            idxn = randrange(0, self.max_len_base_class)
            if idxn >= len(self.dist_base_classes[base_class]):
                idxn = randrange(0, len(self.dist_base_classes[base_class]))
                cls = self.idx_class[self.dist_base_classes[base_class][idxn]]
                image, idx = self.__make_new_train_example(cls)
            else:
                idx = self.dist_base_classes[base_class][idxn]
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
                                             use_multiprocessing=False,
                                             workers=16,
                                             verbose=1)

    def predict(self, predict_generator):
        y = predict_generator.predict(self.model)
        return y

    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)

    def load(self, modelinputpath, custom_objects={}):
        self.model = load_model(modelinputpath, custom_objects=custom_objects)


def train_gpu(gpu, fold, train_path, labels, parameter, train_generator=''):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    os.makedirs(f'models/{parameter.number}/', exist_ok=True)

    train_labels = pd.read_csv(f'./folds/train_{fold}.csv')
    valid_labels = pd.read_csv(f'./folds/valid_{fold}.csv')
    train_ids = train_labels.Id.tolist()
    valid_ids = valid_labels.Id.tolist()

    preprocessor = ImagePreprocessor(parameter)

    if train_generator == 'train_balance':
        print(f'train_generator = {train_generator}')
        training_generator = DataGeneratorTrainBalanceNew(train_ids, labels, parameter, preprocessor)
    else:
        training_generator = DataGenerator(train_ids, labels, parameter, preprocessor)
    validation_generator = DataGenerator(valid_ids, labels, parameter, preprocessor, validation=True)
    predict_generator = PredictGenerator(valid_ids, preprocessor, train_path)

    model = BaseLineModel(parameter)
    model.build_model()
    model.compile_model()
    # name = ''
    # model.load(f'./models/batch_size/{name}.h5', custom_objects={'f1_loss': f1_loss, 'f1': f1})
    model.set_generators(training_generator, validation_generator)
    history = model.learn()

    # os.makedirs(f'models/{parameter.tune}/', exist_ok=True)
    # model.save(parameter.model_name)


def main():
    train_path = '../DATASET/human_protein_atlas/all/train/'
    labels = get_labels('../DATASET/human_protein_atlas/all/train_ord.csv')

    # dataset = None
    # dataset = np.zeros((len(labels), 512, 512, 4), dtype=np.uint8)
    # for n, idx in tqdm(enumerate(labels['Id'].tolist()), total=len(labels)):
    #     image = load_image(train_path, idx)
    #     dataset[n, :, :, :] = image
    # np.save('./train_512x512x4.npy', dataset)
    dataset = np.load('./train_512x512x4.npy')

    param1 = ModelParameter(train_path,
                            lr=0.00003,
                            fcl=[512],
                            row_scale_factor=1,
                            col_scale_factor=1,
                            batch_size=30,
                            n_epochs=100,
                            tune='batch_size',
                            arch='resnet18',
                            dataset=dataset,
                            aug='',
                            number=34,
                            shuffle=True)

    train_gpu('0', 1, train_path, labels, param1, 'train_balance')

    # p1 = Process(target=train_gpu, args=('0', 1, train_path, labels, param1, 'train_balance'))
    #
    # param2 = ModelParameter(train_path,
    #                         lr=0.00003,
    #                         fcl=[1024],
    #                         batch_size=100,
    #                         n_epochs=100,
    #                         tune='batch_size',
    #                         arch='resnet18',
    #                         dataset=dataset,
    #                         aug='strong_aug',
    #                         number=30,
    #                         shuffle=True)
    #
    # p2 = Process(target=train_gpu, args=('1', 1, train_path, labels, param2, ''))
    #
    # p1.start()
    # p2.start()
    #
    # p1.join()
    # p2.join()


if __name__ == '__main__':
    # ordered_labels('../DATASET/human_protein_atlas/extended.csv', '../DATASET/human_protein_atlas/extended_ord.csv')
    # dataset_statistic('../DATASET/human_protein_atlas/extended_ord.csv')
    # dataset_statistic('./train_smallest_15.csv')
    # main()
    predict_submission('checkpoint-ep20-f1(0.3507)-PL(0.493)', 111111, TTA=False)
    # predict_submission('31-52ep-0.4585', 31, TTA=False)
