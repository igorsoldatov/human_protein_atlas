import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
sns.set()

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import RepeatedKFold

data_labels = pd.read_csv("../DATASET/human_protein_atlas/all/train.csv")

splitter = RepeatedKFold(n_splits=4, n_repeats=1, random_state=0)

partitions = []

fold = 0

for train_idx, test_idx in splitter.split(data_labels.index.values):
    partition = {}
    partition["train"] = data_labels.Id.values[train_idx]
    partition["validation"] = data_labels.Id.values[test_idx]
    partitions.append(partition)
    print("TRAIN:", train_idx, "TEST:", test_idx)
    print("TRAIN:", len(train_idx), "TEST:", len(test_idx))

    train_labels = data_labels.loc[data_labels.Id.isin(partition["train"])]
    valid_labels = data_labels.loc[data_labels.Id.isin(partition["validation"])]
    fold = fold + 1
    train_labels.to_csv(f'./folds/train_{fold}.csv')
    valid_labels.to_csv(f'./folds/valid_{fold}.csv')

