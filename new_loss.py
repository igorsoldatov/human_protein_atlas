from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import numpy as np
# from prettytable import PrettyTable
# from prettytable import ALL
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

# ground truth
Y = np.zeros((5, 2))
# first label is assigned to 20 % of observations
Y[0, 0] = 1
# second label is assigned to 80 % of observations
Y[0:4, 1] = 1

# ground truth with shape (BATCH_SIZE, NO_OF_LABELS)
print(Y)

first = {}

for TP in range(2): # TP can be 0..1
    for FP in reversed(range(5)): # FP can be 0..4
        idx = TP*5+(4-FP)
        name = 'TP' + str(TP) + 'FP' + str(FP)
        Yhat1 = np.zeros(5)
        Yhat1[0:TP] = 1
        Yhat1[5-FP:] = 1
        first.update({name: Yhat1})

second = {}

for TP in range(5): # TP can be 0..4
    for FP in reversed(range(2)): # FP can be 0..1
        idx = TP*5+(4-FP)
        name = 'TP' + str(TP) + 'FP' + str(FP)
        Yhat2 = np.zeros(5)
        Yhat2[0:TP] = 1
        Yhat2[5-FP:] = 1
        second.update({name: Yhat2})