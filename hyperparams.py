import numpy as np
import random


def get_lr_random(num, a, b, bash=False):
    lr = np.zeros(num)

    for i in range(num - 2):
        r = random.uniform(a, b)
        alpha = np.power(10, r)
        lr[i] = alpha

    lr[num - 2] = np.power(10, a)
    lr[num - 1] = np.power(10, b)

    lr.sort()

    for i in range(len(lr)):
        if bash:
            print('python ./resnet_50.py --gpu 0 --batch 10 --fold 0 --epochs 1 --lr {:.8f} --units 0 --tune lr'.format(lr[i]))
        else:
            print('alpha={:.8f}'.format(lr[i]))


def get_number_units(num, a, b, bash=False):
    lr = np.zeros(num)

    for i in range(num - 2):
        r = random.randint(a, b)
        lr[i] = r

    lr[num - 2] = a
    lr[num - 1] = b

    lr.sort()

    for i in range(len(lr)):
        if bash:
            print('python ./resnet_50.py --gpu 1 --batch 6 --fold 0 --epochs 1 --lr 0.0001 --units {:.0f} --tune fcl_units'.format(lr[i]))
        else:
            print('units={}'.format(lr[i]))


print('learning rates')
get_lr_random(22, -3.0, -5.0, True)

print('number of units for FCL')
get_number_units(22, 128, 1024, True)