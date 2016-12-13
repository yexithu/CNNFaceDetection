import os
from settings import *
from string import split
import random
from os.path import join
from math import floor

test_rate = 0.15

def list_split(x):
    d = {}
    random.shuffle(x)
    l = len(x)
    test_l = int(floor(l * test_rate))
    d['cross_val'] = x[0:test_l]
    d['test'] = x[test_l: 2 * test_l]
    d['train'] = x[2 * test_l:]
    return d

def main():
    face_list = []
    with open(FACES_LIST) as f:
        face_list = f.readlines()
        face_list = filter(lambda x: x[0]=='0', face_list)
        face_list = map(lambda x: split(x), face_list)
    face_d = list_split(face_list)

    for k in face_d.keys():
        with open(GENDER_TRAIN_LISTS[k], 'w') as f:
            def _(x):
                __ = 1
                if x[1] == 'f':
                    __ = 0
                return '%s %d\n' % (join(CROPED_ROOT, x[0]), __)
            lines = map(_, face_d[k]);
            f.writelines(lines)


if __name__ == '__main__':
    main()
