import os
from settings import *
from string import split
import random
from math import floor
# need naive_train naive_cross_val naive_test

wanted = ['0', '2', '3']
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

def dict_add(a, b):
    c = {}
    for k in a.keys():
        c[k] = a[k] + b[k]
    return c

def dict_insert_root(d, root):
    for k in d.keys():
        d[k] = map(lambda x: os.path.join(root, x), d[k])
    

def get_faces():
    faces = {}
    lines = []
    with open(FACES_LIST, 'r') as f:
        lines = f.readlines()
        lines = map(lambda x: split(x)[0], lines)
    lines_list = map(lambda x: filter(lambda y: y[0] == x, lines), wanted)
    # lines_list = map(lambda x: x[:10], lines_list)
    lines_list = map(list_split, lines_list)
    faces = reduce(dict_add, lines_list)
    dict_insert_root(faces, CROPED_ROOT)
    return faces

def get_nonfaces():
    nonfaces = {}
    lines = []
    lines = os.listdir(RANIMG_ROOT)
    # lines = lines[:10]
    nonfaces = list_split(lines)
    dict_insert_root(nonfaces, RANIMG_ROOT)
    return nonfaces

def balance(faces, nonfaces):
    def usample(l, n):
        new_l = []
        for i in range(n):
            new_l += l
        return new_l

    for k in nonfaces.keys():
        r = int(round(len(faces[k]) / len(nonfaces[k])))
        nonfaces[k] = usample(nonfaces[k], r)


def save_list(faces, nonfaces):
    for k in NAIVE_FACE_TRAIN_LISTS.keys():
        with open(NAIVE_FACE_TRAIN_LISTS[k], 'w') as f:
            f.writelines(map(lambda x: x + '\t1\n', faces[k]))
            f.writelines(map(lambda x: x + '\t0\n', nonfaces[k]))

def main():
    faces = get_faces()
    nonfaces = get_nonfaces()
    balance(faces, nonfaces)
    save_list(faces, nonfaces)    

if __name__ == '__main__':
    main()