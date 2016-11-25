import os
from settings import *
from string import split
import random
from os.path import join
from math import floor
import csv
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

    for k in faces.keys():
        r = int(round(len(nonfaces[k]) / len(faces[k])))
        faces[k] = usample(faces[k], r)


def save_list(faces, nonfaces):
    for k in FACE_NONFACE_TRAIN_LISTS.keys():
        with open(FACE_NONFACE_TRAIN_LISTS[k], 'w') as f:
            f.writelines(map(lambda x: x + ' 1\n', faces[k]))
            f.writelines(map(lambda x: x + ' 0\n', nonfaces[k]))

def main():
    single_list = []
    cropped_list = []
    img_list = []

    with open(SINGLEFACE_LIST) as f:
        single_list = f.readlines()
        single_list = map(lambda x: x.strip(), single_list)

    with open(FACES_LIST) as f:
        cropped_list = f.readlines()
        cropped_list = map(lambda x: split(x)[0], cropped_list)

    with open(FACES_TSV, 'r') as tsv:
        for line in csv.reader(tsv, delimiter='\t'):
            img_list.append(line[0])

    def _assign(fil, to_filter):
        d = {}
        for k, v in to_filter.items():
            d[k] = [x for x in v if x in fil]
        return d

    img_split = list_split(img_list)
    cropped_split = _assign(cropped_list, img_split)
    single_split = _assign(single_list, img_split)
    dict_insert_root(cropped_split, CROPED_ROOT)
    dict_insert_root(single_split, NONFACES_ROOT)

    for k, v in single_split.items():
        new_v = []
        v = map(lambda x: os.path.splitext(x)[0], v)
        imgs_dir = map(lambda x: map( lambda y: join(x, y), os.listdir(x)), v)
        for imgs in imgs_dir:
            new_v.extend(imgs)
        single_split[k] = new_v

    balance(cropped_split, single_split)
    save_list(cropped_split, single_split)



if __name__ == '__main__':
    main()