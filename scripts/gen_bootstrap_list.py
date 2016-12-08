from settings import *
import os
import random


RATIO = 1 / 1  # k * original + 1 * new

single_list = []

with open(SINGLEFACE_LIST) as f:
	single_list = f.readlines()
	single_list = map(lambda x: x.strip(), single_list)


def print_list(list_, file_):
    for line in list_:
        file_.write(" ".join(line))
        file_.write("\n")
    print(len(list_))


def usample(l, n):
    new_l = []
    for i in range(n):
        new_l += l
    return new_l

def extend_list(input_name, output_name):
    # input_file = open(input_name)
    # originals = [line.strip().split(' ') for line in input_file]
    originals = []
    with open(input_name) as f:
        originals = [line.strip().split(' ') for line in f.readlines()]
    positives = filter(lambda x: x[1] == '1', originals)
    negatives = filter(lambda x: x[1] == '0', originals)

    positives = positives
    negatives = negatives
    # print positives
    # print negatives
    
    pos_names = [x[0] for x in positives]
    pos_namesset = set(pos_names)
    pos_names = [x for x in pos_namesset]
    singleface_posnames = filter(lambda x: ('data/faces/' + x) in pos_names, single_list)
    positives = [[x, '1'] for x in pos_names]

    # print len(pos_names)
    # print len(singleface_posnames)
    i = 0
    bootstraps = []
    for singleface in singleface_posnames:
        i += 1
        if i % 100 == 0: print(str(i))
        prefix = os.path.join(BOOTSTRAP_ROOT, singleface[:-4])
        files = os.listdir(prefix)
        false_positives = [
            [os.path.join(prefix, file_), '0']
            for file_ in files
        ]
        bootstraps = bootstraps + false_positives
    # print len(bootstraps)
    # k * negative ~ 1 * bootstrap
    if len(bootstraps) < RATIO * len(negatives):
        # too many negatives
        random.shuffle(negatives)
        negatives = negatives[:len(bootstraps) * RATIO]
    else:  # too many bootstrap
        random.shuffle(bootstraps)
        bootstraps = bootstraps[:len(negatives) / RATIO]

    new_negatives = negatives + bootstraps
    r = int(round(len(new_negatives) / len(positives)))
    new_posives = usample(positives, r)
    with open(output_name, 'w') as f:    
        print_list(new_posives, file_=f)
        print_list(new_negatives, file_=f)


def main():
    for key, value in FACE_NONFACE_TRAIN_LISTS.items():
        extend_list(value, BS_TRAIN_LISTS[key])


main()
