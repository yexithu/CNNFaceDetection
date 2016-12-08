from settings import *
import os
import random


RATIO = 1 / 1  # k * original + 1 * new

single_list = []

with open(SINGLEFACE_LIST) as f:
	single_list = f.readlines()
	single_list = map(lambda x: x.split()[0], single_list)


def print_list(list_, file_):
    for line in list_:
        file_.write(" ".join(line))
        file_.write("\n")
    print(len(list_))


def extend_list(input_name, output_name):
    input_file = open(input_name)
    originals = [line.strip().split(' ') for line in input_file]

    positives = filter(lambda x: x[1] == '1', originals)
    negatives = filter(lambda x: x[1] == '0', originals)

    bootstraps = []
    i = 0
    positives = {a[0]:a[1] for a in positives}
    singleface_positives = filter(lambda x: x[0] in single_list, positives)
    for _ in singleface_positives:
        filename = _
        i += 1
        if i % 100 == 0: print(str(i)+'\t'+filename)
        prefix = os.path.join(BOOTSTRAP_ROOT, filename[11:-4])
        files = os.listdir(prefix)

        false_positives = [
            [os.path.join(prefix, file_), '0']
            for file_ in files
        ]
        bootstraps = bootstraps + false_positives

    print(len(bootstraps))
    # intersect single_list
    output_file = open(output_name, 'w')

    # k * negative ~ 1 * bootstrap
    if len(bootstraps) < RATIO * len(negatives):
        # too many negatives
        random.shuffle(negatives)
        negatives = negatives[:len(bootstraps) * RATIO]
    else:  # too many bootstrap
        random.shuffle(bootstraps)
        bootstraps = bootstraps[:len(negatives) / RATIO]

    print_list(negatives, file_=output_file)
    print_list(bootstraps, file_=output_file)


def main():
    for key, value in FACE_NONFACE_TRAIN_LISTS.items()[:1]:
        extend_list(value, BS_TRAIN_LISTS[key])    

main()
