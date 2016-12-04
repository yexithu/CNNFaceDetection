from settings import *
import os
import random


RATIO = 1 / 1  # k * original + 1 * new


def print_list(list_, file_):
    for line in list_:
        file_.write(" ".join(line))
        file_.write("\n")


def extend_list(input_name, output_name):
    input_file = open(input_name)
    originals = [line.strip().split(' ') for line in input_file]

    positives = filter(lambda x: x[1] == '1', originals)
    negatives = filter(lambda x: x[1] == '0', originals)

    bootstraps = []
    for _ in positives:
        filename = _[0]
        prefix = os.path.join(BOOTSTRAP_ROOT, filename[:-4])
        files = os.listdir(prefix)

        false_positives = [
            [os.path.join(prefix, file_), '0']
            for file_ in files
        ]
        bootstraps = bootstraps.extend(false_positives)

    output_file = open(output_file, 'w')

    # k * negative ~ 1 * bootstrap
    if len(bootstraps) < RATIO * len(negatives):
        # too many negatives
        random.shuffle(negatives)
        negatives = negatives[:len(bootstraps) * RATIO]
    else:  # too many bootstrap
        random.shuffle(bootstraps)
        bootstraps = bootstraps[:len(negatives) / RATIO]

    print_list(negatives, file=output_file)
    print_list(positives, file=output_file)


def main():
    for key, value in FACE_NONFACE_TRAIN_LISTS.items():
        extend_list(value, BS_TRAIN_LISTS[key])
