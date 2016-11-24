import os
from os.path import join
import csv
import cv2
from settings import *
import shutil


INITSIZE = 25
SCALE = 2
INITSTRIDE = 10

MAXSIZE = 50

def clean_nonfaces():
    shutil.rmtree(NONFACES_ROOT)
    os.mkdir(NONFACES_ROOT)



def gen_nonfaces(line):

    prefix = os.path.splitext(line[0])[0]
    prefix = join(NONFACES_ROOT, prefix)
    os.makedirs(prefix)
    # raw_input()
    src = join(AFLW_ROOT, line[0])
    faceid = line[1]
    facex = int(line[2])
    facey = int(line[3])
    facew = int(line[4])
    faceh = int(line[5])
    gender = line[6]

    img = cv2.imread(src)
    rows, cols, channel = img.shape

    win_size = INITSIZE
    stride = INITSTRIDE
    img_size = (cols, rows)
    face_p = (facex, facey)
    face_size = (facew, faceh)

    def check(x, y, l):
        if ((x + l) > cols) or ((y + l) > rows):
            return False

        #judge overlap
        x_in_range = ((x + l) > facex) and (x < (facex + facew))
        y_in_range = ((y + l) > facey) and (y < (facey + faceh))
        if x_in_range and y_in_range:
            return False

        return True

    img_count = 0
    while win_size < rows and win_size < cols:
        y = 0
        while y < rows:
            x = 0
            while x < cols:
                if not check(x, y, win_size):
                    x += stride
                    continue
                patch = img[y: y + win_size, x: x + win_size]
                if win_size > MAXSIZE:
                    patch = cv2.resize(patch, (MAXSIZE, MAXSIZE))
                cv2.imwrite(os.path.join(prefix, str(img_count) + ".jpg"), patch)

                img_count += 1
                x += stride
            y += stride            

        win_size = round(win_size * SCALE)
        stride = round(stride * SCALE)



def main():
    clean_nonfaces()

    single_list = []
    lines = []

    with open(SINGLEFACE_LIST) as f:
        single_list = f.readlines()
        single_list = map(lambda x: x.strip(), single_list)

    count = 0
    in_count = 0
    print "Len Single", len(single_list)    
    with open(FACES_TSV, 'r') as tsv:
        for line in csv.reader(tsv, delimiter='\t'):
            count += 1
            if count % 1000 == 0:
                print count
            if not line[0] in single_list:
                continue
            gen_nonfaces(line)
            in_count += 1
    print "Count", count
    print "In Count", in_count

            


if __name__ == "__main__":
    main()