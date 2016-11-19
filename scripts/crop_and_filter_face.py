import cv2
import csv
from os.path import join
from settings import *

# 0 front 2 left 3  right
faces = []

def main():
    with open(FACES_TSV, 'r') as tsv:
        count = 0
        for line in csv.reader(tsv, delimiter='\t'):
            count = count + 1
            if count % 100 == 0:
                print count
            crop_face(line)
    with open(FACES_LIST, 'w') as f:
        f.writelines(faces)

def crop_face(line):
    src = join(AFLW_ROOT, line[0])
    faceid = line[1]
    facey = int(line[2])
    facex = int(line[3])
    facew = int(line[4])
    faceh = int(line[5])
    gender = line[6]

    img = cv2.imread(src)
    rows, cols, channel = img.shape
    if facex < 0 or facey < 0:
        return
    if ((facex + faceh) >= rows) or ((facey + facew) >= cols):
        return
    croped = img[facex: facex + faceh, facey: facey + facew]
    dst = join(CROPED_ROOT, line[0])
    cv2.imwrite(dst, croped)
    faces.append('%s\t%s\n' % (line[0], gender))


if __name__ == '__main__':
    main()


