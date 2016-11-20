import cv2
from os.path import join
import numpy as np
from settings import NONFACES_ROOT

ran_img_num = 5000;
ran_img_len = 2500
ran_img_size = (50, 50)

def main():
    for i in range(ran_img_num):
        if i % 100 == 0:
            print i
        img = np.zeros(ran_img_len, np.uint8)
        for j in range(ran_img_len):
            img[j] = np.random.randint(256)
        img = img.reshape(50, 50)
        
        fname = "ranimg_{}.jpg".format(i)
        fname = join(NONFACES_ROOT, fname)        
        cv2.imwrite(fname, img)

if __name__ == '__main__':
    main()