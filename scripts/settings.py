AFLW_SQILTE = '/home/yexi/Desktop/AFLW/aflw-db/data/aflw.sqlite'

AFLW_ROOT = '/home/yexi/Desktop/AFLW/aflw/data/flickr'

FACES_TSV = 'data/faces.tsv'

CROPED_ROOT = 'data/faces/'

RANIMG_ROOT = 'data/nonfaces/'

SCENES_ROOT = 'data/scenes/'

FACES_LIST = 'data/faces.list'

SINGLEFACE_LIST = 'data/singleface.list'

NONFACES_ROOT = 'data/nonfaces/'

# train using faces and ran imgs
NAIVE_FACE_TRAIN_LISTS = {
    'train' : 'data/lists/naive_train.list',
    'cross_val': 'data/lists/naive_cross_val.list',
    'test' : 'data/lists/naive_test.list'
}

# train using faces and non faces
FACE_NONFACE_TRAIN_LISTS = {
    'train' : 'data/lists/face_nonface_train.list',
    'cross_val': 'data/lists/face_nonface_cross_val.list',
    'test' : 'data/lists/face_nonface_test.list'    
}