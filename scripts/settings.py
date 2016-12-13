AFLW_SQILTE = '/home/yexi/Desktop/AFLW/aflw-db/data/aflw.sqlite'

AFLW_ROOT = '/home/yexi/Desktop/AFLW/aflw/data/flickr'

FACES_TSV = 'data/faces.tsv'

CROPED_ROOT = 'data/faces/'

RANIMG_ROOT = 'data/nonfaces/'

SCENES_ROOT = 'data/scenes/'

FACES_LIST = 'data/faces.list'

SINGLEFACE_LIST = 'data/singleface.list'

NONFACES_ROOT = 'data/nonfaces/'

BOOTSTRAP_ROOT = 'data/bootstrap/'
BOOTSTRAP1_ROOT = 'data/bootstrap1/'
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

#bootstrap
BS_TRAIN_LISTS = {
    'train' : 'data/lists/bs_train.list',
    'cross_val': 'data/lists/bs_cross_val.list',
    'test' : 'data/lists/bs_tests.list'
}

BS1_TRAIN_LISTS = {
    'train' : 'data/lists/bs1_train.list',
    'cross_val': 'data/lists/bs1_cross_val.list',
    'test' : 'data/lists/bs1_tests.list'
}

FACE_BS1_TRAIN_LISTS = {
    'train' : 'data/lists/face_bs1_train.list',
    'cross_val': 'data/lists/face_bs1_cross_val.list',
    'test' : 'data/lists/face_bs1_tests.list'
}

GENDER_TRAIN_LISTS = {
    'train' : 'data/lists/gender_train.list',
    'cross_val': 'data/lists/gender_cross_val.list',
    'test' : 'data/lists/gender_test.list'
}
