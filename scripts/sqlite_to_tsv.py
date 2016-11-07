import sqlite3
conn = sqlite3.connect('/home/yexi/Desktop/AFLW/aflw-db/data/aflw.sqlite')
conn.row_factory = sqlite3.Row
cur = conn.cursor()
import re
pattern = re.compile('image(\d+).jpg')
from itertools import chain as flatten
import os

def produce_line(filename):
    cur.execute('''
            select face_id
            from Faces
            where file_id = ?
        ''',(
            filename,
    ))
    face_id = cur.fetchone()
    if face_id is None:
        return
    face_id = face_id[0]
    cur.execute('''
            select x,y,w,h
            from FaceRect
            where face_id = ?
        ''',(
            face_id,
    ))
    rect = cur.fetchone()
    cur.execute('''
            select sex, glasses
            from FaceMetaData
            where face_id = ?
        ''',(
            face_id,
    ))
    meta = cur.fetchone()
    return "\t".join(map(str, flatten([
        filename,
        face_id,
        rect['x'],
        rect['y'],
        rect['w'],
        rect['h'],
        meta['sex'],
    ])))

root = '/home/yexi/Desktop/AFLW/aflw/data/flickr'
output_filename = 'data/faces.tsv'
output_file = open(output_filename, 'w')
success = 0
fail = 0
inner_prefix = ['0', '2', '3']

for prefix in inner_prefix:
    for img in os.listdir(os.path.join(root, prefix)):
        line = produce_line(img)
        if line:
            line = os.path.join(prefix, line)
            # print(line, file=output_file)
            output_file.write(line)
            output_file.write('\n')
            success = success + 1
        else:
            print('{} not found in database'.format(img))
            fail = fail + 1
        if success % 100 == 0:
            print(success)

print('success {}, fail {}'.format(success, fail))
