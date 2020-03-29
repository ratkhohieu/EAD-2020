from PIL import Image
import imagehash

import glob
import os.path as osp
import os
import numpy as np

images_2k = '/home/sven/Documents/EAD2019-master/data/EAD2020-Phase-II-Detection_Segmentation-99Frames/originalImages/'
images_files_list_2k = glob.glob(os.path.join(images_2k, '*.jpg'))
print(len(images_files_list_2k))

# for i in images_files_list_4k:
#     hash = imagehash.average_hash(Image.open(i))
#     for j in images_files_list_2k:
#         otherhash = imagehash.average_hash(Image.open(j))
#
#         if hash == otherhash:
#             match = match + 1
#             print(i)
#             print(j)
# print('match = ', match)


with open('train_8classes3.txt', 'w+') as f:
    for item in images_files_list_2k:
        item = item.split('/home/sven/Documents/EAD2019-master/data/EAD2020-Phase-II-Detection_Segmentation-99Frames/originalImages/')
        # item = item.strip('yolo2voc/')
        item = item[1].strip('.jpg')
        print(item)
        f.write("%s\n" % item)
