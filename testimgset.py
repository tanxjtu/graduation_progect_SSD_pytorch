from data.Dataset import GetDataset
# from Dataset import GetDataset  #change back
import os.path as os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


#from .utils.augmentations import SSDAugmentation

# from data import  ACT_utils import *
#from ACT_utils import *   #change back
from data.ACT_utils import *
from copy import deepcopy
import os
import pickle
# from PIL import Image
# import matplotlib.pyplot as plt

UCFSportsCLASS = ('Diving', 'Golf', 'Kicking', 'Lifting',
                  'Riding', 'Run', 'SkateBoarding', 'Swing1',
                  'Swing2', 'Walk')

d = GetDataset('UCFSports')
labels = d.nlabels
torch_Path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UCFSports_Images_Dir = os.path.join(torch_Path, 'Frames')
result = {}

d = GetDataset('UCFSports')
num = 0
for testlist in d.test_vlist():
    # for testlist in d.train_vlist():
    each_video_dir = os.path.join(UCFSports_Images_Dir, testlist)
    video_frame_num = d.nframes(testlist)
    labels = list(d._gttubes[testlist].keys())[0]
    resoliution = d._resolution[testlist]
    for nframe in range(video_frame_num):
        num = num + 1
        imgdir = os.path.join(
            each_video_dir, "{:0>6}".format(nframe + 1) + ".jpg")
            # imgdir, testlist, nframe, labels, resoliution, num
        temp = {}
        temp['name'] = 'train'
        temp['dir'] = imgdir[-15:-4]
        temp['bbox'] = d._gttubes[testlist][labels][0][nframe,1:].tolist()
        temp['difficult'] = 0
        temp['labels'] = labels
        result["{:0>6}".format(num)] = [temp]

        # result["{:0>6}".format(num)] =  [{'name': 'train', 'truncated': 0, 'pose': 'Unspecified', 'bbox': [138, 199, 206, 300], 'difficult': 0}]
det_file = os.path.join(os.getcwd(),'result_UCFS','gt','detections.pkl')
with open(det_file, 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)



filename = os.path.join(os.getcwd(), 'result_UCFS', 'Main.txt')
with open(filename, 'wt') as f:
    for i in range(len(result)):
        f.write('{:s} \n'.format("{:0>6}".format(i+1)))

print('need ot save result pkl')
print('1123')