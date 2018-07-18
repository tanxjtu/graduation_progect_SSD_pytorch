"""
UCFSports Dataset Class
author: Haoliang Tan XJTU
Date : 2018.5.7
"""

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
from PIL import Image
import matplotlib.pyplot as plt

UCFSportsCLASS = ('Diving', 'Golf', 'Kicking', 'Lifting',
                  'Riding', 'Run', 'SkateBoarding', 'Swing1',
                  'Swing2', 'Walk')
d = GetDataset('UCFSports')
labels = d.nlabels
torch_Path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UCFSports_Images_Dir = os.path.join(torch_Path, 'Frames')
#UCFSports_Images_Dir = '/home/thl/Desktop/Frames'


class UCFSportsDetection(data.Dataset):
    """
        UCFSports Detection Dataset Object
        input is image, target is annotation
        Arguments:
            root (string): filepath to UCFSports folder.
            image_set (string): imageset to use (eg. 'train','test')
            transform (callable, optional): transformation to perform on the
                input image
            target_transform (callable, optional): transformation to perform on the
                target `annotation`
                (eg: take in caption string, return tensor of word indices)
            dataset_name (string, optional): which dataset to load
    """

    def __init__(self, root=UCFSports_Images_Dir, transform=None, target_transform=None, dataset_name='UCFSports'):
        self.root = root
        self.name = dataset_name
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = None
        self._imgpath = None
        self.ids = list()

        d = GetDataset(self.name)
        num = 0
        for testlist in d.test_vlist():
        #for testlist in d.train_vlist():
            each_video_dir = os.path.join(UCFSports_Images_Dir, testlist)
            video_frame_num = d.nframes(testlist)
            labels = list(d._gttubes[testlist].keys())[0]
            resoliution = d._resolution[testlist]
            for nframe in range(video_frame_num):
                num = num + 1
                imgdir = os.path.join(
                    each_video_dir, "{:0>6}".format(nframe + 1) + ".jpg")
                self.ids.append(
                    (imgdir, testlist, nframe, labels, resoliution,num))
    def __getitem__(self, index):
        # im, gt, h, w = self.pull_item(index)
        # im = self.get_im_transed(index)
        # gt = self.get_gt()
        im, gt = self.get_im_transed(index)

        return im, gt  # Tensor  im 3 300 300 gt x1,y1,x2,y2 class

    def __len__(self):
        return len(self.ids)

    def pull_item(self,index):
        sig_im_dir = self.ids[index][0]
        img = cv2.imread(sig_im_dir)
        height, width, channels = img.shape
        gt = self.get_gt(index)
        if self.transform is not None:
            gt = np.array(gt)
            img,_,_ = self.transform(img)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            # target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), gt, height, width
        # return img,height,width,channels,gt

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def get_gt(self, index):
        videos = self.ids[index][1]
        nframes = self.ids[index][2]
        labels = self.ids[index][3]
        x1, y1, x2, y2 = d._gttubes[videos][labels][0][nframes][1:]
        resoy, resox = self.ids[index][4][0], self.ids[index][4][1]
        y1 = y1 / resoy
        y2 = y2 / resoy
        x1 = x1 / resox
        x2 = x2 / resox
        gt = [list((x1, y1, x2, y2, labels))]
        gt = np.array(gt)
        return gt

    def get_im_transed(self, index):
        impath = self.ids[index][0]
        img = cv2.imread(impath)
        height, width, channels = img.shape
        target = self.get_gt(index)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        return torch.from_numpy(img).permute(2, 0, 1), target


# test = UCFSportsDetection(transform=SSDAugmentation(300,(104,117,123)))
# test = UCFSportsDetection(transform=BaseTransform(300,(104, 117, 123)) )
# a = test.pull_item(2000)
# print('test')
# show = a[0].numpy().transpose(1,2,0)
# show = show[:, :, (2, 1, 0)]
# plt.figure("testIMg")
# plt.imshow(show)
# plt.show()
# print('test')
# c = test.get_gt(0)# index
# lenn = test.__len__()

# dataset = UCFSportsDetection(transform=SSDAugmentation(300,(104,117,123)))
# ground_truth = test.get_gt(2460)

# a,b = test.get_im_transed(1)
# img_to_show = a.numpy().transpose(1,2,0)
# box = b*300
# plt.figure('test img ')
# plt.imshow(img_to_show)
# plt.show()
# print('test')
