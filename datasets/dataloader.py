import cv2
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import glob
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tensorflow as tf

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, input_path,label_path,label_name='',transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(input_path,'*.jpg'))
        self.mask_files = []
        self.transforms = transform
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(label_path,os.path.basename(img_path).split('.')[0]+label_name+'.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path)
            label = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            label = F.one_hot(torch.from_numpy(label).to(torch.int64),6)
            datalabel = np.concatenate((data,label),axis=2)
            datalabel = np.transpose(datalabel,[2,0,1])
            if self.transforms!=None:
              datalabel = self.transforms(torch.from_numpy(datalabel).float())
            datalabel = np.transpose(datalabel,[1,2,0])
            data = datalabel[:,:,0:3]
            label = datalabel[:,:,3:9]
            return data,label

    def __len__(self):
        return len(self.img_files)