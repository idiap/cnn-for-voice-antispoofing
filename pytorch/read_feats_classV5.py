# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by - Srikanth Madikeri (2017)
# Based on the code from Ivan Himawan (QUT, Brisbane)

from __future__ import print_function, division
import os
import sys
import math
import torch
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pickle
import h5py
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class ASVSpoofTrainData(Dataset):

    def __init__(self, transform=None):
        data_fn = 'train_info.lst'
        with open(data_fn, 'rb') as f:
            data = pickle.load(f)
        self.classes = [data['labels'][k] for k in data['names']]
        self.num_obj = len(self.classes)
        self.classes = torch.from_numpy(np.array(self.classes)).long()

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        mat = pickle.load(open('data/768/train-files/'+str(idx)+'.npy', 'rb'))
        return (torch.from_numpy(mat).float(), self.classes[idx])


class ASVSpoofDevData(Dataset):

    def __init__(self, transform=None):
        classes = dict([ln.strip().split() for ln in open('dev-keys')])
        files = [ln.strip().split('/')[-1].split('.')[0] for ln in open('filelist_dev')]
        self.classes = torch.from_numpy(np.array([0 if classes[k]=="spoof" else 1 for k in files])).long()
        self.num_obj = len(self.classes)

    def __len__(self):
        return self.num_obj

    def __getitem__(self, idx):
        if idx > self.num_obj:
            raise IndexError()

        mat = np.load(open('data/768/dev-files/'+str(idx)+'.npy', 'rb'))
        return (torch.from_numpy(mat).float(), self.classes[idx])

class ASVSpoofTestData(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_fn = 'eval_info.lst'
        with open(self.data_fn, 'rb') as f:
            data = pickle.load(f)

        self.fnames = data['names'] # assume it is sorted

        self.classes = {}
        with open("ASVspoof2017_eval_v2_key.trl.txt") as f:
            for line in f:
                (key, val) = line.split()
                self.classes[key] = 0 if val == 'spoof' else 1
        self.fnames_sub = [x for x in self.fnames if x in self.classes]
        self.labels = torch.from_numpy(np.array([self.classes[x] for x in self.fnames if x in self.classes])).long()
        self.indices = []
        idx = -1
        for x in self.fnames:
            idx += 1
            if x in self.classes:
                self.indices.append(idx)

    def __len__(self):
        return len(self.fnames_sub)

    def __getitem__(self, idx):        
        if idx > len(self.fnames_sub):
            raise IndexError()
        actual_idx = self.indices[idx]
        file_idx = int(math.floor(actual_idx/1000))+1
        mat = pickle.load(open('data/768/eval-files/'+str(file_idx)+'/'+str(actual_idx)+'.npy', 'rb'))
        return (torch.from_numpy(mat).float(), self.labels[idx])

if __name__ == '__main__':
    d = ASVSpoofTrainData()
    dl = DataLoader(d, batch_size=100)
    for x, l in dl:
        print("New batch")

