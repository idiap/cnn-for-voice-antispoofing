
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by - Srikanth Madikeri (2017)
# Modified by - Milos Cernak (2017)

import sys
import os
import pickle
import math
import h5py
import numpy as np
from sklearn import preprocessing

split_train = True
split_test = True
split_dev = True
featfolder = './logpow'

if len(sys.argv) >= 2:
    featfolder = sys.argv[1]

if split_train:
    print("Processing train data")
    opfolder = 'data/768/train-files'
    if not os.path.exists(opfolder):
        try:
            os.makedirs(opfolder, exist_ok=True)
        except:
            sys.stderr.write('ERROR: Unable to create directory %s' % opfolder)
            sys.stderr.write('EXITING\n')
            quit(1)
    data_fn = 'train_info.lst'
    with open(data_fn, 'rb') as f:
        data = pickle.load(f)
    classes = [data['labels'][k] for k in data['names']]
    num_obj = len(classes)
    for idx in range(num_obj):
        print("Processing idx: %d" % idx)
        nFeatures = 768
        actual_idx = idx
        file_idx = int(math.floor(actual_idx/1000))+1
        itemidx = actual_idx - (file_idx-1)*1000
        file_idx = int(math.floor(actual_idx/1000))+1
        fileptr = h5py.File(featfolder+'/normallogpower_train_' + str(file_idx) + '.mat')
        mat = np.array(fileptr[fileptr['total'][itemidx][0]]).T
        fileptr.close()
        shape = mat.shape[1] * mat.shape[0] // nFeatures
        filter_x = (shape < 400/2)

        if filter_x:
            rr = int(math.ceil(400.0/shape))
            temp = mat.reshape(nFeatures,-1)
            tttt = np.tile(temp,rr)
            mat = tttt.reshape(nFeatures,-1)[:,0:400]

        shape = mat.shape[1] * mat.shape[0] // nFeatures
        filter_s = (shape < 400)
        if filter_s:
            temp = mat.reshape(nFeatures,-1)
            temo = mat.reshape(nFeatures,-1)[:,0:400-shape]
            mat = np.hstack([temp,temo])

        filter_l = (shape > 400)
        if filter_l:
            mat = mat.reshape(nFeatures,-1)[:,0:400]

        mat = (mat.reshape(nFeatures,-1) - np.mean(mat.reshape(nFeatures,-1),axis=1,keepdims=True))
        mat = np.divide(mat.reshape(nFeatures,-1),np.std(mat.reshape(nFeatures,-1),axis=1,keepdims=True))

        mat = preprocessing.scale(mat.reshape(nFeatures,-1))
        f = open(opfolder+'/'+str(idx)+'.npy', 'wb')
        pickle.dump(mat, f)
        f.close()

if split_dev:
    print("Processing dev data")
    opfolder = 'data/768/dev-files'
    if not os.path.exists(opfolder):
        try:
            os.makedirs(opfolder, exist_ok=True)
        except:
            sys.stderr.write('ERROR: Unable to create directory %s' % opfolder)
            sys.stderr.write('EXITING\n')
            quit(1)
    classes = dict([ln.strip().split() for ln in open('dev-keys')])
    classes = dict([(k, 0 if classes[k]=="spoof" else 1) for k in classes])
    num_obj = len(classes)
    for idx in range(num_obj):
        print("Processing idx: %d" % idx)
        nFeatures = 768
        actual_idx = idx
        file_idx = int(math.floor(actual_idx/1000))+1
        itemidx = actual_idx - (file_idx-1)*1000
        file_idx = int(math.floor(actual_idx/1000))+1
        fileptr = h5py.File(featfolder+'/normallogpower_dev_' + str(file_idx) + '.mat')
        mat = np.array(fileptr[fileptr['total'][itemidx][0]]).T
        fileptr.close()
        shape = mat.shape[1] * mat.shape[0] // nFeatures
        filter_x = (shape < 400/2)

        if filter_x:
            rr = int(math.ceil(400.0/shape))
            temp = mat.reshape(nFeatures,-1)
            tttt = np.tile(temp,rr)
            mat = tttt.reshape(nFeatures,-1)[:,0:400]

        shape = mat.shape[1] * mat.shape[0] // nFeatures
        filter_s = (shape < 400)
        if filter_s:
            temp = mat.reshape(nFeatures,-1)
            temo = mat.reshape(nFeatures,-1)[:,0:400-shape]
            mat = np.hstack([temp,temo])

        filter_l = (shape > 400)
        if filter_l:
            mat = mat.reshape(nFeatures,-1)[:,0:400]

        mat = (mat.reshape(nFeatures,-1) - np.mean(mat.reshape(nFeatures,-1),axis=1,keepdims=True))
        mat = np.divide(mat.reshape(nFeatures,-1),np.std(mat.reshape(nFeatures,-1),axis=1,keepdims=True))

        mat = preprocessing.scale(mat.reshape(nFeatures,-1))
        f = open(opfolder+'/'+str(idx)+'.npy', 'wb')
        np.save(f, mat, allow_pickle=False)
        f.close()

if split_test:
    print("Processing test data")
    opfolder = 'data/768/eval-files'
    if not os.path.exists(opfolder):
        try:
            os.makedirs(opfolder, exist_ok=True)
        except:
            sys.stderr.write('ERROR: Unable to create directory %s' % opfolder)
            sys.stderr.write('EXITING\n')
            quit(1)
    data_fn = 'eval_info.lst'
    with open(data_fn, 'rb') as f:
        data = pickle.load(f)

    fnames = data['names'] # assume it is sorted
    num_obj = len(fnames)
    idx = -1
    for file_idx in range(1, 16):
        nFeatures = 768
        currfilename = featfolder+'/normallogpower_eval_' + str(file_idx) + '.mat'
        fileptr = h5py.File(currfilename)
        for i in range(len(fileptr['total'])):
            idx += 1
            print("Processing idx: %d" % idx)
            mat = np.array(fileptr[fileptr['total'][i][0]]).T
            shape = mat.shape[1] * mat.shape[0] // nFeatures
            filter_x = (shape < 400/2)
            if filter_x:
                rr = int(math.ceil(400.0/shape))
                temp = mat.reshape(nFeatures,-1)
                tttt = np.tile(temp,rr)
                mat = tttt.reshape(nFeatures,-1)[:,0:400]

            shape = mat.shape[1] * mat.shape[0] // nFeatures
            filter_s = (shape < 400)
            if filter_s:
                temp = mat.reshape(nFeatures,-1)
                temo = mat.reshape(nFeatures,-1)[:,0:400-shape]
                mat = np.hstack([temp,temo])

            filter_l = (shape > 400)
            if filter_l:
                mat = mat.reshape(nFeatures,-1)[:,0:400]

            mat = (mat.reshape(nFeatures,-1) - np.mean(mat.reshape(nFeatures,-1),axis=1,keepdims=True))
            mat = np.divide(mat.reshape(nFeatures,-1),np.std(mat.reshape(nFeatures,-1),axis=1,keepdims=True))

            mat = preprocessing.scale(mat.reshape(nFeatures,-1))

            try:
                os.makedirs(opfolder+'/'+str(file_idx), exist_ok=True)
            except:
                pass
                
            f = open(opfolder+'/'+str(file_idx)+'/'+str(idx)+'.npy', 'wb')
            pickle.dump(mat, f)
            f.close()

        fileptr.close()
        

