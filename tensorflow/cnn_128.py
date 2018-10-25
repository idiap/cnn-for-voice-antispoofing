# Copyright (c) 2018 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
# Written by Ivan Himawan <i.himawan@qut.edu.au>,
#
# This file is part of Handbook of Biometric Anti-Spoofing 2.

# Acknowledgements: https://www.tensorflow.org, https://github.com/swaroopgj/chirps

import tensorflow as tf
import numpy as np
import cPickle as pickle
import sys
import math
import os
import h5py

from scipy.io import loadmat
from sklearn import preprocessing

batch_size = 32
beta = 0.0001
directory = 'normallogpower_models'
nFeatures = 128 # a spectrogram 128x400
nTime = 400

flags=False
training=True
testing=True
devding=True

init=float(sys.argv[1])
seed = int(init*10000)
print seed

tf.reset_default_graph()
tf.set_random_seed(seed)

def prep_data(seed=1024):

    data_fn = 'train_info.lst'
    print "Using %s" % data_fn
    with open(data_fn, 'rb') as f:
        data = pickle.load(f)
    classes = [data['labels'][k] for k in data['names']]
    classes = np.asarray([[1, 0] if l == 0 else [0, 1] for l in classes]) 

    samples = []
    f = h5py.File('normallogpower_train.mat')
    num_obj = len(f['total'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['total'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    for r in range(len(samples)):
       samples[r] = (samples[r].reshape(nFeatures,-1) - np.mean(samples[r].reshape(nFeatures,-1),axis=1,keepdims=True)).reshape(1,-1)
       samples[r] = np.divide(samples[r].reshape(nFeatures,-1),np.std(samples[r].reshape(nFeatures,-1),axis=1,keepdims=True))

    samples_pt1 = samples[0:1508]
    samples_pt2 = samples[1508:]
    classes_pt1 = classes[0:1508]
    classes_pt2 = classes[1508:]

    np.random.seed(seed)
    xrd = np.random.permutation(len(samples_pt1))
    samples_pt1 = [samples_pt1[i] for i in xrd]
    xrd = np.random.permutation(len(samples_pt2))
    samples_pt2 = [samples_pt2[i] for i in xrd]

    trn1 = int(round(len(samples_pt1)*0.9))
    trn2 = int(round(len(samples_pt2)*0.9))
    datatr = np.concatenate([samples_pt1[0:trn1],samples_pt2[0:trn2]])
    labstr = np.concatenate([classes_pt1[0:trn1],classes_pt2[0:trn2]])
    datate = np.concatenate([samples_pt1[trn1:],samples_pt2[trn2:]])
    labste = np.concatenate([classes_pt1[trn1:],classes_pt2[trn2:]])

    # permutate
    ord = np.random.permutation(len(datatr))
    datatr = [datatr[i] for i in ord]
    labstr = labstr[ord]
    datatr = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in datatr])

    ord = np.random.permutation(len(datate))
    datate = [datate[i] for i in ord]
    labste = labste[ord]
    datate = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in datate])

    del data

    return datatr, labstr, datate, labste

def prep_dev_data():

    data_fn = 'dev_info.lst'
    print "Using %s" % data_fn
    with open(data_fn, 'rb') as f:
        data = pickle.load(f)

    samples = []
    f = h5py.File('normallogpower_dev.mat')
    num_obj = len(f['total'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['total'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    for r in range(len(samples)):
       samples[r] = (samples[r].reshape(nFeatures,-1) - np.mean(samples[r].reshape(nFeatures,-1),axis=1,keepdims=True)).reshape(1,-1)
       samples[r] = np.divide(samples[r].reshape(nFeatures,-1),np.std(samples[r].reshape(nFeatures,-1),axis=1,keepdims=True))

    samples = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])

    fnames = data['names']

    del data
    return samples, fnames

def prep_test_data():

    data_fn = 'eval_info.lst'
    print "Using %s" % data_fn
    with open(data_fn, 'rb') as f:
        data = pickle.load(f)

    samples = []

    f = h5py.File('normallogpower_eval.mat')
    num_obj = len(f['total'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['total'][p][0]
       samples.append(np.array(f[mat]).T) 

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    for r in range(len(samples)):
       samples[r] = (samples[r].reshape(nFeatures,-1) - np.mean(samples[r].reshape(nFeatures,-1),axis=1,keepdims=True)).reshape(1,-1)
       samples[r] = np.divide(samples[r].reshape(nFeatures,-1),np.std(samples[r].reshape(nFeatures,-1),axis=1,keepdims=True))

    samples = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])

    fnames = data['names']

    del data
    return samples, fnames

# Conv Nets
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=init, seed=seed)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv_maxpool(x, w_conv, b_conv):
    return tf.nn.max_pool(tf.nn.relu(conv2d(x, w_conv) + b_conv), ksize=[1, 2, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

def model(x, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2, keep_prob_conv, keep_prob_fc):
    # Conv-Pool layers
    h_conv1 = tf.nn.dropout(conv_maxpool(x, w_conv1, b_conv1), keep_prob_conv, seed=seed)
    h_conv2 = tf.nn.dropout(conv_maxpool(h_conv1, w_conv2, b_conv2), keep_prob_conv, seed=seed)
    h_conv3 = tf.nn.dropout(conv_maxpool(h_conv2, w_conv3, b_conv3), keep_prob_conv, seed=seed)
    # FC layers
    #print(h_conv3.get_shape())
    h_conv3_flat = tf.reshape(h_conv3, [-1, w_fc1.get_shape().as_list()[0]])
    #print(h_conv3_flat.get_shape())
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)
    #print(h_fc1.get_shape())
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc, seed=seed)
    # output
    return tf.matmul(h_fc1_drop, w_fc2) + b_fc2

if __name__ == "__main__":

    X = tf.placeholder("float", [None, nFeatures, nTime, 1])
    Y = tf.placeholder("float", [None, 2])
    
    w_conv1 = weight_variable(shape=[3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    w_conv2 = weight_variable(shape=[3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    w_conv3 = weight_variable(shape=[3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    
    w_fc1 = weight_variable(shape=[16 * 50 * 128, 256])
    b_fc1 = bias_variable([256])
    w_fc2 = weight_variable(shape=[256, 2])
    b_fc2 = bias_variable([2])

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    py_x = model(X, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2, p_keep_conv, p_keep_hidden)
    cost_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    cost_l2 = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(w_conv3) + \
              tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2)

    cost = cost_softmax + beta*cost_l2

    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)
    auc = tf.contrib.metrics.streaming_auc(predict_op, tf.argmax(Y, 1))

    if training:
        if not os.path.exists(directory):
              os.makedirs(directory)
        lfile = open(directory+'/log'+'_'+str(init)+'.txt',"w")
        cv_accs = []
        loss_func = []
        saver = tf.train.Saver(max_to_keep=100)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # training data
        trX, trY, teX, teY = prep_data(seed=seed)
        print trX.shape
        print trY.shape
        print teX.shape
        print teY.shape
        
        trX = trX.reshape(-1, nFeatures, nTime, 1)
        teX = teX.reshape(-1, nFeatures, nTime, 1)
        saved_model_counter = 0
        for i in range(30):
            sess.run(tf.local_variables_initializer())
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX) + 1, batch_size))
            loss_epoch = []
            for start, end in training_batch:
                _, loss_iter = sess.run([train_op, cost_softmax],
                                        feed_dict={X: trX[start:end], Y: trY[start:end],
                                                   p_keep_conv: 0.5, p_keep_hidden: 0.5})
                loss_epoch.append(loss_iter)
            loss_func.append(np.mean(loss_epoch))
            pred = sess.run(predict_op, feed_dict={X: teX, p_keep_conv: 1.0, p_keep_hidden: 1.0})
            test_accuracy = np.mean(np.argmax(teY, axis=1) == pred)
            test_auc = sess.run(auc, feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})

            print(i, loss_func[-1], test_accuracy, test_auc[1])
            lfile.write("%s , %s , %s , %s\n" % (str(i), str(loss_func[-1]), str(test_accuracy), str(test_auc[1])))
            cv_accs.append(test_accuracy)
        if not flags:
           saver.save(sess, directory+'/model1'+'_'+str(init), global_step=saved_model_counter)
           print("saved model %d: %f" % (saved_model_counter, test_accuracy))
           saved_model_counter += 1
        lfile.close()
    model_fname = directory+'/model1'+'_'+str(init)+'-0'
    test_fname = directory+'/eval'+'_'+str(init)+'.csv'
    dev_fname = directory+'/dev'+'_'+str(init)+'.csv'
    if testing:
        sess = tf.Session()
        new_saver = tf.train.Saver()
        new_saver.restore(sess, model_fname)
        testX, fnames = prep_test_data()

        testX = testX.reshape(-1, nFeatures, nTime, 1)
        logits = np.asarray([sess.run(py_x,
                                      feed_dict={X: testX[i, ][None, ], p_keep_conv: 1.0,
                                                 p_keep_hidden: 1.0})
                             for i in range(len(testX))]).squeeze()
        probs = tf.nn.softmax(logits)
        test_probs = sess.run(probs)
        test_probs = test_probs[:, 1]
        # final probs
        final_probs = []
        final_fnames = []
        for i, f in enumerate(fnames):
                final_fnames.append(f)
                final_probs.append(test_probs[i])
        final_probs = np.array(final_probs)
        with open(test_fname,'w') as writer:
            for i in range(len(final_fnames)):
                writer.write("%s,%f\n"%(final_fnames[i].split('.wav')[0], final_probs[i]))

        writer.close()

    if devding:
        sess = tf.Session()
        new_saver = tf.train.Saver()
        new_saver.restore(sess, model_fname)
        devX, fnames_dev = prep_dev_data()

        devX = devX.reshape(-1, nFeatures, nTime, 1)
        logits_dev = np.asarray([sess.run(py_x,
                                      feed_dict={X: devX[i, ][None, ], p_keep_conv: 1.0,
                                                 p_keep_hidden: 1.0})
                             for i in range(len(devX))]).squeeze()
        probs_dev = tf.nn.softmax(logits_dev)
        test_probs_dev = sess.run(probs_dev)
        test_probs_dev = test_probs_dev[:, 1]
        # final probs
        final_probs_dev = []
        final_fnames_dev = []
        for i, f in enumerate(fnames_dev):
                final_fnames_dev.append(f)
                final_probs_dev.append(test_probs_dev[i])
        final_probs_dev = np.array(final_probs_dev)
        with open(dev_fname,'w') as writer:
            for i in range(len(final_fnames_dev)):
                writer.write("%s,%f\n"%(final_fnames_dev[i].split('.wav')[0], final_probs_dev[i]))
        writer.close()

