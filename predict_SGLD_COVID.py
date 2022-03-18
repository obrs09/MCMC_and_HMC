from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import scipy.ndimage as ndim
import matplotlib.colors as mcolors
import pandas as pd
import time
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import argparse
import matplotlib
from src.Stochastic_Gradient_Langevin_Dynamics.model import *
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import Optimizer

import collections
import h5py, sys
import gzip
import os
import math
import numpy as np
import pandas as pd

try:
    import cPickle as pickle
except:
    import pickle


import time
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import matplotlib

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='predict Bayesian Neural Net on MNIST with Stochastic Gradient Langevin Dynamics')
    parser.add_argument('--use_preconditioning', type=int, nargs='?', action='store', default=0,
                        help='Use RMSprop preconditioning. Default: 0.')

    parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                        help='Standard deviation of prior. Default: 0.1.')
    parser.add_argument('--epochs', type=int, nargs='?', action='store', default=200,
                        help='How many epochs to train. Default: 200.')
    parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                        help='learning rate. I recommend 1e-3 if preconditioning, else 1e-4. Default: 1e-3.')
    parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='SGLD_models',
                        help='Where to save learnt weights and train vectors. Default: \'SGLD_models\'.')
    parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='SGLD_results',
                        help='Where to save learnt training plots. Default: \'SGLD_results\'.')
    parser.add_argument('--batch_size', type=int, nargs='?', action='store', default=20,
                        help='How many batch_size to train. Default: 20.')
    parser.add_argument('--image_trans_size', type=int, nargs='?', action='store', default=64,
                        help='image_trans_size to train. Default: 64.')
    parser.add_argument('--save_data', type=bool, nargs='?', action='store', default=True,
                        help='save predicted data or not. Default: True.')

    args = parser.parse_args()
    save_data = args.save_data
    image_trans_size = args.image_trans_size
    prior_sig = args.prior_sig
    transform_covid19 = transforms.Compose([
        transforms.Resize(image_trans_size),
        transforms.CenterCrop(image_trans_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1)
    ])

    classes = 2

    trainset = torchvision.datasets.ImageFolder(root="./notebooks/data/COVID/train", transform=transform_covid19)
    valset = torchvision.datasets.ImageFolder(root="./notebooks/data/COVID/test", transform=transform_covid19)

    train_data_len = len(trainset.targets)
    test_data_len = len(valset.targets)
    use_cuda = torch.cuda.is_available()
    NTrainPoints = train_data_len
    print('mm',args.use_preconditioning)
    # Where to save models weights
    models_dir = args.models_dir
    # Where to save plots and error, accuracy vectors
    results_dir = args.results_dir
    if args.use_preconditioning == 1:
        print('ccccc')
        models_dir = 'p' + models_dir
        results_dir = 'p' + results_dir


    mkdir(models_dir)
    mkdir(results_dir)
    # ------------------------------------------------------------------------------------------------------
    # train config
    print('models_dir', models_dir)
    batch_size = args.batch_size
    nb_epochs = args.epochs
    log_interval = 1


    if use_cuda:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                  num_workers=0)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=0)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                                  num_workers=0)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=0)

    ## ---------------------------------------------------------------------------------------------------------------------
    # net dims
    cprint('c', '\nNetwork:')

    lr = args.lr
    prior_sig = args.prior_sig
    ########################################################################################
    if args.use_preconditioning == 1:
        net = Net_langevin(lr=lr, channels_in=1, side_in=image_trans_size, cuda=use_cuda, classes=classes, N_train=NTrainPoints,
                           prior_sig=prior_sig, nhid=1200, use_p=True)
        pstr = 'p'
        print('predict pSGLD in')
    else:
        net = Net_langevin(lr=lr, channels_in=1, side_in=image_trans_size, cuda=use_cuda, classes=classes, N_train=NTrainPoints,
                           prior_sig=prior_sig, nhid=1200, use_p=False)
        print("predict SGLD in")

    if args.use_preconditioning == 1:
        pstr = 'p'
        with open(models_dir + '/state_dicts.pkl', 'rb') as input:
            net.weight_set_samples = pickle.load(input)
    else:
        with open(models_dir + '/state_dicts.pkl', 'rb') as input:
            net.weight_set_samples = pickle.load(input)

    if use_cuda:
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=0)

    else:
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=0)
    test_cost = 0  # Note that these are per sample
    test_err = 0
    nb_samples = 0
    test_predictions = np.zeros((test_data_len, classes))

    Nsamples = 90

    net.set_mode_train(False)

    for j, (x, y) in enumerate(valloader):
        cost, err, probs = net.sample_eval(x, y, Nsamples, logits=False) # , logits=True
        #print('nettttttttttttttttttttttttttttttttttt', probs)
        test_cost += cost
        test_err += err.cpu().numpy()
        test_predictions[nb_samples:nb_samples+len(x), :] = probs.numpy()
        nb_samples += len(x)

    # test_cost /= nb_samples
    test_err /= nb_samples
    cprint('b', '    Loglike = %5.6f, err = %1.6f\n' % (-test_cost, test_err))


    x_dev = []
    y_dev = []
    for x, y in valloader:
        x_dev.append(x.cpu().numpy())
        y_dev.append(y.cpu().numpy())

    x_dev = np.concatenate(x_dev)
    y_dev = np.concatenate(y_dev)
    #print(x_dev.shape)
    #print(y_dev.shape)

    im_ind = np.random.randint(0, y_dev.shape[0])
    #im_ind = 90
    print("image number:", im_ind)


    x, y = x_dev[im_ind], y_dev[im_ind]
    x_rot = np.expand_dims(ndim.interpolation.rotate(x[0, :, :], 0, reshape=False, cval=-0.42421296), 0)

    print("real number:", y)

    #plt.imshow(ndim.interpolation.rotate(x_dev[im_ind,0,:,:], 0, reshape=False))


    ims=[]


    ims.append(x_rot[:,:,:])


    ims = np.concatenate(ims)

    net.set_mode_train(False)

    y = np.ones(ims.shape[0])*y
    ims = np.expand_dims(ims, axis=1)

    cost, err, probs = net.sample_eval(torch.from_numpy(ims), torch.from_numpy(y), Nsamples=Nsamples, logits=False) # , logits=True

    predictions = probs.numpy()

    #print("predictions", predictions)

    print("error", err.cpu().numpy())


    # predictions.max(axis=1)[0]
    # selections = (predictions[:,i] == predictions.max(axis=1))
    print("predict", predictions.argmax())

    print(im_ind)

    #print(valset[im_ind][1])

    print(valset.class_to_idx)

    y_true = []
    y_pred = []
    prob = []
    for i in range(0,test_data_len):
        x, y = x_dev[i], y_dev[i]
        x_rot = np.expand_dims(ndim.interpolation.rotate(x[0, :, :], 0, reshape=False, cval=-0.42421296), 0)
        #print("real number:",y)
        y_true.append(y)
        #plt.imshow( ndim.interpolation.rotate(x_dev[im_ind,0,:,:], 0, reshape=False))
        #plt.show()
        ims=[]
        ims.append(x_rot[:,:,:])
        ims = np.concatenate(ims)
        net.set_mode_train(False)
        y = np.ones(ims.shape[0])*y
        ims = np.expand_dims(ims, axis=1)
        cost, err, probs = net.sample_eval(torch.from_numpy(ims), torch.from_numpy(y), Nsamples=Nsamples, logits=False) # , logits=True
        predictions = probs.numpy()
        prob.append(predictions)
    #     print("predictions", predictions)
    #     print("error", err.cpu().numpy())
        y_pred.append(predictions.argmax())
        torch.cuda.empty_cache()

    #print(y_pred)

    prob = np.array(prob)
    prob = prob.reshape(test_data_len, classes)

    if save_data == True:
        save_path = 'SGLD_predict_data'
        if args.use_preconditioning == 1:
            save_path = pstr + save_path
            print('in pSGLD path', save_path)
        mkdir(save_path)
        file_name = "SGLD_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs, lr, batch_size, image_trans_size)
        completeName = os.path.join(save_path, file_name)
        print('c', completeName)
        if os.path.exists(completeName):
            os.remove(completeName)
        # df = pd.DataFrame(prob)
        # df.to_csv(completeName)
        np.savetxt(completeName, prob, delimiter=",")
        # file1 = open(completeName, "w")
        # for i in range(0, 41):
        #
        #     file1.write(str(prob[i]))
        #     file1.write("\n")
        # file1.close()

    #plt.show()



