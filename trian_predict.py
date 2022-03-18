from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bootstrap_Ensemble.model import *
import copy

import os
import numpy as np
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
image_trans_size = 64
batch_size = 5
nb_epochs = 50
lr = 0.00001
prior_sig = 0.1
models_dir = 'models_SGLD_COVID150'
results_dir = 'results_SGLD_COVID150'
pSGLD = 'False'
save_data = 'True'

# def train():
#
#     print('python train_SGLD_COVID.py --use_preconditioning %s --prior_sig %f --epochs %d --lr %f --models_dir %s --results_dir %s --batch_size %d --image_trans_size %d'
#           %('False', prior_sig, nb_epochs, lr, models_dir, results_dir, batch_size, image_trans_size))
#     torch.cuda.empty_cache()
#
#     os.system('python train_SGLD_COVID.py --use_preconditioning %s --prior_sig %f --epochs %d --lr %f --models_dir %s --results_dir %s --batch_size %d --image_trans_size %d'
#           %('False', prior_sig, nb_epochs, lr, models_dir, results_dir, batch_size, image_trans_size))
#
#     torch.cuda.empty_cache()
#     return 0
#
# def predict():
#
#     print('python predict_SGLD_COVID.py --use_preconditioning %s --prior_sig %f --epochs %d --lr %f --models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s'
#           %(pSGLD, prior_sig, nb_epochs, lr, models_dir, results_dir, batch_size, image_trans_size, save_data))
#
#     torch.cuda.empty_cache()
#
#     os.system('python predict_SGLD_COVID.py --use_preconditioning %s --prior_sig %f --epochs %d --lr %f --models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s'
#           %(pSGLD, prior_sig, nb_epochs, lr, models_dir, results_dir, batch_size, image_trans_size, save_data))
#
#     torch.cuda.empty_cache()
#
#
#
#     return 0

class TP(object):
    def __init__(self, image_trans_size, batch_size, nb_epochs, lr, models_dir, results_dir, prior_sig, pSGLD, save_data, n_samples, sample_freq, burn_in):
        self.image_trans_size = image_trans_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.prior_sig = prior_sig
        self.pSGLD = bool(pSGLD)
        self.save_data = save_data
        self.n_samples = n_samples
        self.sample_freq = sample_freq
        self.burn_in = burn_in

    def train_SGLD(self):
        print('class out', self.pSGLD)
        if self.pSGLD:
            train_sgld_str = 'python train_SGLD_COVID.py --use_preconditioning 1 --prior_sig %f --epochs %d --lr %f ' \
                         '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d'\
                         % (self.prior_sig, self.nb_epochs, self.lr,
                            self.models_dir, self.results_dir, self.batch_size, self.image_trans_size)
            torch.cuda.empty_cache()
        else:
            train_sgld_str = 'python train_SGLD_COVID.py --use_preconditioning 0 --prior_sig %f --epochs %d --lr %f ' \
                             '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d' \
                             % (self.prior_sig, self.nb_epochs, self.lr,
                                self.models_dir, self.results_dir, self.batch_size, self.image_trans_size)
            torch.cuda.empty_cache()

        os.system(train_sgld_str)
        torch.cuda.empty_cache()

    def save_prediction_SGLD(self):
        print('class out', self.pSGLD)
        if self.pSGLD:
            print('a')
            spsstr = 'python predict_SGLD_COVID.py --use_preconditioning 1 --prior_sig %f --epochs %d --lr %f ' \
                 '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s'\
                 % (self.prior_sig, self.nb_epochs, self.lr,
                    self.models_dir, self.results_dir, self.batch_size, self.image_trans_size, self.save_data)
            torch.cuda.empty_cache()
        else:
            print('b')
            spsstr = 'python predict_SGLD_COVID.py --use_preconditioning 0 --prior_sig %f --epochs %d --lr %f ' \
                 '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s' \
                 % (self.prior_sig, self.nb_epochs, self.lr,
                    self.models_dir, self.results_dir, self.batch_size, self.image_trans_size, self.save_data)
            torch.cuda.empty_cache()

        os.system(spsstr)
        torch.cuda.empty_cache()

    def not_save_prediction_SGLD(self):
        nspsstr = 'python predict_SGLD_COVID.py --use_preconditioning %s --prior_sig %f --epochs %d --lr %f ' \
                  '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d -save_data %s'\
                  % (self.pSGLD, self.prior_sig, self.nb_epochs, self.lr,
                     self.models_dir, self.results_dir, self.batch_size, self.image_trans_size, "False")
        torch.cuda.empty_cache()

        os.system(nspsstr)
        torch.cuda.empty_cache()


    def train_SGHMC(self):
        tsghmc = 'python train_SGHMC_COVID.py --epochs %d --sample_freq %d --burn_in %d --lr %f ' \
                 '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d'\
                         % (self.nb_epochs, self.sample_freq, self.burn_in, self.lr, self.models_dir, self.results_dir, self.batch_size, self.image_trans_size)
        torch.cuda.empty_cache()

        os.system(tsghmc)
        torch.cuda.empty_cache()

    def save_prediction_SGHMC(self):
        spsghmc = 'python predict_SGHMC_COVID.py --epochs %d --sample_freq %d --burn_in %d --lr %f ' \
                  '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s'\
                 % (self.nb_epochs, self.sample_freq, self.burn_in, self.lr,
                    self.models_dir, self.results_dir, self.batch_size, self.image_trans_size, self.save_data)
        torch.cuda.empty_cache()

        os.system(spsghmc)
        torch.cuda.empty_cache()

    def not_save_prediction_SGHMC(self):
        nspsghmc = 'python predict_SGHMC_COVID.py --epochs %d --sample_freq %d --burn_in %d --lr %f ' \
                   '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s'\
                  % (self.nb_epochs, self.sample_freq, self.burn_in, self.lr,
                     self.models_dir, self.results_dir, self.batch_size, self.image_trans_size, "False")
        torch.cuda.empty_cache()

        os.system(nspsghmc)
        torch.cuda.empty_cache()


    def train_BBB(self):
        tbbbstr = 'python train_BayesByBackprop_COVID.py --model %s --prior_sig %f --epochs %d --lr %f --n_samples %d ' \
                  '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d'\
                  % ('Local_Reparam', self.prior_sig, self.nb_epochs, self.lr, self.n_samples,
                     self.models_dir, self.results_dir, self.batch_size, self.image_trans_size)
        torch.cuda.empty_cache()

        os.system(tbbbstr)
        torch.cuda.empty_cache()

    def save_prediction_BBB(self):
        spsbbbstr = 'python predict_BBB_COVID.py --model %s --prior_sig %f --epochs %d --lr %f --n_samples %d ' \
                    '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s'\
                 % ('Local_Reparam', self.prior_sig, self.nb_epochs, self.lr, self.n_samples,
                    self.models_dir, self.results_dir, self.batch_size, self.image_trans_size, self.save_data)
        torch.cuda.empty_cache()

        os.system(spsbbbstr)
        torch.cuda.empty_cache()

    def not_save_prediction_BBB(self):
        nspsbbbstr = 'python train_predict_BBB_COVID.py --model %s --prior_sig %f --epochs %d --lr %f --n_samples %d ' \
                     '--models_dir %s --results_dir %s --batch_size %d --image_trans_size %d --save_data %s'\
                     %('Local_Reparam', self.prior_sig, self.nb_epochs, self.lr, self.n_samples,
                       self.models_dir, self.results_dir, self.batch_size, self.image_trans_size, "False")
        torch.cuda.empty_cache()

        os.system(nspsbbbstr)
        torch.cuda.empty_cache()


    def train_all(self):
        self.train_SGLD()
        torch.cuda.empty_cache()
        self.train_BBB()
        torch.cuda.empty_cache()
        self.save_prediction_SGLD()
        torch.cuda.empty_cache()
        self.save_prediction_BBB()
        torch.cuda.empty_cache()

    def predict_with_save_data(self):
        self.save_prediction_SGLD()
        torch.cuda.empty_cache()
        self.save_prediction_BBB()
        torch.cuda.empty_cache()

    def predict_without_save_data(self):
        self.not_save_prediction_SGLD()
        torch.cuda.empty_cache()
        self.not_save_prediction_BBB()
        torch.cuda.empty_cache()



# image_trans_size_SGHMC = 64
# batch_size_SGHMC = 5
# nb_epochs_SGHMC = 50
# lr_SGHMC = 0.001
# prior_sig_SGHMC = 0.1
#
# pSGLD_SGHMC = 'False'
# save_data_SGHMC = 'True'
# n_samples = 90
# sample_freq = 2
# burn_in = 20
#
# sample_freq_SGHMC = 5
# burn_in_SGHMC = 20
# models_dir_SGHMC = 'models_SGHMC_COVID150'
# results_dir_SGHMC = 'results_SGHMC_COVID150'
#
# model_SGHMC = TP(image_trans_size_SGHMC, batch_size_SGHMC, nb_epochs_SGHMC, lr_SGHMC, models_dir_SGHMC, results_dir_SGHMC,
#                 prior_sig_SGHMC, pSGLD_SGHMC, save_data_SGHMC, n_samples, sample_freq_SGHMC, burn_in_SGHMC)
# model_SGHMC.train_SGHMC()
# #model_SGHMC.save_prediction_SGHMC()


# image_trans_size_BBB = 64
# batch_size_BBB = 5
# nb_epochs_BBB = 50
# lr_BBB = 0.00001
# prior_sig_BBB = 0.1
# models_dir_BBB = 'models_BBB_COVID150'
# results_dir_BBB = 'results_BBB_COVID150'
# pSGLD_BBB = 'False'
# save_data_BBB = 'True'
# n_samples = 90
# sample_freq = 2
# burn_in = 20
#
# model_BBB = TP(image_trans_size_BBB, batch_size_BBB, nb_epochs_BBB, lr_BBB, models_dir_BBB, results_dir_BBB,
#                prior_sig_BBB, pSGLD_BBB, save_data_BBB, n_samples, sample_freq, burn_in)
# model_BBB.train_BBB()
# #model_BBB.save_prediction_BBB()
