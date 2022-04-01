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

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import Optimizer
import pandas
import collections
import h5py, sys
import gzip
import os
import math

import pandas as pd

import time
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import matplotlib

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
import pandas as pd
from trian_predict import TP

from roc_data import roc

def mcc_detail(mcc_list):
    adata = mcc_list[mcc_list[:, 0].argsort()][:, 1]
    aindex = mcc_list[mcc_list[:, 0].argsort()][:, 0]
    aaa = np.sort(np.unique(adata, return_index=True)[1])
    #aindex[np.where(adata == adata[aaa[0]])]
    a = []
    b = []
    for i in range(len(aaa)):
        a.append([min(aindex[np.where(adata == adata[aaa[i]])]),
                  max(aindex[np.where(adata == adata[aaa[i]])])])
        b.append(adata[aaa[i]])
    return a, b


start = [0, 1]
end = [0, 1]
len_cd = []
sns_data = []

torch.cuda.empty_cache()

########################################################################################################################
#SGLD
image_trans_size_SGLD = 128
batch_size_SGLD = 40
nb_epochs_SGLD = 100
lr_SGLD = 0.00001
prior_sig_SGLD = 0.1
models_dir_SGLD = 'models_SGLD_COVID150'
results_dir_SGLD = 'results_SGLD_COVID150'
pSGLD_SGLD = False
save_data_SGLD = True
n_samples_SGLD = 20
sample_freq = 2
burn_in_SGLD = 200

print('pSGLD_SGLD',pSGLD_SGLD)
model_SGLD = TP(image_trans_size_SGLD, batch_size_SGLD, nb_epochs_SGLD, lr_SGLD, models_dir_SGLD, results_dir_SGLD,
                prior_sig_SGLD, pSGLD_SGLD, save_data_SGLD, n_samples_SGLD, sample_freq, burn_in_SGLD)
print("train SGLD")
#model_SGLD.train_SGLD()
print("predict SGLD")
#model_SGLD.save_prediction_SGLD()

########################################################################################################################
#pSGLD
image_trans_size_pSGLD = 128
batch_size_pSGLD = 40
nb_epochs_pSGLD = 100
lr_pSGLD = 0.00005
prior_sig_pSGLD = 0.1
models_dir_pSGLD = 'models_pSGLD_COVID150'
results_dir_pSGLD = 'results_pSGLD_COVID150'
pSGLD_pSGLD = True
save_data_pSGLD = True
n_samples_pSGLD = 20
sample_freq_pSGLD = 2
burn_in_pSGLD = 200





print('pSGLD_pSGLD', pSGLD_pSGLD)
model_pSGLD = TP(image_trans_size_pSGLD, batch_size_pSGLD, nb_epochs_pSGLD, lr_pSGLD, models_dir_pSGLD, results_dir_pSGLD,
                 prior_sig_pSGLD, pSGLD_pSGLD, save_data_pSGLD, n_samples_pSGLD, sample_freq_pSGLD, burn_in_pSGLD)
print("train pSGLD")
#model_pSGLD.train_SGLD()
print("predict pSGLD")
#model_pSGLD.save_prediction_SGLD()

########################################################################################################################
#SGHMC

image_trans_size_SGHMC = 128
batch_size_SGHMC = 40
nb_epochs_SGHMC = 100
lr_SGHMC = 0.001
prior_sig_SGHMC = 0.1
models_dir_SGHMC = 'models_SGHMC_COVID150'
results_dir_SGHMC = 'results_SGHMC_COVID150'
pSGLD_SGHMC = False
save_data_SGHMC = True
n_samples_SGHMC = 20
sample_freq_SGHMC = 2
burn_in_SGHMC = 20





model_SGHMC = TP(image_trans_size_SGHMC, batch_size_SGHMC, nb_epochs_SGHMC, lr_SGHMC, models_dir_SGHMC, results_dir_SGHMC,
                prior_sig_SGHMC, pSGLD_SGHMC, save_data_SGHMC, n_samples_SGHMC, sample_freq_SGHMC, burn_in_SGHMC)
#model_SGHMC.train_SGHMC()
#model_SGHMC.save_prediction_SGHMC()

########################################################################################################################
#BBB



image_trans_size_BBB = 128
batch_size_BBB = 40
nb_epochs_BBB = 100
lr_BBB = 0.00005
prior_sig_BBB = 0.1
models_dir_BBB = 'models_BBB_COVID150'
results_dir_BBB = 'results_BBB_COVID150'
pSGLD_BBB = 'False'
save_data_BBB = 'True'
n_samples_BBB = 5
sample_freq_BBB = 2
burn_in_BBB = 20

model_BBB = TP(image_trans_size_BBB, batch_size_BBB, nb_epochs_BBB, lr_BBB, models_dir_BBB, results_dir_BBB,
               prior_sig_BBB, pSGLD_BBB, save_data_BBB, n_samples_BBB, sample_freq_BBB, burn_in_BBB)
print("train BBB")
#model_BBB.train_BBB()
print("predict BBB")
#model_BBB.save_prediction_BBB()

########################################################SGLD############################################################
print('train, predict over')


save_path_SGLD = 'SGLD_predict_data'
file_name_SGLD = "SGLD_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_SGLD, lr_SGLD, batch_size_SGLD, image_trans_size_SGLD)
completeName_SGLD = os.path.join(save_path_SGLD, file_name_SGLD)
print(completeName_SGLD)
with open(completeName_SGLD) as file_name_S:
    prob_SGLD = np.loadtxt(file_name_S, delimiter=",")
    file_name_S.close()

save_path_SGLD = 'SGLD_predict_data'
file_name_SGLD = "SGLD_train_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_SGLD, lr_SGLD, batch_size_SGLD, image_trans_size_SGLD)
completeName_SGLD = os.path.join(save_path_SGLD, file_name_SGLD)
print(completeName_SGLD)
with open(completeName_SGLD) as file_name_S:
    prob_SGLD_train = np.loadtxt(file_name_S, delimiter=",")
    file_name_S.close()

##################################################pSGLD#################################################################

save_path_pSGLD = 'pSGLD_predict_data'
file_name_pSGLD = "pSGLD_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_pSGLD, lr_pSGLD, batch_size_pSGLD, image_trans_size_pSGLD)
completeName_pSGLD = os.path.join(save_path_pSGLD, file_name_pSGLD)
print(completeName_pSGLD)
with open(completeName_pSGLD) as file_name_p:
    prob_pSGLD = np.loadtxt(file_name_p, delimiter=",")
    file_name_p.close()

save_path_pSGLD = 'pSGLD_predict_data'
file_name_pSGLD = "pSGLD_train_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_pSGLD, lr_pSGLD, batch_size_pSGLD, image_trans_size_pSGLD)
completeName_pSGLD = os.path.join(save_path_pSGLD, file_name_pSGLD)
print(completeName_pSGLD)
with open(completeName_pSGLD) as file_name_p:
    prob_pSGLD_train = np.loadtxt(file_name_p, delimiter=",")
    file_name_p.close()

########################################################SGHMC############################################################


save_path_SGHMC = 'SGHMC_predict_data'
file_name_SGHMC = "SGHMC_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_SGHMC, lr_SGHMC, batch_size_SGHMC, image_trans_size_SGHMC)
completeName_SGHMC = os.path.join(save_path_SGHMC, file_name_SGHMC)

with open(completeName_SGHMC) as file_name:
    prob_SGHMC = np.loadtxt(file_name, delimiter=",")
    file_name.close()

save_path_SGHMC = 'SGHMC_predict_data'
file_name_SGHMC = "SGHMC_train_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs_SGHMC, lr_SGHMC, batch_size_SGHMC, image_trans_size_SGHMC)
completeName_SGHMC = os.path.join(save_path_SGHMC, file_name_SGHMC)

with open(completeName_SGHMC) as file_name:
    prob_SGHMC_train = np.loadtxt(file_name, delimiter=",")
    file_name.close()

####################################################BBB#################################################################

save_path_BBB = 'BBB_predict_data'
file_name_BBB = "BBB_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
            % (nb_epochs_BBB, lr_BBB, batch_size_BBB, image_trans_size_BBB)
completeName_BBB = os.path.join(save_path_BBB, file_name_BBB)
print(completeName_BBB)
with open(completeName_BBB) as file_name_B:
    prob_BBB = np.loadtxt(file_name_B, delimiter=",")
    file_name_B.close()

save_path_BBB = 'BBB_predict_data'
file_name_BBB = "BBB_train_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
            % (nb_epochs_BBB, lr_BBB, batch_size_BBB, image_trans_size_BBB)
completeName_BBB = os.path.join(save_path_BBB, file_name_BBB)
print(completeName_BBB)
with open(completeName_BBB) as file_name_B:
    prob_BBB_train = np.loadtxt(file_name_B, delimiter=",")
    file_name_B.close()

########################################################################################################################
#print(prob_SGLD)
#print(prob_pSGLD)
#print(prob_BBB)

ROC_SGLD = roc(prob_SGLD, prob_SGLD_train)
ROC_SGLD.threshold_list()
ROC_SGLD.threshold_list_train()

fpr_SGLD = ROC_SGLD.fpr_eval()
tpr_SGLD = ROC_SGLD.tpr_eval()

fpr_SGLD_train = ROC_SGLD.fpr_eval_train()
tpr_SGLD_train = ROC_SGLD.tpr_eval_train()

cd_SGLD = ROC_SGLD.crit_point()
cd_SGLD_test = ROC_SGLD.crit_point_test()
C2_SGLD, cd_range_SGLD = ROC_SGLD.confusion_matrix_plot_crit()

tnr_SGLD = ROC_SGLD.tnr_eval()
ppv_SGLD = ROC_SGLD.ppv_eval()
acc_SGLD = ROC_SGLD.acc_eval()
f1_SGLD = ROC_SGLD.f1_eval()
mcc_SGLD, mcc_list_SGLD = ROC_SGLD.mcc_eval()

roc_auc_SGLD = metrics.auc(fpr_SGLD, tpr_SGLD)
roc_auc_SGLD_train = metrics.auc(fpr_SGLD_train, tpr_SGLD_train)
print(roc_auc_SGLD)

print(C2_SGLD)
len_cd.append(len(C2_SGLD))
#print(C2_SGLD.ravel())
sns_data.append(C2_SGLD)
print('len_SGLD', len((C2_SGLD)))

a_SGLD, b_SGLD = mcc_detail(mcc_list_SGLD)

print('mcc_list_SGLD', a_SGLD, b_SGLD)


if len(C2_SGLD) == 1:
    fig_SGLD, axs_SGLD = plt.subplots(1, len(C2_SGLD))
    sns.heatmap(data=C2_SGLD[0], annot=True, fmt='d')
    axs_SGLD.set_xlabel('Pred_SGLD')
    axs_SGLD.set_ylabel('True_SGLD')
    axs_SGLD.set_title("SGLD with " + str(cd_range_SGLD[0][0]) + '~' + str(cd_range_SGLD[0][1]))
else:
    fig_SGLD, axs_SGLD = plt.subplots(1, len(C2_SGLD))
    for i in range(len(C2_SGLD)):
        sns.heatmap(data=C2_SGLD[i], ax=axs_SGLD[i], annot=True, fmt='d')
        axs_SGLD[i].set_xlabel('Pred_SGLD')
        axs_SGLD[i].set_ylabel('True_SGLD')
        axs_SGLD[i].set_title("SGLD" + str(i+1) + ' with ' + str(cd_range_SGLD[i][0]) + '~' + str(cd_range_SGLD[i][1]))

########################################################################################################################


ROC_pSGLD = roc(prob_pSGLD, prob_pSGLD_train)
ROC_pSGLD.threshold_list()
ROC_pSGLD.threshold_list_train()

fpr_pSGLD = ROC_pSGLD.fpr_eval()
tpr_pSGLD = ROC_pSGLD.tpr_eval()

fpr_pSGLD_train = ROC_pSGLD.fpr_eval_train()
tpr_pSGLD_train = ROC_pSGLD.tpr_eval_train()

cd_pSGLD = ROC_pSGLD.crit_point()
cd_pSGLD_test = ROC_pSGLD.crit_point_test()
C2_pSGLD, cd_range_pSGLD = ROC_pSGLD.confusion_matrix_plot_crit()

tnr_pSGLD = ROC_pSGLD.tnr_eval()
ppv_pSGLD = ROC_pSGLD.ppv_eval()
acc_pSGLD = ROC_pSGLD.acc_eval()
f1_pSGLD = ROC_pSGLD.f1_eval()
mcc_pSGLD, mcc_list_pSGLD = ROC_pSGLD.mcc_eval()

a_pSGLD, b_pSGLD = mcc_detail(mcc_list_pSGLD)

roc_auc_pSGLD = metrics.auc(fpr_pSGLD, tpr_pSGLD)
roc_auc_pSGLD_train = metrics.auc(fpr_pSGLD_train, tpr_pSGLD_train)
print(roc_auc_pSGLD)

print(C2_pSGLD)
print('len_pSGLD', len(C2_pSGLD))
#print(C2_pSGLD.ravel())
len_cd.append(len(C2_pSGLD))
sns_data.append(C2_pSGLD)

if len(C2_pSGLD) == 1:
    fig_pSGLD, axs_pSGLD = plt.subplots(1, len(C2_pSGLD))
    sns.heatmap(data=C2_pSGLD[0], annot=True, fmt='d')
    axs_pSGLD.set_xlabel('Pred_pSGLD')
    axs_pSGLD.set_ylabel('True_pSGLD')
    axs_pSGLD.set_title("pSGLD with " + str(cd_range_pSGLD[0][0]) + '~' + str(cd_range_pSGLD[0][1]))
else:

    fig_pSGLD, axs_pSGLD = plt.subplots(1, len(C2_pSGLD))
    for i in range(len(C2_pSGLD)):
        sns.heatmap(data=C2_pSGLD[i], ax=axs_pSGLD[i], annot=True, fmt='d')
        axs_pSGLD[i].set_xlabel('Pred_pSGLD')
        axs_pSGLD[i].set_ylabel('True_pSGLD')
        axs_pSGLD[i].set_title("pSGLD" + str(i+1) + ' with ' + str(cd_range_pSGLD[i][0]) + '~' + str(cd_range_pSGLD[i][1]))


########################################################################################################################


ROC_SGHMC = roc(prob_SGHMC, prob_SGHMC_train)
ROC_SGHMC.threshold_list()
ROC_SGHMC.threshold_list_train()

fpr_SGHMC = ROC_SGHMC.fpr_eval()
tpr_SGHMC = ROC_SGHMC.tpr_eval()

fpr_SGHMC_train = ROC_SGHMC.fpr_eval_train()
tpr_SGHMC_train = ROC_SGHMC.tpr_eval_train()

cd_SGHMC = ROC_SGHMC.crit_point()
cd_SGHMC_test = ROC_SGHMC.crit_point_test()
C2_SGHMC, cd_range_SGHMC = ROC_SGHMC.confusion_matrix_plot_crit()

tnr_SGHMC = ROC_SGHMC.tnr_eval()
ppv_SGHMC = ROC_SGHMC.ppv_eval()
acc_SGHMC = ROC_SGHMC.acc_eval()
f1_SGHMC = ROC_SGHMC.f1_eval()
mcc_SGHMC, mcc_list_SGHMC = ROC_SGHMC.mcc_eval()

a_SGHMC, b_SGHMC = mcc_detail(mcc_list_SGHMC)

roc_auc_SGHMC = metrics.auc(fpr_SGHMC, tpr_SGHMC)
roc_auc_SGHMC_train = metrics.auc(fpr_SGHMC_train, tpr_SGHMC_train)
print(roc_auc_SGHMC)

print(C2_SGHMC)
#print(C2_SGHMC.ravel())
len_cd.append(len(C2_SGHMC))
print('len_SGHMC', len((C2_SGHMC)))
print('----------------------------', mcc_list_SGHMC, a_SGHMC, b_SGHMC)
# plt.figure(3)
# sns.heatmap(C2_SGHMC, annot=True)
# plt.xlabel('Pred_SGHMC')
# plt.ylabel('True_SGHMC')
sns_data.append(C2_SGHMC)

print('sns_data', sns_data)

if len(C2_SGHMC) == 1:
    fig_SGHMC, axs_SGHMC = plt.subplots(1, len(C2_SGHMC))
    sns.heatmap(data=C2_SGHMC[0], annot=True, fmt='d')
    axs_SGHMC.set_xlabel('Pred_SGHMC')
    axs_SGHMC.set_ylabel('True_SGHMC')
    axs_SGHMC.set_title("SGHMC with " + str(cd_range_SGHMC[0][0]) + '~' + str(cd_range_SGHMC[0][1]))
else:

    fig_SGHMC, axs_SGHMC = plt.subplots(1, len(C2_SGHMC))
    for i in range(len(C2_SGHMC)):
        sns.heatmap(data=C2_SGHMC[i], ax=axs_SGHMC[i], annot=True, fmt='d')
        axs_SGHMC[i].set_xlabel('Pred_SGHMC')
        axs_SGHMC[i].set_ylabel('True_SGHMC')
        axs_SGHMC[i].set_title("SGHMC" + str(i+1) + ' with ' + str(cd_range_SGHMC[i][0]) + '~' + str(cd_range_SGHMC[i][1]))


########################################################################################################################


ROC_BBB = roc(prob_BBB, prob_BBB_train)
ROC_BBB.threshold_list()
ROC_BBB.threshold_list_train()

fpr_BBB = ROC_BBB.fpr_eval()
tpr_BBB = ROC_BBB.tpr_eval()

fpr_BBB_train = ROC_BBB.fpr_eval_train()
tpr_BBB_train = ROC_BBB.tpr_eval_train()

cd_BBB = ROC_BBB.crit_point()
cd_BBB_test = ROC_BBB.crit_point_test()
C2_BBB, cd_range_BBB = ROC_BBB.confusion_matrix_plot_crit()

tnr_BBB = ROC_BBB.tnr_eval()
ppv_BBB = ROC_BBB.ppv_eval()
acc_BBB = ROC_BBB.acc_eval()
f1_BBB = ROC_BBB.f1_eval()
mcc_BBB, mcc_list_BBB = ROC_BBB.mcc_eval()

a_BBB, b_BBB = mcc_detail(mcc_list_BBB)

roc_auc_BBB = metrics.auc(fpr_BBB, tpr_BBB)
roc_auc_BBB_train = metrics.auc(fpr_BBB_train, tpr_BBB_train)
print(roc_auc_BBB)

print(C2_BBB)
#print(C2_BBB.ravel())
len_cd.append(len(C2_BBB))
print('len_BBB', len((C2_BBB)))
sns_data.append(C2_BBB)



if len(C2_BBB) == 1:
    fig_BBB, axs_BBB = plt.subplots(1, len(C2_BBB))
    sns.heatmap(data=C2_BBB[0], annot=True, fmt='d')
    axs_BBB.set_xlabel('Pred_BBB')
    axs_BBB.set_ylabel('True_BBB')
    axs_BBB.set_title("BBB with " + str(cd_range_BBB[0][0]) + '~' + str(cd_range_BBB[0][1]))
else:

    fig_BBB, axs_BBB = plt.subplots(1, len(C2_BBB))
    for i in range(len(C2_BBB)):
        sns.heatmap(data = C2_BBB[i], ax=axs_BBB[i], annot=True, fmt='d')
        axs_BBB[i].set_xlabel('Pred_BBB')
        axs_BBB[i].set_ylabel('True_BBB')
        axs_BBB[i].set_title("BBB" + str(i+1) + ' with ' + str(cd_range_BBB[i][0]) + '~' + str(cd_range_BBB[i][1]))

#sns_data = np.array(sns_data)
print('--sns_data', sns_data)
sns_title = ['SGLD', 'pSGLD', 'SGHMC', 'BBB']
cd_range = [cd_range_SGLD, cd_range_pSGLD, cd_range_SGHMC, cd_range_BBB]
mcc_list = [b_SGLD, b_pSGLD, b_SGHMC, b_BBB]
fig, axs = plt.subplots(4, max(len_cd))


# for i in range(4):
#     for j in range(len_cd[i]):
#         print('sns_data[i][j]', sns_data[i], sns_data[i][j])
#         sns.heatmap(sns_data[i][j], ax=axs[i, j], annot=True, fmt='d')
#         axs[i, j].set_xlabel('Pred_' + sns_title[i])
#         axs[i, j].set_ylabel('True_' + sns_title[i])
#         axs[i, j].set_title(sns_title[i] + str(j+1) + ' with ' + str(cd_range[i][j][0]) + '~' + str(cd_range[i][j][1]))



# sns.heatmap(data=C2_SGLD, ax=axs[0, 0], annot=True)
# axs[0, 0].set_xlabel('Pred_SGLD')
# axs[0, 0].set_ylabel('True_SGLD')
# axs[0, 0].set_title('SGLD')
# sns.heatmap(data=C2_pSGLD, ax=axs[0, 1], annot=True)
# axs[0, 1].set_xlabel('Pred_pSGLD')
# axs[0, 1].set_ylabel('True_pSGLD')
# axs[0, 1].set_title('pSGLD')
# sns.heatmap(data=C2_SGHMC, ax=axs[1, 0], annot=True)
# axs[1, 0].set_xlabel('Pred_SGHMC')
# axs[1, 0].set_ylabel('True_SGHMC')
# axs[1, 0].set_title('SGHMC')
# sns.heatmap(data=C2_BBB, ax=axs[1, 1], annot=True)
# axs[1, 1].set_xlabel('Pred_BBB')
# axs[1, 1].set_ylabel('True_BBB')
# axs[1, 1].set_title("BBB")

########################################################################################################################
a = ['SGLD', 'pSGLD', 'SGHMC', 'BBB']
b = ['fpr', 'tpr', 'tnr', 'ppv', 'acc', 'f1', 'mcc']
output_table = np.zeros([len(a), len(b)])
ot = {}
for i in (b):
    for j in (a):
        name_str = i + '_' + j
        cd_str = 'cd_' + j
        val_cd = locals()[cd_str]

        val = locals()[name_str]

        print(name_str)
        if i == 'ppv' or i == 'f1' or i =='mcc' or i == 'acc':

            #ot[name_str] = np.unique(val)
            ot[name_str] = val[np.sort(np.unique(val, return_index=True)[1])]
        else:

            if len(np.unique(val)) > 1:

                ot[name_str] = val[val_cd][np.sort(np.unique(val[val_cd], return_index=True)[1])]
                #ot[name_str] = np.unique(val[val_cd])
            else:
                ot[name_str] = val[np.sort(np.unique(val, return_index=True)[1])]
                #ot[name_str] = np.unique(val)

print('yoyoyoy', ot)
print('mcc_list', mcc_list)
for i in range(len(b)):
    for j in range(len(a)):
        name_str = b[i] + '_' + a[j]
        #print(ot[name_str])
        #print(type(ot[name_str]))
        #output_table[i, j] = list(str(list(ot[name_str])))
        print(name_str, ot[name_str])

print('f')
print(cd_SGLD)
print(cd_pSGLD)
print(cd_SGHMC)
print(cd_BBB)



kk = []
ss = []
for i in ot.values():
    kk.append(str(i))
    ss.append(i)
    #print(str(i))

toot = {}

# print('//////////////////////////////fpr_BBB', fpr_BBB,
#       '//////////////////////////////tpr_BBB', tpr_BBB,
#       '//////////////////////////////cd_BBB', cd_BBB,
#       '//////////////////////////////C2_BBB', C2_BBB,
#       '//////////////////////////////cd_range_BBB', cd_range_BBB,
#       '//////////////////////////////tnr_BBB', tnr_BBB,
#       '//////////////////////////////ppv_BBB', ppv_BBB,
#       '//////////////////////////////acc_BBB', acc_BBB,
#       '//////////////////////////////f1_BBB', f1_BBB,
#       '//////////////////////////////mcc_BBB', mcc_BBB,
#       '//////////////////////////////mcc_list_BBB', mcc_list_BBB,
#       '//////////////////////////////a_BBB', a_BBB,
#       '//////////////////////////////b_BBB', b_BBB,
#       '//////////////////////////////roc_auc_BBB', roc_auc_BBB)

ThresSplitOutputTable = []
# for i in range(len(b)):
#     for j in range(len(a)):
#         print('len(cd_range[j])', len(cd_range[j]))
#         for l in range(len(cd_range[j])):
#             name_str = b[i] + '_' + a[j]
#             if b[i] == 'acc':
#                 ThresSplitOutputTable.append(ot[name_str][0])
#             elif b[i] == 'mcc':
#                 if len(mcc_list[j]) == 1:
#                     ThresSplitOutputTable.append(mcc_list[j][0])
#                 else:
#                     ThresSplitOutputTable.append(mcc_list[j][l])
#             else:
#                 ThresSplitOutputTable.append(ot[name_str][l])

for i in range(len(b)):
    for j in range(len(a)):
        #print('len(cd_range[j])', len(cd_range[j]))
        for l in range(len(cd_range[j])):
            name_str = b[i] + '_' + a[j]

            if b[i] == 'fpr' or b[i] == 'tpr' or b[i] == 'tnr':
                #print(locals()[name_str])
                #print(cd_range[j][l][0])
                ThresSplitOutputTable.append(locals()[name_str][int(cd_range[j][l][0]*1000)])
            elif b[i] == 'mcc':
                if len(mcc_list[j]) == 1:
                    ThresSplitOutputTable.append(mcc_list[j][0])
                else:
                    ThresSplitOutputTable.append(mcc_list[j][l])
            else:
                ThresSplitOutputTable.append(ot[name_str][l])


kk = np.array(kk)
kk = np.reshape(kk, [len(b), len(a)])

col_name = []
for i in range(len(a)):
    for j in range(len(cd_range[i])):
        col_name.append(a[i] + ' with ' + str(cd_range[i][j][0]) + '~' + str(cd_range[i][j][1]))

print('col_name', col_name)


ThresSplitOutputTable = np.reshape(ThresSplitOutputTable, [len(b), sum([len(i) for i in cd_range])])
tsotdf = pd.DataFrame(ThresSplitOutputTable, columns=col_name, index=b)

otdf = pd.DataFrame(kk, columns=a, index=b)
pd.set_option("display.max_columns", None)
print('df', otdf)
print('cd_range', cd_range)
#print('ThresSplitOutputTable', ThresSplitOutputTable)

print('tsotdf', tsotdf)

completeName = "./tsotdf.csv"
if os.path.exists(completeName):
    os.remove(completeName)


tsotdf.to_csv(completeName)
########################################################################################################################

threshold5 = int(len(fpr_SGLD)/2)

plt.figure(20)
plt.plot(start, end)

plt.plot(fpr_SGLD, tpr_SGLD, lw=1, color='red', label="SGLD, area=%0.2f)" % (roc_auc_SGLD))
plt.plot(fpr_pSGLD, tpr_pSGLD, lw=1, color='blue', label="pSGLD, area=%0.2f)" % (roc_auc_pSGLD))
plt.plot(fpr_SGHMC, tpr_SGHMC, lw=1, color='green', label="SGHMC, area=%0.2f)" % (roc_auc_SGHMC))
plt.plot(fpr_BBB, tpr_BBB, lw=1, color='orange', label="BBB, area=%0.2f)" % (roc_auc_BBB))

plt.plot(fpr_SGLD[cd_SGLD], tpr_SGLD[cd_SGLD], 'o', color='red')
plt.plot(fpr_pSGLD[cd_pSGLD], tpr_pSGLD[cd_pSGLD], 'o', color='blue')
plt.plot(fpr_SGHMC[cd_SGHMC], tpr_SGHMC[cd_SGHMC], 'o', color='green')
plt.plot(fpr_BBB[cd_BBB], tpr_BBB[cd_BBB], 'o', color='orange')


plt.plot(fpr_SGLD[cd_SGLD_test], tpr_SGLD[cd_SGLD_test], '*', color='red')
plt.plot(fpr_pSGLD[cd_pSGLD_test], tpr_pSGLD[cd_pSGLD_test], '*', color='blue')
plt.plot(fpr_SGHMC[cd_SGHMC_test], tpr_SGHMC[cd_SGHMC_test], '*', color='green')
plt.plot(fpr_BBB[cd_BBB_test], tpr_BBB[cd_BBB_test], '*', color='orange')


plt.plot(fpr_SGLD[threshold5], tpr_SGLD[threshold5], '^', color='red')
plt.plot(fpr_pSGLD[threshold5], tpr_pSGLD[threshold5], '^', color='blue')
plt.plot(fpr_SGHMC[threshold5], tpr_SGHMC[threshold5], '^', color='green')
plt.plot(fpr_BBB[threshold5], tpr_BBB[threshold5], '^', color='orange')

plt.xlim([0.00, 1.0])
plt.ylim([0.00, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
#plt.legend(['sgld','psgld','bbb'])
plt.legend(loc="lower right")

plt.figure(21)
plt.plot(start, end)
plt.plot(fpr_SGLD_train, tpr_SGLD_train, color='red', label="SGLD, area=%0.2f)" % (roc_auc_SGLD_train))
plt.plot(fpr_pSGLD_train, tpr_pSGLD_train, color='blue', label="pSGLD, area=%0.2f)" % (roc_auc_pSGLD_train))
plt.plot(fpr_SGHMC_train, tpr_SGHMC_train, color='green', label="SGHMC, area=%0.2f)" % (roc_auc_SGHMC_train))
plt.plot(fpr_BBB_train, tpr_BBB_train, color='orange', label="BBB, area=%0.2f)" % (roc_auc_BBB_train))

plt.plot(fpr_SGLD_train[cd_SGLD], tpr_SGLD_train[cd_SGLD], 'o', color='red')
plt.plot(fpr_pSGLD_train[cd_pSGLD], tpr_pSGLD_train[cd_pSGLD], 'o', color='blue')
plt.plot(fpr_SGHMC_train[cd_SGHMC], tpr_SGHMC_train[cd_SGHMC], 'o', color='green')
plt.plot(fpr_BBB_train[cd_BBB], tpr_BBB_train[cd_BBB], 'o', color='orange')

plt.xlim([0.00, 1.0])
plt.ylim([0.00, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC for training set")
plt.legend(loc="lower right")

plt.show()