import torch
import numpy as np
import torch.nn as nn
import sys

import numpy as np
from models.NonBayesianModels.AlexNet import AlexNet
from main_frequentist import getModel
import data
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
import config_bayesian as cfg
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import config_frequentist as cfg
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data as dat
from main_bayesian import getModel
import config_bayesian as cfg
import os

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softplus(x):
    return np.log(1 + np.exp(x))

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
layer_type = cfg.layer_type
activation_type = cfg.activation_type
priors = cfg.priors

train_ens = cfg.train_ens
valid_ens = cfg.valid_ens
nb_epochs = cfg.n_epochs
lr = cfg.lr_start
num_workers = cfg.num_workers
valid_size = cfg.valid_size
batch_size = cfg.batch_size
beta_type = cfg.beta_type

save_data = True
use_preconditioning = 0
pstr = "p"
image_trans_size = 224
# def imshow(image, ax=None, title=None, normalize=True):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
#     image = image.numpy().transpose((1, 2, 0))
#
#     if normalize:
#         mean = np.array([0.5, 0.5, 0.5])
#         std = np.array([0.5, 0.5, 0.5])
#         image = std * image + mean
#         image = np.clip(image, 0, 1)
#
#     ax.imshow(image)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.tick_params(axis='both', length=0)
#     ax.set_xticklabels('')
#     ax.set_yticklabels('')
#
#     return ax

def gaussian(x, mu, sig):
    y = 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-1/2 * np.power(((x - mu) / sig), 2))
    return y

def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)

    net_out, k = model(input_images)

    pred = torch.mean(net_out, dim=0).cpu().detach().numpy()
    if normalized:
        prediction = F.softplus(net_out)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)

    else:
        p_hat = F.softmax(net_out, dim=1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    epistemic = np.diag(epistemic)

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
    aleatoric = np.diag(aleatoric)

    return pred, epistemic, aleatoric, net_out, prediction, k

if __name__ == '__main__':

    train_on_gpu = torch.cuda.is_available()

    net_type = 'alexnet'
    classes = ["covid19", "non"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layer_type = cfg.layer_type
    activation_type = cfg.activation_type

    trainset, testset, inputs, outputs = dat.getDataset('covid19')

    model = getModel(net_type, inputs, outputs, priors=None, layer_type=layer_type, activation_type=activation_type)
    model.load_state_dict(torch.load('./checkpoints/covid19/bayesian/model_alexnet_bbb_softplus.pt'))
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    #see image
    transform_see = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # testset_see = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/test",
    #                                            transform=transform_see)


    #predict img
    transform_covid19 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    testset = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/test",
                                               transform=transform_covid19)

    print('from', testset.class_to_idx)
    error_num = 0
    prep = []
    prob = []

    for num_test in range(len(testset)):
        test_img_see = testset[num_test]
        test_img = testset[num_test]
        test_loader = torch.utils.data.DataLoader(test_img)
        real = test_img[1]
        test_img = test_img[0].to(device)
        pred_covid19, epistemic, aleatoric, net_out, prediction, k = get_uncertainty_per_image(model, test_img, T=100, normalized=True)
        final_prob_covid19 = pred_covid19/sum(pred_covid19)
        predict = np.argmax(final_prob_covid19)
        if real == predict:
            error = 0
        else:
            error = 1
            error_num += 1

        if num_test % 100 == 0:
            print(num_test)

        prep.append(predict)
        prob.append(final_prob_covid19)
        torch.cuda.empty_cache()

    print('error%', error_num/len(testset))
    print(prob)

    if save_data == True:
        save_path = '../Bnn-covid-conv/BBB_predict_data'
        if use_preconditioning == 1:
            save_path = pstr + save_path
            print('in pSGLD path', save_path)
        mkdir(save_path)
        file_name = "BBB_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs, lr, batch_size, image_trans_size)
        completeName = os.path.join(save_path, file_name)
        print('c', completeName)
        if os.path.exists(completeName):
            os.remove(completeName)
        # df = pd.DataFrame(prob)
        # df.to_csv(completeName)
        np.savetxt(completeName, prob, delimiter=",")


#######################################################################################################################




    trainset = torchvision.datasets.ImageFolder(root="./covid-chestxray-dataset/output/train",
                                               transform=transform_covid19)

    print('from', trainset.class_to_idx)
    error_num = 0
    prep = []
    prob = []

    for num_train in range(len(trainset)):
        # test_img_see = trainset[num_test]
        train_img = trainset[num_train]
        train_loader = torch.utils.data.DataLoader(train_img)
        real = train_img[1]
        train_img = train_img[0].to(device)
        pred_covid19, epistemic, aleatoric, net_out, prediction, k = get_uncertainty_per_image(model, train_img, T=100,
                                                                                               normalized=True)
        final_prob_covid19 = pred_covid19 / sum(pred_covid19)
        predict = np.argmax(final_prob_covid19)
        if real == predict:
            error = 0
        else:
            error = 1
            error_num += 1
        if num_train % 100 == 0:
            print(num_train)

        prep.append(predict)
        prob.append(final_prob_covid19)
        torch.cuda.empty_cache()

    print(error_num / len(trainset))
    print(prob)



    if save_data == True:
        save_path = '../Bnn-covid-conv/BBB_predict_data'
        if use_preconditioning == 1:
            save_path = pstr + save_path
            print('in pSGLD path', save_path)
        mkdir(save_path)
        file_name = "BBB_train_epochs=%d_lr=%f_batch_size=%d_image_trans_size=%d.csv" \
                    % (nb_epochs, lr, batch_size, image_trans_size)
        completeName = os.path.join(save_path, file_name)
        print('c', completeName)
        if os.path.exists(completeName):
            os.remove(completeName)
        # df = pd.DataFrame(prob)
        # df.to_csv(completeName)
        np.savetxt(completeName, prob, delimiter=",")

