from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import argparse
import matplotlib
from src.Stochastic_Gradient_Langevin_Dynamics.model import *
import matplotlib.pyplot as plt

matplotlib.use('Agg')


if __name__ == '__main__':
    print('sgld')
    parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on MNIST with Stochastic Gradient Langevin Dynamics')
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
    args = parser.parse_args()


    image_trans_size = args.image_trans_size
    prior_sig = args.prior_sig
    transform_covid19 = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=(0, 180)),
        transforms.Resize(image_trans_size),
        transforms.CenterCrop(image_trans_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1)
    ])

    trainset = torchvision.datasets.ImageFolder(root="./notebooks/data/COVID/train", transform=transform_covid19)
    valset = torchvision.datasets.ImageFolder(root="./notebooks/data/COVID/test", transform=transform_covid19)

    train_data_len = len(trainset.targets)
    test_data_len = len(valset.targets)
    use_cuda = torch.cuda.is_available()
    NTrainPoints = train_data_len

    # Where to save models weights
    models_dir = args.models_dir
    # Where to save plots and error, accuracy vectors
    results_dir = args.results_dir

    if args.use_preconditioning == 1:
        models_dir = 'p' + models_dir
        results_dir = 'p' + results_dir

    mkdir(models_dir)
    mkdir(results_dir)
    # ------------------------------------------------------------------------------------------------------
    # train config

    batch_size = args.batch_size
    nb_epochs = args.epochs
    log_interval = 1


    # ------------------------------------------------------------------------------------------------------
    # dataset
    cprint('c', '\nData:')

    # load data

    # data augmentation
    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])
    #
    # use_cuda = torch.cuda.is_available()
    #
    # trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
    # valset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)

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
    ########################################################################################

    if args.use_preconditioning == 1:
        net = Net_langevin(lr=lr, channels_in=1, side_in=image_trans_size, cuda=use_cuda, classes=2, N_train=NTrainPoints,
                           prior_sig=prior_sig, nhid=1200, use_p=True)
        pstr = 'p'
        print('pstr')

    else:
        net = Net_langevin(lr=lr, channels_in=1, side_in=image_trans_size, cuda=use_cuda, classes=2, N_train=NTrainPoints,
                           prior_sig=prior_sig, nhid=1200, use_p=False)

    #Net_langevin(lr=lr, channels_in=1, side_in=image_trans_size, cuda=use_cuda, classes=2, N_train=NTrainPoints, prior_sig=prior_sig)

    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    epoch = 0
    cprint('c', '\nTrain:')

    ## weight saving parameters #######
    start_save = 15
    save_every = 2  # We sample every 2 epochs as I have found samples to be correlated after only 1
    N_saves = 100  # Max number of saves
    ###################################

    print('  init cost variables:')
    kl_cost_train = np.zeros(nb_epochs)
    pred_cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)

    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_err = np.inf

    nb_its_dev = 1

    tic0 = time.time()
    for i in range(epoch, nb_epochs):

        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0

        for x, y in trainloader:
            cost_pred, err = net.fit(x, y)

            err_train[i] += err
            pred_cost_train[i] += cost_pred
            nb_samples += len(x)

        pred_cost_train[i] /= nb_samples
        err_train[i] /= nb_samples

        toc = time.time()
        net.epoch = i
        # ---- print
        print("it %d/%d, Jtr_pred = %f, err = %f, " % (i, nb_epochs, pred_cost_train[i], err_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))

        # ---- save weights
        if i >= start_save and i % save_every == 0:
            net.save_sampled_net(max_samples=N_saves)

        # ---- dev
        if i % nb_its_dev == 0:
            net.set_mode_train(False)
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                cost, err, probs = net.eval(x, y)

                cost_dev[i] += cost
                err_dev[i] += err
                nb_samples += len(x)

            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples

            cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))

            if err_dev[i] < best_err:
                best_err = err_dev[i]
                cprint('b', 'best test error')
                net.save(models_dir+'/theta_best.dat')

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # Save weight samples from the posterior
    save_object(net.weight_set_samples, models_dir+'/state_dicts.pkl')

    ## ---------------------------------------------------------------------------------------------------------------------
    # results
    cprint('c', '\nRESULTS:')
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = np.min(cost_dev)
    best_cost_train = np.min(pred_cost_train)
    err_dev_min = err_dev[::nb_its_dev].min()

    print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
    print('  err_dev: %f' % (err_dev_min))
    print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
    print('  time_per_it: %fs\n' % (runtime_per_it))

    ## Save results for plots
    np.save(results_dir + '/cost_train.npy', pred_cost_train)
    np.save(results_dir + '/cost_dev.npy', cost_dev)
    np.save(results_dir + '/err_train.npy', err_train)
    np.save(results_dir + '/err_dev.npy', err_dev)

    ## ---------------------------------------------------------------------------------------------------------------------
    # fig cost vs its

    textsize = 15
    marker = 5

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
    ax1.plot(pred_cost_train, 'r--')
    ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('classification costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(dpi=100)
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel('% error')
    ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * err_dev[::nb_its_dev], 'b-')
    ax2.semilogy(100 * err_train, 'r--')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/err.png', bbox_extra_artists=(lgd,), box_inches='tight')