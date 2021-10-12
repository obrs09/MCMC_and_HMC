import numpy as np
import time
import matplotlib.pyplot as plt
#from config import *
from Gibbs import Gibbs
from MH import MH

y_1 = 3
y_2 = 6
y = [y_1, y_2]
sigma_x = 1
sigma_y = 1
rho = 0.8
#theta_1_0 = -5
#theta_2_0 = -10
Covariance_matrix = [[sigma_x, rho], [rho, sigma_y]]
scale = 0.5
MAX = 1000

n = 3
sn = -10
en = 10
theta_1_init = np.linspace(sn, en, n)
theta_2_init = np.linspace(sn, en, n)

i = 0
fig1, axs1 = plt.subplots(4, n * n)
fig2, axs2 = plt.subplots(4, n * n)
for theta_1_0 in theta_1_init:
    for theta_2_0 in theta_2_init:
        print(i)
        gibbs_start_time = time.time()

        x = Gibbs(y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0)
        a1, a2 = x.Gibbs_sample(y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0)

        gibbs_end_time = time.time()
        cal_time_gibbs = round(gibbs_end_time - gibbs_start_time, 3)
        #print("Gibbs.py finished")
        #print("time is", cal_time_gibbs)

        #####################################################################################

        MH_start_time = time.time()

        y = MH(y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0, scale)
        b1, b2 = y.MH_al(y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0)

        MH_end_time = time.time()
        cal_time_MH = round(MH_end_time - MH_start_time, 3)
        #print("MH.py finished")
        #print("time is", cal_time_MH)

        fig1.suptitle(f'Gibbs y_1={y_1}, y_2={y_2}, rho={rho}, num_point = {MAX}')
        axs1[0, i].plot(a1, a2, '.')
        axs1[0, i].plot(y_1, y_2, '*', color='r')
        axs1[0, i].set_title(f'theta_1 = {theta_1_0}, '
                             f'\ntheta_2 = {theta_2_0}, '
                             f'\ntime={cal_time_gibbs}')

        axs1[1, i].plot(a1, color='b')
        axs1[1, i].plot(a2, color='g')
        axs1[2, i].acorr(a1, maxlags = 10)
        axs1[3, i].acorr(a2, maxlags = 10)

        # fig2, axs2 = plt.subplots(4, n*n)
        fig2.suptitle(f'MH algo y_1={y_1}, y_2={y_2}, rho={rho}, num_point = {MAX}')
        axs2[0, i].plot(b1, b2, '.')
        axs2[0, i].plot(y_1, y_2, '*', color='r')
        axs2[0, i].set_title(f'theta_1 = {theta_1_0}, '
                             f'\ntheta_2 = {theta_2_0}, '
                             f'\ntime={cal_time_MH}, '
                             f'\nscale = {scale}')
        axs2[1, i].plot(b1, color='b')
        axs2[1, i].plot(b2, color='g')
        axs2[2, i].acorr(b1, maxlags = 10)
        axs2[3, i].acorr(b2, maxlags = 10)

        i = i + 1


plt.show()