import numpy as np

y_1 = 3
y_2 = 6
y = [y_1, y_2]
sigma_x = 1
sigma_y = 1
rho = 0.8

theta_1_0 = 0
theta_2_0 = 0

Covariance_matrix = [[sigma_x, rho], [rho, sigma_y]]

MAX = 500

theta_1 = np.zeros(MAX)
theta_2 = np.zeros(MAX)
theta_1[0] = theta_1_0
theta_2[0] = theta_2_0
theta = np.array([theta_1, theta_2])


