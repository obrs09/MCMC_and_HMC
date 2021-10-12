import numpy as np
import matplotlib.pyplot as plt
#from config import *

class Gibbs(object):
    def __init__(self, y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0):
        self.y_1 = y_1
        self.y_2 = y_2
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho
        self.MAX = MAX
        self.theta_1_0 = theta_1_0
        self.theta_2_0 = theta_2_0
        self.theta_1 = np.zeros(MAX)
        self.theta_2 = np.zeros(MAX)
        self.theta_1[0] = theta_1_0
        self.theta_2[0] = theta_2_0
        self.theta = np.array([self.theta_1, self.theta_2])

    def Gibbs_sample(self, y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0):
        for i in range(1, self.MAX):
            self.theta_1_temp = np.random.normal(self.y_1 + self.rho*(self.theta_2[i-1] - y_2), 1 - self.rho*self.rho, 1)
            self.theta_1[i] = self.theta_1_temp
            self.theta_2_temp = np.random.normal(self.y_2 + self.rho*(self.theta_1[i] - y_1), 1 - self.rho*self.rho, 1)
            self.theta_2[i] = self.theta_2_temp
        return self.theta_1, self.theta_2


