import numpy as np

class HMC(object):
    def __init__(self, y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0, scale, m1, m2):
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
        self.scale = scale
        self.m1 = m1
        self.m2 = m2


    def binorm(self, x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
        self.result = 1 / (2 * np.pi * self.sigma_x * self.sigma_y * np.sqrt(1 - self.rho ** 2)) * np.exp(-1 / (2 * (1 - self.rho ** 2)) * (
                    ((x - mu_x) / self.sigma_x) ** 2 - 2 * self.rho * ((x - mu_x) / self.sigma_x) * ((y - mu_y) / self.sigma_y) + (
                        (y - mu_y) / self.sigma_y) ** 2))

        return self.result

    def HMC_al(self, y_1, y_2, sigma_x, sigma_y, rho, MAX, theta_1_0, theta_2_0):
        self.momentum1 = np.random.multivariate_normal([0, self.m1])
        self.momentum2 = np.random.multivariate_normal([0, self.m2])

        self.Covariance_matrix = [[self.scale*self.sigma_x, self.scale*self.rho], [self.scale*self.rho, self.scale*self.sigma_y]]
        for i in range(1, self.MAX):
            self.theta_temp = np.random.multivariate_normal([self.theta_1[i-1], self.theta_2[i-1]], self.Covariance_matrix, 1)
            self.c1 = self.binorm(self.theta_temp[0][0], self.theta_temp[0][1], self.y_1, self.y_2, self.sigma_x, self.sigma_y, self.rho)
            self.c2 = self.binorm(self.theta_1[i-1], self.theta_2[i-1], self.y_1, self.y_2, self.sigma_x, self.sigma_y, self.rho)
            # print(self.c2)
            self.c = self.c1/self.c2
            if np.random.random_sample() < min(1, self.c):
                self.theta_1[i] = self.theta_temp[0][0]
                self.theta_2[i] = self.theta_temp[0][1]
            else:
                self.theta_1[i] = self.theta_1[i-1]
                self.theta_2[i] = self.theta_2[i-1]
        return self.theta_1, self.theta_2