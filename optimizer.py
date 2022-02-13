import numpy as np
import tensorflow as tf
from copy import deepcopy
import time


def get_rmsd(sample): return np.sqrt(np.mean(np.square(sample['adv']-sample['ori'] + 1e-18)))


def get_alpha(ao, rmsd_first_thres, grad_func, sample, epsilon):
    alpha = 10
    rmsd_error = 100
    start = time.time()
    ori_sample = deepcopy(sample)
    while rmsd_error > rmsd_first_thres / 100 and time.time() - start < 3:
        fuck = deepcopy(ori_sample)
        rmsd = get_rmsd(ao.optimizer(grad_func, fuck, 0, alpha, epsilon))
        rmsd_error = np.abs(rmsd - rmsd_first_thres)
        alpha = alpha / rmsd * rmsd_first_thres
        #print(alpha, rmsd, rmsd_error)
    return alpha


def update(sample, direction_value, alpha, epsilon):
    sample['adv'] = np.clip(sample['adv'] + alpha * direction_value[0], 0, 255)
    sample['adv'] = np.clip(sample['adv'], sample['ori'] - epsilon, sample['ori'] + epsilon)
    return sample


class Attack_Optimizer:
    def __init__(self, optimizer_name):
        if   optimizer_name == 'GD':           self.optimizer = self._GD
        elif optimizer_name == 'MGD':          self.optimizer = self._MGD
        elif optimizer_name == 'NAGD':         self.optimizer = self._NAGD
        elif optimizer_name == 'RMSP':         self.optimizer = self._RMSP
        elif optimizer_name == 'Adam':         self.optimizer = self._Adam
        elif optimizer_name == 'AdamW':        self.optimizer = self._AdamW
        elif optimizer_name == 'LAdam':        self.optimizer = self._LAdam
        elif optimizer_name == 'Yogi':         self.optimizer = self._Yogi
        elif optimizer_name == 'MSVAG':        self.optimizer = self._MSVAG
        elif optimizer_name == 'AdaB':         self.optimizer = self._AdaB
        elif optimizer_name == 'BFGS':         self.optimizer = self._BFGS
        elif optimizer_name == 'Shampoo':      self.optimizer = self._Shampoo
        elif optimizer_name == 'MAS':          self.optimizer = self._MAS
        elif optimizer_name == 'AAdam':        self.optimizer = self._AAdam
        else: raise ValueError('Invalid Optimizer Name')

    def _GD(self, grad_func, sample, iter_done, alpha, epsilon): 
        direction_value = grad_func(sample['adv'])
        return update(sample, direction_value, alpha, epsilon)

    def _MGD(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.momentum_parameter = 0.9

        if iter_done == 0: self.momentum_gradient = 0
        
        self.momentum_gradient = self.momentum_parameter * self.momentum_gradient + grad_func(sample['adv'])
        #print(np.mean(grad_func(sample['adv'])), np.mean(self.momentum_gradient))
        
        direction_value = self.momentum_gradient
        return update(sample, direction_value, alpha, epsilon)
        
    def _NAGD(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.momentum_parameter = 0.9

        if iter_done == 0:
            self.momentum_gradient = 0
            self.nes_image = sample['adv']
        else: self.nes_image = update(sample, self.momentum_gradient, alpha * self.momentum_parameter, epsilon)['adv']
        
        self.momentum_gradient = self.momentum_parameter * self.momentum_gradient + grad_func(self.nes_image)
        
        direction_value = self.momentum_gradient
        return update(sample, direction_value, alpha, epsilon)

    def _RMSP(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.rho = 0.9
        self.delta = 1e-8

        grad_value = grad_func(sample['adv'])
        if iter_done == 0: self.z = 0
            
        self.z = self.rho * self.z + (1-self.rho) * (grad_value * grad_value)
        
        direction_value = grad_value / np.sqrt(self.z + self.delta)
        return update(sample, direction_value, alpha, epsilon)

    def _Adam(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.delta = 1e-8

        grad_value = grad_func(sample['adv'])
        if iter_done == 0:
            self.m = 0
            self.v = 0

        self.m = (self.beta1 * self.m + (1-self.beta1) * grad_value) #/ (1 - self.beta1 ** (iter_done+1))
        self.v = (self.beta2 * self.v + (1-self.beta2) * (grad_value * grad_value)) #/ (1 - self.beta2 ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        
        direction_value = self.m / (np.sqrt(self.v) + self.delta)
        return update(sample, direction_value, alpha, epsilon)

    def _LAdam(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.delta = 1e-8
        self.lbda = 1e-12

        grad_value = grad_func(sample['adv']) - sample['adv'] * self.lbda
        if iter_done == 0:
            self.m = 0
            self.v = 0

        self.m = (self.beta1 * self.m + (1-self.beta1) * grad_value) #/ (1 - self.beta1 ** (iter_done+1))
        self.v = (self.beta2 * self.v + (1-self.beta2) * (grad_value * grad_value)) #/ (1 - self.beta2 ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        
        direction_value = self.m / (np.sqrt(self.v) + self.delta)
        return update(sample, direction_value, alpha, epsilon)

    def _AdamW(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.delta = 1e-8
        self.phy = 0.005

        grad_value = grad_func(sample['adv'])
        if iter_done == 0:
            self.m = 0
            self.v = 0

        self.m = (self.beta1 * self.m + (1-self.beta1) * grad_value) #/ (1 - self.beta1 ** (iter_done+1))
        self.v = (self.beta2 * self.v + (1-self.beta2) * (grad_value * grad_value)) #/ (1 - self.beta2 ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        
        direction_value = self.m / (np.sqrt(self.v) + self.delta) + sample['adv'] * self.phy
        return update(sample, direction_value, alpha, epsilon)

    def _Yogi(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.delta = 1e-8

        grad_value = grad_func(sample['adv'])
        grad_value_square = grad_value * grad_value
        if iter_done == 0:
            self.m = 0
            self.v = 0

        self.m = (self.beta1 * self.m + (1-self.beta1) * grad_value) #/ (1 - self.beta1 ** (iter_done+1))
        self.v = (self.v - (1-self.beta2) * grad_value_square * np.sign(self.v - grad_value_square)) #/ (1 - self.beta2 ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        
        direction_value = self.m / (np.sqrt(self.v) + self.delta)
        return update(sample, direction_value, alpha, epsilon)

    def _MSVAG(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.beta = 0.9
        self.delta = 1e-8

        grad_value = grad_func(sample['adv'])
        if iter_done == 0:
            self.m = 0
            self.v = 0

        self.m = (self.beta * self.m + (1-self.beta) * grad_value) #/ (1 - self.beta ** (iter_done+1))
        self.v = (self.beta * self.v + (1-self.beta) * (grad_value * grad_value)) #/ (1 - self.beta ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        self.rho = (1-self.beta) * (1 + self.beta ** (iter_done+2)) / (1 + self.beta) / (1 - self.beta ** (iter_done+2))
        #self.m_square = self.m * self.m
        #self.s = (self.v - self.m_square) / (1 - self.rho)
        
        #try:
        direction_value = (1-self.rho) * (self.m * self.m) / ((1-2*self.rho) * (self.m * self.m) + self.rho * self.v) * self.m
        #with warnings.catch_warnings():
            #print('Caught')
            #direction_value = grad_value
        return update(sample, direction_value, alpha, epsilon)

    def _AdaB(self, grad_func, sample, iter_done, alpha, epsilon): 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.delta = 1e-8

        grad_value = grad_func(sample['adv'])
        if iter_done == 0:
            self.m = 0
            self.v = 0

        self.m = (self.beta1 * self.m + (1-self.beta1) * grad_value) #/ (1 - self.beta1 ** (iter_done+1))
        self.v = (self.beta2 * self.v + (1-self.beta2) * ((grad_value-self.m) * (grad_value-self.m))) #/ (1 - self.beta2 ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        
        direction_value = self.m / (np.sqrt(self.v) + self.delta)
        return update(sample, direction_value, alpha, epsilon)

    def _BFGS(self, grad_func, sample, iter_done, alpha, epsilon): 
        grad_value = grad_func(sample['adv'])
        sample_reshape = sample['adv'].reshape(-1, 1)
        grad_value_reshape = grad_value.reshape(-1, 1)
        if iter_done == 0:
            self.I = np.identity(sample_reshape.shape[0], dtype=np.bool) ### GPU memory out
            self.G = 0
            self.g_before = 0
            self.sample_before = 0

        A = (sample_reshape - self.sample_before).dot((grad_value_reshape - self.g_before).T) \
            / (sample_reshape - self.sample_before).T.dot(grad_value_reshape - self.g_before)
        self.G = (self.I - A).dot(self.G).dot((self.I - A).T) + A
        self.sample_before = sample_reshape
        self.g_before = grad_value_reshape

        direction_value = self.G.dot(grad_value_reshape).reshape(sample['adv'].shape)
        return update(sample, direction_value, alpha, epsilon)
        
    def _Shampoo(self, grad_func, sample, iter_done, alpha, epsilon): 
        # https://github.com/moskomule/shampoo.pytorch
        grad_value = grad_func(sample['adv'])
        original_size = grad_value.shape
        order = 4
        self.delta = 1e-8
        if iter_done == 0:
            self.precond, self.inv_precond = {}, {}
            for dim_id, dim in enumerate(original_size):
                self.precond[dim_id] = self.delta * np.eye(dim)

        def matrix_power(matrix, power):
            u, s, v = np.linalg.svd(matrix)
            return u @ np.diag(s ** power) @ (v)

        for dim_id, dim in enumerate(original_size): 
            transposed_dim = [dim_id, 1, 2, 3]
            transposed_dim[dim_id] = 0
            grad_value = grad_value.transpose(transposed_dim[0], transposed_dim[1], transposed_dim[2], transposed_dim[3])
            
            transposed_size = grad_value.shape
            grad_value = grad_value.reshape(dim, -1)
            grad_t = grad_value.T
            self.precond[dim_id] += grad_value @ grad_t
 
            if dim_id == order - 1:
                grad_value = grad_t @ matrix_power(self.precond[dim_id], (-1/order))
                grad_value = grad_value.reshape(original_size)
            else:
                grad_value = matrix_power(self.precond[dim_id], (-1/order)) @ grad_value
                grad_value = grad_value.reshape(transposed_size)

        direction_value = grad_value
        return update(sample, direction_value, alpha, epsilon)

    def _MAS(self, grad_func, sample, iter_done, alpha, epsilon): 
        """
        grad_value = grad_func(sample['adv'])
        original_size = grad_value.shape
        order = 4
        self.delta = 1e-8
        if iter_done == 0:
            self.precond, self.inv_precond = {}, {}
            for dim_id, dim in enumerate(original_size):
                self.precond[dim_id] = self.delta * np.eye(dim)

        def matrix_power(matrix, power):
            u, s, v = np.linalg.svd(matrix)
            return u @ np.diag(s ** power) @ (v)

        for dim_id, dim in enumerate(original_size): 
            transposed_dim = [dim_id, 1, 2, 3]
            transposed_dim[dim_id] = 0
            grad_value = grad_value.transpose(transposed_dim[0], transposed_dim[1], transposed_dim[2], transposed_dim[3])
            
            transposed_size = grad_value.shape
            grad_value = grad_value.reshape(dim, -1)
            grad_t = grad_value.T
            self.precond[dim_id] += grad_value @ grad_t
 
            if dim_id == order - 1:
                grad_value = grad_t @ matrix_power(self.precond[dim_id], (-1/order))
                grad_value = grad_value.reshape(original_size)
            else:
                grad_value = matrix_power(self.precond[dim_id], (-1/order)) @ grad_value
                grad_value = grad_value.reshape(transposed_size)
        """
        

        self.beta = 0.9
        self.delta = 1e-8

        grad_value = grad_func(sample['adv'])
        if iter_done == 0:
            self.m = 0
            self.v = 0

        self.m = (self.beta * self.m + (1-self.beta) * grad_value) / (1 - self.beta ** (iter_done+1))
        self.v = (self.beta * self.v + (1-self.beta) * ((grad_value-self.m) * (grad_value-self.m))) / (1 - self.beta ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        self.rho = (1-self.beta) * (1 + self.beta ** (iter_done+2)) / (1 + self.beta) / (1 - self.beta ** (iter_done+2))
        #self.m_square = self.m * self.m
        #self.s = (self.v - self.m_square) / (1 - self.rho)
        
        direction_value = (1-self.rho) * (self.m * self.m) / ((1-2*self.rho) * (self.m * self.m) + self.rho * self.v) * self.m
        if np.sum(np.isnan(direction_value)) > 0: direction_value = grad_value
        
        #direction_value = grad_value
        return update(sample, direction_value, alpha, epsilon)


    def _AAdam(self, grad_func, sample, iter_done, alpha, epsilon):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.delta = 1e-8

        grad_value = grad_func(sample['adv'])
        if iter_done == 0:
            self.m = 0
            #self.v = 0
            return update(sample, grad_value, alpha, epsilon)

        self.m = (self.beta1 * self.m + (1-self.beta1) * grad_value) #/ (1 - self.beta1 ** (iter_done+1))
        #self.v = (self.beta2 * self.v + (1-self.beta2) * ((grad_value-self.m) * (grad_value-self.m))) #/ (1 - self.beta2 ** (iter_done+1))
        #if iter_done != 0: self.m = self.m / (1 - self.beta1 ** (iter_done+1)); self.v = self.v / (1 - self.beta2 ** (iter_done+1))
        
        direction_value = self.m / (np.abs(sample['adv']-sample['ori']) + self.delta)
        return update(sample, direction_value, alpha, epsilon)