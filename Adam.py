#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:38:38 2020

@author: yuncong
"""
import numpy as np
'''
Adam optimal algorithm
input:
    weights:params for tuning
    alpha: learning rate
    gradient:gradient of cost function
output:
    theta:params after tuning
'''


class AdamOptimiser:
    def __init__(self,weights,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0      #momentum term
        self.v = 0      #rmsprop term
        self.t = 0      #number of iteration
        self.theta = weights  #params for tuning
        
    def backward_pass(self,gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.theta = self.theta - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.theta
    
    