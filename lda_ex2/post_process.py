#!/usr/bin/env python2

import sys
import re

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
#from scipy.stats import entropy
from os import listdir
from os.path import isfile, join

def post_process():
    #pattern1 = re.compile('last_sample_[0-9]+\.npz')
    #pattern2 = re.compile('phi_church1_[0-9]+.csv')
    #pattern3 = re.compile('phi_church2_[0-9]+.csv')
    pattern1 = re.compile('lls_[0-9]+\.npz')
    pattern2 = re.compile('ll_church1_[0-9]+.csv')
    pattern3 = re.compile('ll_church2_[0-9]+.csv')
    data_dir = 'data'
    files_pb = [join(data_dir,f)  
        for f in listdir(data_dir) 
        if isfile(join(data_dir,f)) and re.match(pattern1, f)]
    #all_phis = [np.load(f)['arr_1'] for f in files_pb]
    all_lls = [np.load(f)['lls'] for f in files_pb]
    all_lls = np.array(all_lls)
    lls_sem = stats.sem(all_lls)
    avg_lls = np.average(all_lls, axis=0)
    files_church1 = [join(data_dir,f)
        for f in listdir(data_dir) 
        if isfile(join(data_dir,f)) and re.match(pattern2, f)]
    all_church1 = [[float(i) for i in open(f,'r').read().strip().split()] 
        for f in files_church1]
    all_church1 = np.array(all_church1)
    avg_church1 = np.average(all_church1, axis=0)
    church1_sem = stats.sem(all_church1)
    #all_phis_church1 = [np.loadtxt(open(f,'rb')) for f in files_church1]
    files_church2 = [join(data_dir,f)  
        for f in listdir(data_dir) 
        if isfile(join(data_dir,f)) and re.match(pattern3, f)]
    all_church2 = [[float(i) for i in open(f,'r').read().strip().split()] 
        for f in files_church2]
    all_church2 = np.array(all_church2)
    avg_church2 = np.average(all_church2, axis=0)
    church2_sem = stats.sem(all_church2)
    #all_phis_church2 = [np.loadtxt(open(f,'rb')) for f in files_church2]        
    #for i,(phi,phi_church1, phi_church2) in enumerate(
    #    zip(all_phis, all_phis_church1, all_phis_church2)):
    #    T = int(phi.shape[0])
    #    L = int(T/2)
    #    phi = [phi_row.reshape(L,L) for phi_row in phi]
    #    phi_church1 = [phi_row.reshape(L,L) for phi_row in phi_church1]
    #    phi_church2 = [phi_row.reshape(L,L) for phi_row in phi_church2]
    #    plot_topics(T, phi, 'data/phi_pb_{}'.format(i))
    #    plot_topics(T, phi_church1, 'data/phi_church1_{}'.format(i))
    #    plot_topics(T, phi_church2, 'data/phi_church2_{}'.format(i))
    
    x = np.arange(all_lls.shape[1])
    plt.plot(x, avg_lls, linestyle='-.', color='b', label='PB')
    plt.fill_between(x, avg_lls-lls_sem, avg_lls+lls_sem, color='b', alpha=0.3)    
    plt.xlabel('Iterations')
    plt.ylabel('Log Likelihood')
    plt.title('Average Log Likelihood +/- Standard Error')      
    
    plt.plot(x, avg_church1, linestyle='--', color='r', label='Church1')
    plt.fill_between(x, avg_church1-church1_sem, 
        avg_church1+church1_sem, color='r', alpha=0.3)
    
    plt.plot(x, avg_church2, linestyle=':', color='g', label='Church2')
    plt.fill_between(x, avg_church2-church2_sem, 
        avg_church2+church2_sem, color='g', alpha=0.3)
        
    plt.legend(loc='lower right')
    plt.savefig('data/lls.pdf', format='pdf')

def main():
    post_process()
    
if __name__=="__main__":
    main()
