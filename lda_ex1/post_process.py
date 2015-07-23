#!/usr/bin/env python2

import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from os import listdir
from os.path import isfile, join

def post_process():
    pattern = re.compile('lls_([0-9]+)_[0-9]+\.npz')
    data_dir = 'data'
    files = [join(data_dir,f)  
        for f in listdir(data_dir) 
        if isfile(join(data_dir,f)) and re.match(pattern, f)]
    all_lls = [np.load(f)['lls'] for f in files]
    all_lls = np.array(all_lls)
    avg_lls = np.average(all_lls, axis=0)
    lls_sem = 1.96*stats.sem(all_lls)
    x = np.arange(all_lls.shape[1])
    plt.plot(x, avg_lls, linestyle='-.', color='b', label='PB_ungibbs')
    plt.fill_between(x, avg_lls-lls_sem, avg_lls+lls_sem, color='b', alpha=0.3)    
    plt.xlabel('Iterations')
    plt.ylabel('Log Likelihood')
    plt.title('Average Log Likelihood and 95% confidence interval (normal)')      
    
    '''
    pattern2 = re.compile('lls_amcmc_([0-9]+)_[0-9]+\.npz')
    files = [join(data_dir,f)  
        for f in listdir(data_dir) 
        if isfile(join(data_dir,f)) and re.match(pattern2, f)]
    all_lls = [np.load(f)['lls'] for f in files]
    all_lls = np.array(all_lls)
    avg_lls = np.average(all_lls, axis=0)
    lls_sem = 1.96*stats.sem(all_lls)
    x = np.arange(all_lls.shape[1])
    plt.plot(x, avg_lls, linestyle='-.', color='g', label='PB_amcmc')
    plt.fill_between(x, avg_lls-lls_sem, avg_lls+lls_sem, color='g', alpha=0.3)
    '''
    
    # Gibbs ll
    gibbs_lls = np.array(pd.read_csv(join(data_dir, 'gibbs_ll'), 
        sep=' ', header=None)).T
    avg_gibbs_lls = np.average(gibbs_lls, axis=0)
    gibbs_lls_sem = 1.96*stats.sem(gibbs_lls) 
    x_gibbs = np.arange(gibbs_lls.shape[1])   
    plt.plot(x_gibbs, avg_gibbs_lls, linestyle='-', color='r', label='Collapsed Gibbs')
    plt.fill_between(x_gibbs, avg_gibbs_lls-gibbs_lls_sem, 
        avg_gibbs_lls+gibbs_lls_sem, color='r', alpha=0.3)
    
    # VEM ll
    #vem_lls = np.array(pd.read_csv(join(data_dir, 'vem_ll'), 
    #    sep=' ', header=None)).T
    #avg_vem_lls = np.average(vem_lls, axis=0)
    #vem_lls_sem = 1.96*stats.sem(vem_lls)
    #x_vem = np.arange(vem_lls.shape[1])    
    #plt.plot(x_vem, avg_vem_lls, label='VEM')
    #plt.fill_between(x_vem, avg_vem_lls-vem_lls_sem, 
    #    avg_vem_lls+vem_lls_sem, alpha=0.3)
    
    # save plot
    plt.legend(loc='lower right')
    plt.savefig('data/lls.pdf', format='pdf') 

def main():
    post_process()
    
if __name__=='__main__':
    main()
