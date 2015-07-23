#!/usr/bin/env python2

import re

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from os import listdir
from os.path import isfile, join

def gen_perm(inds, choice):
    perm = [inds[0]]
    gen_perm_r(inds[1:], choice, perm)
    return perm

def gen_perm_r(inds, choice, perm):
    if inds.size == 0:
        return perm
    else:
        perm.insert(choice[0], inds[0])
        #print inds[0], choice[0], perm
        gen_perm_r(inds[1:], choice[1:], perm)

def post_process():
    N = 10
    pattern1 = re.compile('lls_([0-9]+)\.npz')
    pattern2 = re.compile('avg_samples_([0-9]+)\.npz')
    data_dir = 'data'
    files2 = [ (re.search(pattern2, f).group(1), join(data_dir,f) )  
        for f in listdir(data_dir) 
        if isfile(join(data_dir,f)) and re.match(pattern2, f)]
    all_rims = [np.load(f) for _,f in files2]    
    pis = []
    catss = []
    for rim in all_rims:
        pi = rim['arr_0'][0]
        cats = [rim['arr_{}'.format(i)] for i in range(1,N)]
        pis.append(pi)
        catss.append(cats)
    avg_pi = np.average(np.array(pis), axis=0)
    avg_cats = []
    for i in range(N-1):
       cats_l = [cats[i] for cats in catss]
       avg_cats.append(np.average(cats_l, axis=0))
    
    # maximum likelihood categories
    choices = np.vstack([np.argmax(avg_cat, axis=1) for avg_cat in avg_cats])
    # build the ML permutations
    inds = np.arange(10)
    perms = [gen_perm(inds, choice) for choice in choices.T]
    # replaces numbers with strings
    item_l = ['shrimp', 'sea eel', 'tuna', 'squid', 'sea urchin', 
        'salmon roe', 'egg', 'fatty tuna', 'tuna roll', 'cucumber roll'] 
    perms = [list(row) for row in np.array(perms).T]
    profiles = [[item_l[el] for el in perm] for perm in perms ]
    s = ('\hline '+' & '.join(['$\pi_{} = {}$'.format(i+1,round(el,3)) 
        for i,el in enumerate(list(avg_pi))])+
        '\\\\ \hline ' +' \\\\ '.join([' & '.join(row) 
        for row in profiles])+'\\\\ \hline')
    with open('data/table.tex', 'w') as f:
        f.write(s)
    
    files = [ (re.search(pattern1, f).group(1), join(data_dir,f) )  
        for f in listdir(data_dir) 
        if isfile(join(data_dir,f)) and re.match(pattern1, f)]
    all_lls = [np.load(f)['lls'] for _,f in files]
    all_lls = np.array(all_lls)
    avg_lls = np.average(all_lls, axis=0)
    lls_sem = 1.96*stats.sem(all_lls)
    x = np.arange(all_lls.shape[1])
    plt.plot(x, avg_lls, linestyle='-.', color='b', label='PB')
    plt.fill_between(x, avg_lls-lls_sem, avg_lls+lls_sem, color='b', alpha=0.3)    
    plt.xlabel('Iterations')
    plt.ylabel('Log Likelihood')
    plt.title('Average Joint Prob. and 95% confidence interval (normal)')
    plt.savefig('data/avg_ll.pdf', format='pdf')
    
def main():
    post_process()
    
if __name__=='__main__':
    main()
    

