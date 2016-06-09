#!/usr/bin/env python2

import os
import shutil
import sys

import numpy as np

from numpy.random import multinomial
from numpy.random import dirichlet

sys.path.append("..") # for utils
from utils import (rewrite_dir, generate_lda, write_pb, write_txt, plot_topics,
    gen_lda2)

def write_pb_txt(data_dir, idx, W, L, T, D, N, phi, alpha, beta):
    #B = generate_lda(T, W, D, N, phi, alpha)
    B,phi = gen_lda2(T, W, D, N, alpha)
    write_pb(data_dir, idx, W, T, D, alpha, beta, B)
    write_txt(data_dir, idx, B)

def data_gen_params():
    W = 25
    L = int(np.sqrt(W))
    params_plp_15 = {
        'seed'      : 1234,     # rng seed
        'n_exp'     : 10,       # number of experiments
        'W'         : W,        # vocabulary size 
        'L'         : L,        # image size
        'T'         : 2*L,      # n. of topics 
        'D'         : 1000,     # n. of documents
        'N'         : 100,      # n. of words per document
        'alpha'     : 1.,       # hyper-param for mixture of topics (theta)
        'beta'      : 1.        # hyper-param for topic distribs (phi)
                                # used only as param in pb
    }
    
    params_ijar_16 = {
        'seed'      : 1234,     # rng seed
        'n_exp'     : 10,       # number of experiments
        'W'         : W,        # vocabulary size 
        'L'         : L,        # image size
        'T'         : 2*L,      # n. of topics 
        'D'         : 100,     # n. of documents
        'N'         : 100,      # n. of words per document
        'alpha'     : 1.,       # hyper-param for mixture of topics (theta)
        'beta'      : 1.        # hyper-param for topic distribs (phi)
                                # used only as param in pb
    }
    
    params = params_ijar_16
    return params

def data_gen(seed, n_exp, W, L, T, D, N, alpha, beta):
    np.random.seed(seed)
    data_dir = 'data'
    # phi is given as the horizontal and vertical topics on the 5X5 images
    phi = [np.zeros((L, L)) for i in range(T)]
    line = 0
    for phi_t in phi:
        if line >= L:
            trueLine = int(line - L)
            phi_t[:,trueLine] = 1./L*np.ones(L)
        else:
            phi_t[line] = 1./L*np.ones(L)
        line += 1
           
    rewrite_dir(data_dir)
    plot_topics(T, phi, 'data/lda_ground_phi')
    [write_pb_txt(data_dir, i, W, L, T, D, N, phi, alpha, beta) 
        for i in range(n_exp)]

def data_gen_plp15():
    data_gen(**data_gen_params())
   
def main():
    data_gen_plp15()
        
if __name__=='__main__':
    main()
