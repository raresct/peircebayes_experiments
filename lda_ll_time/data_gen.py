#!/usr/bin/env python2

import numpy as np
import sys

sys.path.append("..") # for utils
from utils import rewrite_dir, generate_lda, write_pb, write_stan, write_prism, write_txt

def write_data(data_dir, idx, W, L, T, D, N, phi, alpha, beta, chains):
    B = generate_lda(T, W, D, N, phi, alpha)
    write_pb(data_dir, idx, W, T, D, alpha, beta, B)
    write_stan(data_dir, idx, W, T, D, alpha, beta, B, chains=chains)
    write_prism(data_dir, idx, W, T, D, alpha, beta, B)
    write_txt(data_dir, idx, B, T, alpha, beta)

def data_gen():
    ### IN
    ## experiment stuff
    np.random.seed(1234)
    data_dir    = 'data'
    n_exp       = 1  # number of experiments
    ## LDA stuff
    W           = 25  #  word vocabulary
    L           = int(np.sqrt(W)) # image size
    T           = 2*L # topics
    D           = 100 # documents
    N           = 100 # words per document
    alpha       = 1.  # hyper-param for mixture of topics (theta)
    beta        = 1.  # hyper-param for topic distribs (phi),
                      # used only as param in pb

    # stan
    chains = 1

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
    [write_data(data_dir, i, W, L, T, D, N, phi, alpha, beta, chains)
        for i in range(n_exp)]

def main():
    data_gen()

if __name__=='__main__':
    main()
