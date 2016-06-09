#!/usr/bin/env python2

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.cross_validation import KFold
import sys
import pickle
import os
import itertools
import collections

sys.path.append("..") # for utils
from utils import (rewrite_dir, generate_lda, write_pb_cv,
    write_stan_cv, write_prism_cv, write_txt_cv)

def write_cv_data(K, data_dir, idx, W, L, T, D, N, phi, alpha, beta, chains):
    B = generate_lda(T, W, D, N, phi, alpha)
    # split cv data
    B_sparse = csr_matrix(B)
    Bs = [dok_matrix((D, W), dtype=np.float32) for k in range(K)]
    test_counts = [dok_matrix((D, W), dtype=np.float32) for k in range(K)]
    for d in range(B_sparse.shape[0]):
        crow = B_sparse[d,:].tocoo()
        list_of_tokens = []
        for term_idx,count in itertools.izip(crow.col, crow.data):
            list_of_tokens += [term_idx]*count
        list_of_tokens = list(np.random.permutation(np.array(list_of_tokens)))
        kf = KFold(len(list_of_tokens), n_folds=K)
        for k,(train, test) in enumerate(kf):
            l = [list_of_tokens[i] for i in train]
            dict_of_counts = collections.Counter(l)
            for w,count in dict_of_counts.iteritems():
                Bs[k][d,w] = count
            l = [list_of_tokens[i] for i in test]
            dict_of_counts = collections.Counter(l)
            for w,count in dict_of_counts.iteritems():
                test_counts[k][d,w] = count
    Bs = [csr_matrix(i) for i in Bs]
    test_counts = [csr_matrix(i) for i in test_counts]
    for i,counts in enumerate(test_counts):
        pickle.dump(counts,
            open(os.path.join(data_dir, 'counts_{}.pkl'.format(i)), 'w'))
    write_pb_cv(data_dir, idx, W, T, D, alpha, beta, Bs, write_params=False)
    write_stan_cv(data_dir, idx, W, T, D, alpha, beta, Bs, chains=chains, write_params=False)
    write_prism_cv(data_dir, idx, W, T, D, alpha, beta, Bs, write_params=False)
    write_txt_cv(data_dir, idx, Bs, T, alpha, beta, write_params=False)

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
    D           = 1000 # documents
    N           = 100 # words per document
    alpha       = 1.  # hyper-param for mixture of topics (theta)
    beta        = 1.  # hyper-param for topic distribs (phi),
                      # used only as param in pb

    # stan
    chains = 1

    # CV
    K = 5 # folds

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
    [write_cv_data(K, data_dir, i, W, L, T, D, N, phi, alpha, beta, chains)
        for i in range(n_exp)]

def main():
    data_gen()

if __name__=='__main__':
    main()
