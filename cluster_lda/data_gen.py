#!/usr/bin/env python2

import numpy as np
import re
import os
import sys
import pickle
import itertools

from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("..") # for utils
from utils import rewrite_dir

def get_counts_cat(pref, cat):
    pattern = re.compile('.+\.txt')
    data_dir = join(pref,cat)
    files = [ join(data_dir,f) 
        for f in listdir(data_dir)
        if isfile(join(data_dir,f)) and re.match(pattern, f)]
    docs = []
    sub_cats = []
    for fname in files:
        with open(fname, 'r') as f:
            for line in f:
                fields = line.split('\t')
                tokens = [t for t in ' '.join(fields[2:]).strip().split(' ') 
                    if len(t)>1]
                docs.append(' '.join(tokens))
                sub_cats.append(fields[1].split(' '))
    return docs, sub_cats, len(docs)

def get_counts(pref):
    ns = []
    all_docs = []
    all_sub_cats = []
    for cat in ['q-fin', 'stat', 'q-bio', 'cs', 'physics']:
        docs, sub_cats, n_docs = get_counts_cat(pref, cat)
        ns.append(n_docs)
        all_docs += docs
        all_sub_cats += sub_cats       
    vectorizer = CountVectorizer(stop_words='english')
    counts = vectorizer.fit_transform(all_docs)
    with open(join(pref, 'arxiv_info.txt'), 'w') as fout:
        #fout.write('Size of dataset before pre-processing:' +
        #    ' {} documents.\n'.format(len(data0)))
        fout.write('Size of dataset after pre-processing: {} documents\n'.format(
            counts.shape[0]))
        fout.write('Vocabulary: {} tokens\n'.format(counts.shape[1]))
        fout.write('Average document length: {} tokens\n'.format(
            np.average(np.sum(counts.toarray(), axis=1))))
        fout.write('Number of tokens: {} tokens\n'.format(
            np.sum(counts.toarray())))    
    pickle.dump(vectorizer.vocabulary_, open(join(pref, 'vocab.pkl'), 'w'))
    pickle.dump(all_sub_cats, open(join(pref, 'sub_cats.pkl'), 'w'))
    pickle.dump(ns, open(join(pref, 'ns.pkl'), 'w'))
    return counts,vectorizer

def write_pb(counts, vect, lda_f, obs_f, hlda_f, lda2_f, T, T2, alpha, alpha2, beta):
    D = counts.shape[0]
    V = counts.shape[1]
    with open(lda_f, 'w') as fout:
        fout.write('''
% LDA arxiv

% needed to ground constraints
:- enforce_labeling(true).

''')
        fout.write('''
% prob distribs
pb_dirichlet({}, theta, {}, {}).
pb_dirichlet({}, phi, {}, {}).

% plate
pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [Topic in 1..{}, theta(Topic,Doc), phi(Token,Topic)]
).

'''.format(alpha, T, D, beta, V, T, T))
    with open(lda2_f, 'w') as fout:
        fout.write('''
% LDA2 arxiv

% needed to ground constraints
:- enforce_labeling(true).

''')
        fout.write('''
% prob distribs
pb_dirichlet({}, theta, {}, {}).
pb_dirichlet({}, phi, {}, {}).

% plate
pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [Topic in 1..{}, theta(Topic,Doc), phi(Token,Topic)]
).

'''.format(alpha2, T2, D, beta, V, T2, T2))
    C = int(np.sqrt(T))
    alpha2 = 50./float(C)
    with open(hlda_f, 'w') as fout:
        fout.write('''
% HLDA arxiv

% needed to ground constraints
:- enforce_labeling(true).

''')
        fout.write('''
% prob distribs
pb_dirichlet({}, theta0, {}, {}).
pb_dirichlet({}, theta1, {}, {}).
pb_dirichlet({}, phi, {}, {}).

% plate
pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [Cluster in 1..{}, theta0(Cluster, Doc), TopicMin is (Cluster-1)*{}+1, TopicMax is TopicMin+{}, Topic in TopicMin..TopicMax, theta1(Topic,Doc), phi(Token,Topic)]
).

'''.format(alpha2, C, D, alpha, T, D, beta, V, T, C, C, C-1))
    with open(obs_f, 'w') as fout:
        for d in range(counts.shape[0]):
            out_str = 'observe(d({}), ['.format(d+1)
            list_str = []
            crow = counts[d,:].tocoo()
            for term_idx,count in itertools.izip(crow.col, crow.data):
                list_str.append('(w({}),{})'.format(term_idx+1, count))
            out_str += ','.join(list_str)
            out_str += ']).\n'
            fout.write(out_str)
    print 'Done writing pb.'

def data_gen():
    pref = 'data'
    pb_obs = join(pref, 'arxiv_obs.pb')
    pb_lda = join(pref, 'pb_lda.pb')
    pb_hlda = join(pref, 'pb_hlda.pb')
    pb_lda2 = join(pref, 'pb_lda2.pb')
    T = 25 # number of topics
    T2 = 5
    alpha = 50./float(T)
    alpha2 = 50./float(T2)
    beta = 0.1
    
    #rewrite_dir(pref)
    counts,vect = get_counts(pref)
    #write_plda(counts, vect, plda_fstr)
    write_pb(counts, vect, pb_lda, pb_obs, pb_hlda, pb_lda2, T, T2, alpha, alpha2, beta)
    #write_pb_constraints(counts, vect, pb_c_fstr, T, alpha, beta)


def main():
    data_gen()

if __name__=='__main__':
    main()

