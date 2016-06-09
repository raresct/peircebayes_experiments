#!/usr/bin/env python2

import numpy as np
import pickle
import os
import itertools
import nltk
import re
import itertools
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import sys

sys.path.append("..") # for utils
from utils import rewrite_dir

def can_be_noun(test_word):
    synsets = nltk.corpus.wordnet.synsets(test_word)
    if len(synsets) == 0:
        return True
    for s in synsets:
        if s.pos == 'n':
            return True
    return False

def get_counts(pref):
    cats = [ 'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
    news_all = fetch_20newsgroups(subset='all',
        remove=('headers', 'footers', 'quotes'), categories=cats)
    data0 = news_all.data
    data1 = [d for d in data0 if d and not d.isspace()]
    # lowercase, lemmatize and remove stop words

    tokens_l = [nltk.word_tokenize(d) for d in data1]
    wnl = WordNetLemmatizer()
    stop = nltk.corpus.stopwords.words('english')
    pattern = re.compile('^[a-zA-Z]+$')
    data2 = [[t.strip().lower() for t in tokens
            if t.strip() not in stop and re.match(pattern, t.strip()) and len(t)>1]
        for tokens in tokens_l]
    data3 = []
    for tokens in data2:
        doc = ' '.join([wnl.lemmatize(t) for t in tokens ])
        if doc and not d.isspace():
            data3.append(doc)

    vectorizer = CountVectorizer(stop_words=None)
    counts = vectorizer.fit_transform(data3)
    with open(os.path.join(pref, 'news20_info.txt'), 'w') as fout:
        fout.write('Size of dataset before pre-processing:' +
            ' {} documents.\n'.format(len(data0)))
        fout.write('Size of dataset after pre-processing: {} documents\n'.format(
            counts.shape[0]))
        fout.write('Vocabulary: {} words\n'.format(counts.shape[1]))
        fout.write('Average document length: {} words\n'.format(
            np.average(np.sum(counts.toarray(), axis=1))))
    pickle.dump(vectorizer.vocabulary_, open(os.path.join(pref, 'vocab.pkl'),
        'w'))
    pickle.dump(counts, open(os.path.join(pref, 'counts.pkl'),
        'w'))
    return counts,vectorizer

def write_plda(counts, vect, file_str):
    with open(file_str, 'w') as fout:
        for d in range(counts.shape[0]):
            fout.write(' '.join(
                ['{} {}'.format(term.decode('ascii', 'ignore'), counts[d,i])
                for term,i in vect.vocabulary_.iteritems() if counts[d,i]>0])
                +'\n')
    print 'Done writing plda.'

def write_pb_counts(counts, vect, file_str, T, alpha, beta):
    D = counts.shape[0]
    V = counts.shape[1]
    with open(file_str, 'w') as fout:
        fout.write('''
% LDA newscomp

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

def write_pb_counts_constraints(counts, vect, file_str, T, alpha, beta):
    D = counts.shape[0]
    V = counts.shape[1]
    with open(file_str, 'w') as fout:
        fout.write('''
% LDA newscomp + constraints

% needed to ground constraints
:- enforce_labeling(true).

''')
        # write seed constraints
        words_t1 = ['hardware', 'machine', 'memory', 'cpu']
        words_t2 = ['software', 'program', 'version', 'shareware']
        str_t1 = '\n'.join(['seed({}, [1]).'.format(vect.vocabulary_[word]+1)
            for word in words_t1])
        str_t2 = '\n'.join(['seed({}, [2]).'.format(vect.vocabulary_[word]+1)
            for word in words_t2])
        fout.write('''
seed_naf(Token) :- seed(Token, _).

{}

{}
        '''.format(str_t1, str_t2))
        fout.write('''
% prob distribs
pb_dirichlet({}, theta, {}, {}).
pb_dirichlet({}, phi, {}, {}).

% plate
pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList),
        \+ seed_naf(Token)],
    Count,
    [Topic in 1..{}, theta(Topic,Doc), phi(Token,Topic)]
).

pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList),
        seed_naf(Token)],
    Count,
    [seed(Token, TopicList), member(Topic, TopicList),
        theta(Topic,Doc), phi(Token,Topic)]
).

'''.format(alpha, T, D, beta, V, T, T))
        for d in range(counts.shape[0]):
            out_str = 'observe(d({}), ['.format(d+1)
            list_str = []
            crow = counts[d,:].tocoo()
            for term_idx,count in itertools.izip(crow.col, crow.data):
                list_str.append('(w({}),{})'.format(term_idx+1, count))
            out_str += ','.join(list_str)
            out_str += ']).\n'
            fout.write(out_str)
    print 'Done writing pb constraints.'

def write_stan_counts(counts, vect, file_str, T, alpha, beta, chains=1):
    data_dir = 'data'
    model_file = os.path.join(data_dir, 'lda.stan')
    with open(model_file, 'w') as fout:
        fout.write('''
data {
    int<lower=2> K; // num topics
    int<lower=2> V; // num words
    int<lower=1> M; // num docs
    int<lower=1> N; // total word instances
    int<lower=1,upper=V> w[N]; // word n
    int<lower=1,upper=M> doc[N]; // doc ID for word n
    vector<lower=0>[K] alpha; // topic prior
    vector<lower=0>[V] beta; // word prior
}

parameters {
    simplex[K] theta[M]; // topic dist for doc m
    simplex[V] phi[K]; // word dist for topic k
}

model {
    for (m in 1:M)
        theta[m] ~ dirichlet(alpha); // prior
    for (k in 1:K)
        phi[k] ~ dirichlet(beta); // prior
    for (n in 1:N) {
        real gamma[K];
        for (k in 1:K)
            gamma[k] <- log(theta[doc[n],k]) + log(phi[k,w[n]]);
            increment_log_prob(log_sum_exp(gamma)); // likelihood
    }
}''')
    fname = os.path.join(data_dir, file_str)
    N = 0
    D = counts.shape[0]
    W = counts.shape[1]
    alpha_l = [alpha]*T
    beta_l = [beta]*W
    w = []
    doc = []

    for d in range(counts.shape[0]):
        crow = counts[d,:].tocoo()
        for term_idx,count in itertools.izip(crow.col, crow.data):
            if count>0:
                w += [term_idx+1]*count
                N += count
                doc += [d+1]*count
    w_str = '['+','.join([str(i) for i in w])+']'
    doc_str = '['+','.join([str(i) for i in doc])+']'

    with open(fname, 'w') as fout:
        fout.write('''
import pystan
import numpy as np

lda_dat = {{
    'K' : {},
    'V' : {},
    'M' : {},
    'N' : {},
    'w' : {},
    'doc' : {},
    'alpha' : {},
    'beta' : {}
}}

fit = pystan.stan(file='{}', data=lda_dat, chains={}, **kwargs)
np.savez('data/stan_ll', fit.get_logposterior()[0])
np.savez('data/stan_phi', fit['phi'][-1,:,:])
'''.format(T, W, D, int(N), w_str, doc_str, alpha_l, beta_l, model_file, chains))

    print 'Done writing stan.'

def write_prism_counts(counts, vect, fname, T, alpha, beta):
    fname = os.path.join('data', fname)
    D = counts.shape[0]
    W = counts.shape[1]
    theta_str = 'values(theta(_), [1-{}], a@{}).'.format(T,alpha)
    phi_str = 'values(phi(_), [1-{}], a@{}).'.format(W,beta)

    with open(fname, 'w') as fout:
        fout.write('''% LDA

{}

{}

generate(W, D) :-
    msw(theta(D), T),
    msw(phi(T), W).

repl(X, N, L) :-
    length(L, N),
    maplist(Y, Y = X, L).

goal_list(GL) :-
    observe(d(D),L),
    member((w(W),C), L),
    repl((W,D), C, GL).

go :-
    findall(generate(W,D), (goal_list(GL), member((W,D), GL)), Goals),
    learn_b(Goals).
prism_main :-
    set_prism_flag(learn_mode,both),
    go,
    save_phi,
    save_phi_d,
    save_theta.

save_phi :-
    findall(Param, get_sw(phi(_), [_,_,Param]), Params),
    open('data/out_lda_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

save_phi_d :-
    findall(Param, get_sw_d(phi(_), [_,_,Param]), Params),
    open('data/out_d_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

save_theta :-
    findall(Param, get_sw(theta(_), [_,_,Param]), Params),
    open('data/out_theta_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

'''.format(theta_str, phi_str))#, burn_in, lag, n_samples))
        for d in range(counts.shape[0]):
            out_str = 'observe(d({}), ['.format(d+1)
            list_str = []
            crow = counts[d,:].tocoo()
            for term_idx,count in itertools.izip(crow.col, crow.data):
                list_str.append('(w({}),{})'.format(term_idx+1, count))
            out_str += ','.join(list_str)
            out_str += ']).\n'
            fout.write(out_str)
    print('Done writing prism.')

def write_tm_files(counts, vect, fname, T, alpha, beta):
    count_file = os.path.join('data', 'counts.txt')
    with open(count_file, 'w') as fout:
        cx = counts.tocoo()
        prev_i = -1
        for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
            if not (i-1 == prev_i or i == prev_i):
                print('no words in document: {}'.format(i-1))
            prev_i = i
            fout.write('{} {} {}\n'.format(i+1, j+1, v))
    fname = os.path.join('data', fname)
    with open(fname, 'w') as fout:
        fout.write('''
# install packages if necessary
list.of.packages <- c("topicmodels", "tm", "slam")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# load packages
lapply(list.of.packages, library, character.only=T)

K = {}

run_gibbs <- function(f){{
    data = read.table(f, quote="\\\"")
    model = LDA(simple_triplet_matrix(data[,1], data[,2], data[,3]), K, method="Gibbs", control=list(
      seed = 1234,
      alpha = {},
      delta = {}, # this is beta
      keep = 1, # save log likelihood every iteration
      iter = 400,
      burnin = 0
      ))
    model
}}
model = run_gibbs("{}")
model@gamma
TODO write to file
'''.format(T, alpha, beta, count_file))
    print('Done writing tm.')

def data_gen():
    np.random.seed(1234)
    pref = 'data'
    plda_f = 'news20comp.plda'
    pb_f = 'news20comp.pb'
    pb_c_f = 'news20comp_c.pb'

    T = 20 # number of topics
    alpha = 50./float(T)
    beta = 0.01

    rewrite_dir(pref)
    plda_fstr = os.path.join(pref,plda_f)
    pb_fstr = os.path.join(pref,pb_f)
    pb_c_fstr = os.path.join(pref,pb_c_f)
    counts,vect = get_counts(pref)

    use_seeds = False
    if len(sys.argv)==2 and sys.argv[1] == 'seeds':
        use_seeds = True

    if use_seeds:
        write_pb_counts_constraints(counts, vect, pb_c_fstr, T, alpha, beta)
        write_prism_counts_constraints(counts, vect, 'news20_prism.psm', T, alpha, beta)
    else:
        write_pb_counts(counts, vect, pb_fstr, T, alpha, beta)
        write_stan_counts(counts, vect, 'news20_stan.py', T, alpha, beta, 1)
        write_prism_counts(counts, vect, 'news20_prism.psm', T, alpha, beta)
        write_tm_files(counts, vect, 'news20_tm.R', T, alpha, beta)

def main():
    data_gen()

if __name__=='__main__':
    main()

