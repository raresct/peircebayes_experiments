#!/usr/bin/env python2

import numpy as np
import subprocess
import shlex
import re
import sys
import time
import pickle

from os import listdir
from os.path import isfile, join

sys.path.append("..") # for utils
from utils import call_cmd

def run_experiment():
    data_dir = 'data'
    np.random.seed(1234)
    times = []
    metrics = []
    topics = [6,8,10,12,14]
    alpha = 1.
    beta = 1.
    D = 1000
    W = 25
    for j,seed in enumerate(np.random.choice(5000, 10, replace=False)+1):
        print(j)
        metric_pb = do_pb(data_dir, j, seed, topics, alpha, beta, D, W)
        #print metric_pb
        metric_stan = do_stan(data_dir, j, seed, topics, alpha, beta, D, W)
        #print metric_stan
        metric_prism = do_prism(data_dir, j, seed, topics, alpha, beta, D, W)
        #print metric_prism

        metric_tm_gibbs = do_tm_gibbs(data_dir, j, seed, topics, alpha, beta, D, W)
        #print metric_tm_gibbs

        metric_tm_vem = do_tm_vem(data_dir, j, seed, topics, alpha, beta, D, W)
        #print metric_tm_vem

        #times.append((time_pb, time_stan, time_prism, time_tm_gibbs, time_tm_vem))
        metrics.append((metric_pb, metric_stan, metric_prism, metric_tm_gibbs, metric_tm_vem))

    pickle.dump((topics, metrics), open(join(data_dir, 'run_result.pkl'), 'w'))
    #with open('data/times', 'w') as f:
    #    f.write('pb:\t{}\t{}\n'.format(np.average(times_pb),np.std(times_pb)))
    #    f.write('stan:\t{}\t{}\n'.format(np.average(times_stan)))

def do_pb(data_dir, j, seed, topics, alpha, beta, D, W):
    print 'Doing PB.'
    pb_file = join(data_dir,'lda_0.pb')
    new_pb_file = join(data_dir,'new_lda_0.pb')
    # change this if you're not me
    path_to_pb_out = '/home/rares/p/peircebayes_experiments/lda_ll_topics/data/peircebayes'
    cmd_str  = 'peircebayes {} -n {} -m lda -d -t -s {} -o '+path_to_pb_out
    it = 200

    metric_pb = []

    for T in topics:
        print '\t{}'.format(T)

        with open(pb_file, 'r') as fin, open(new_pb_file, 'w') as fout:
            fout.write('''pb_dirichlet({}, mu, {}, {}).
pb_dirichlet({}, phi, {}, {}).

generate(Doc, Token) :-
    Topic in 1..{},
    mu(Topic,Doc),
    phi(Token,Topic).

'''.format(alpha, T, D, beta, W, T, T))
            fout.write(fin.read())

        start = time.time()
        call_cmd(cmd_str.format(new_pb_file, it, seed))
        #time_pb.append(time.time()-start)
        lls = np.load(join(path_to_pb_out, 'lls.npz'))['lls']
        ll = np.average(lls[-10:])
        metric_pb.append(ll)
    return metric_pb

def do_stan(data_dir, j, seed, topics, alpha, beta, D, W):
    print 'Doing Stan.'

    it = 20
    metric_stan = []


    for T in topics:
        print '\t{}'.format(T)
        stan_file = join(data_dir, 'lda_stan_kwargs.py')
        out_file = join(data_dir, 'stan_ll_{}.npz'.format(j))
        phi_file = join(data_dir, 'stan_phi_{}.npz'.format(j))
        with open(join(data_dir, 'lda_stan_0.py'), 'r') as fin, open(
                stan_file, 'w') as fout:
            fout.write("kwargs = {{'seed': {}, 'iter':{} }}\n".format(seed, it))
            fout.write("out_file = '{}'\n".format(out_file))
            fout.write("phi_file = '{}'\n".format(phi_file))
            for line in fin:
                if 'lda_dat = {' in line:
                    fout.write(line)
                    alpha_l = [alpha]*T
                    beta_l = [beta]*W
                    fout.write('''  'K' : {},
    'alpha' : {},
    'beta' : {},
                    '''.format(T, alpha_l, beta_l))
                else:
                    fout.write(line)

        start = time.time()
        call_cmd('python3 {}'.format(stan_file))
        #time_stan.append(time.time()-start)

        logpost = np.load(out_file)['arr_0'][-1]
        metric_stan.append(logpost)
    return metric_stan

def do_prism(data_dir, j, seed, topics, alpha, beta, D, W):
    print 'Doing Prism.'

    prism_file = join(data_dir, 'lda_prism_0.psm')
    in_prism = join(data_dir, 'in_prism_{}'.format(j))
    out_prism = join(data_dir, 'out_prism_{}'.format(j))
    vfe_prism = join(data_dir, 'vfe_prism_{}'.format(j))

    metric_prism = []

    for T in topics:
        print '\t{}'.format(T)

        theta_str = 'values(theta(_), [1-{}], a@{}).'.format(T,alpha)
        phi_str = 'values(phi(_), [1-{}], a@{}).'.format(W,beta)
        with open(prism_file, 'r') as fin, open(in_prism, 'w') as fout:
            fout.write('''

{}

{}
'''.format(theta_str, phi_str))
            fout.write(fin.read())
            fout.write('''
prism_main :-
    random_set_seed({}),
    set_prism_flag(learn_mode,both),
    go,
    learn_statistics(free_energy, V),
    open('{}',write, Stream), write(Stream,V), close(Stream),
    save_phi.

save_phi :-
    findall(Param, get_sw(phi(_), [_,_,Param]), Params),
    open('{}',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

'''.format(seed, vfe_prism, out_prism))

        start = time.time()
        call_cmd('upprism {}'.format(in_prism))
        #time_prism.append(time.time()-start)

        with open(vfe_prism, 'r') as fin:
            vfe = float(fin.read().strip())
        metric_prism.append(vfe)
    return metric_prism

def do_tm_gibbs(data_dir, j, seed,  topics, alpha, beta, D, W):
    print 'Doing topicmodels-Gibbs.'

    gibbs_file = join(data_dir, 'gibbs_lda.R')
    final_gibbs = join(data_dir, 'gibbs_lda_{}.R'.format(j))
    gibbs_ll = join(data_dir, 'gibbs_ll_{}.R'.format(j))
    it = 200

    metric_gibbs = []

    for T in topics:
        print '\t{}'.format(T)

        with open(gibbs_file, 'r') as fin, open(final_gibbs, 'w') as fout:
            fout.write('''
K = {}
alpha = {}
beta = {}
iter = {}
seed = {}
            '''.format(T, alpha, beta, it, seed))
            fout.write(fin.read())
            fout.write('''
gibbs_lls = lapply(list.files("{}", pattern='lda_0.txt', full.names=T), run_gibbs)
write.table(gibbs_lls, "{}", sep=" ", row.names=F, col.names=F)
'''.format(data_dir, gibbs_ll))

        start = time.time()
        call_cmd('Rscript {}'.format(final_gibbs))
        #time_gibbs.append(time.time()-start)
        with open(gibbs_ll, 'r') as fin:
            lls = [float(line.strip()) for line in fin]
        ll = np.average(lls[-10:])
        metric_gibbs.append(ll)

    return metric_gibbs

def do_tm_vem(data_dir, j, seed, topics, alpha, beta, D, W):
    print 'Doing topicmodels-VEM.'

    vem_file = join(data_dir, 'vem_lda.R')
    final_vem = join(data_dir, 'vem_lda_{}.R'.format(j))
    vem_ll = join(data_dir, 'vem_ll_{}.R'.format(j))


    metric_vem = []

    for T in topics:
        print '\t{}'.format(T)

        with open(vem_file, 'r') as fin, open(final_vem, 'w') as fout:
            fout.write('''
K = {}
alpha = {}
beta = {}
iter = {}
seed = {}
            '''.format(T, alpha, beta, 500, seed))
            fout.write(fin.read())
            fout.write('''
vem_lls = lapply(list.files("{}", pattern='lda_0.txt', full.names=T), run_vem)
write.table(vem_lls, "{}", sep=" ", row.names=F, col.names=F)
'''.format(data_dir, vem_ll))

        start = time.time()
        call_cmd('Rscript {}'.format(final_vem))
        #time_vem.append(time.time()-start)
        with open(vem_ll, 'r') as fin:
            lls = [float(line.strip()) for line in fin]
        ll = lls[-1]
        metric_vem.append(ll)

    return metric_vem

def main():
    run_experiment()

if __name__=='__main__':
    main()
