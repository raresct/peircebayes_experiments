#!/usr/bin/env python2

import numpy as np
import subprocess
import shlex
import re
import sys
import time
import pickle
import ast

from os import listdir
from os.path import isfile, join

sys.path.append("..") # for utils
from utils import call_cmd

def run_experiment():
    data_dir = 'data'
    np.random.seed(1234)
    
    T = 10
    counts = pickle.load(open(join(data_dir, 'counts.pkl'), 'r'))
    
    times = []
    metrics = []
    for j,seed in enumerate(np.random.choice(5000, 10, replace=False)+1):
        print(j)
        time_pb, metric_pb = do_pb(data_dir, j, seed, T, counts)
        #print time_pb
        #print metric_pb
        time_stan, metric_stan = do_stan(data_dir, j, seed, T, counts)
        #print time_stan
        #print metric_stan
        time_prism, metric_prism = do_prism(data_dir, j, seed, T, counts)
        #print time_prism
        #print metric_prism

        time_tm_gibbs, metric_tm_gibbs = do_tm_gibbs(data_dir, j, seed, T, counts)
        #print time_tm_gibbs
        #print metric_tm_gibbs

        time_tm_vem, metric_tm_vem = do_tm_vem(data_dir, j, seed, T, counts)
        #print time_tm_vem
        #print metric_tm_vem

        times.append((time_pb, time_stan, time_prism, time_tm_gibbs, time_tm_vem))
        metrics.append((metric_pb, metric_stan, metric_prism, metric_tm_gibbs, metric_tm_vem))

    pickle.dump((times, metrics), open(join(data_dir, 'run_result.pkl'), 'w'))
    #with open('data/times', 'w') as f:
    #    f.write('pb:\t{}\t{}\n'.format(np.average(times_pb),np.std(times_pb)))
    #    f.write('stan:\t{}\t{}\n'.format(np.average(times_stan)))

def ull(mu, phi, T, counts):
    ll = 0
    for (d, doc) in enumerate(counts):
        for (w, count) in enumerate(doc):
            if count>0:
                ll_w = 0
                for t in range(T):
                    ll_w += mu[d,t]*phi[t,w]
                ll += count*np.log(ll_w)     
    return ll

def do_pb(data_dir, j, seed, T, counts):
    print 'Doing PB.'
    pb_file = 'lda_0.pb'
    # change this if you're not me
    path_to_pb_out = '/home/rares/p/peircebayes_experiments/lda_ull_time/data/peircebayes'
    cmd_str  = 'peircebayes {} -n {} -m lda -d -t -s {} -o '+path_to_pb_out

    time_pb = []
    metric_pb = []

    for i in [50,100,150,200]:
        print '\t{}'.format(i)

        start = time.time()
        call_cmd(cmd_str.format(join(data_dir, pb_file), i, seed))
        time_pb.append(time.time()-start)

        # get mu and phi
        theta = np.load(join(path_to_pb_out, 'avg_samples.npz'))
        mu = theta['arr_0']
        phi = theta['arr_1']
        
        # call ull
        ll = ull(mu, phi, T, counts)
        metric_pb.append(ll)
    return time_pb, metric_pb

def do_stan(data_dir, j, seed, T, counts):
    print 'Doing Stan.'
    time_stan = []
    metric_stan = []
    for i in [5,10,15,20]:
        print '\t{}'.format(i)
        stan_file = join(data_dir, 'lda_stan_kwargs.py')
        out_file = join(data_dir, 'stan_ll_{}.npz'.format(j))
        phi_file = join(data_dir, 'stan_phi_{}.npz'.format(j))
        theta_file = join(data_dir, 'stan_theta_{}.npz'.format(j))
        with open(join(data_dir, 'lda_stan_0.py'), 'r') as fin, open(
                stan_file, 'w') as fout:
            fout.write("kwargs = {{'seed': {}, 'iter':{} }}\n".format(seed, i))
            fout.write("out_file = '{}'\n".format(out_file))
            fout.write("phi_file = '{}'\n".format(phi_file))
            fout.write("theta_file = '{}'\n".format(theta_file))
            fout.write(fin.read())

        start = time.time()
        call_cmd('python3 {}'.format(stan_file))
        time_stan.append(time.time()-start)

        mu = np.load(theta_file)['arr_0']
        phi = np.load(phi_file)['arr_0']
        ll = ull(mu, phi, T, counts)
        metric_stan.append(ll)
    return time_stan, metric_stan

def do_prism(data_dir, j, seed, T, counts):
    print 'Doing Prism.'

    prism_file = join(data_dir, 'lda_prism_0.psm')
    in_prism = join(data_dir, 'in_prism_{}'.format(j))
    phi_prism = join(data_dir, 'phi_prism_{}'.format(j))
    theta_prism = join(data_dir, 'theta_prism_{}'.format(j))
    vfe_prism = join(data_dir, 'vfe_prism_{}'.format(j))

    time_prism = []
    metric_prism = []

    for i in [50,100,150,200]:
        print '\t{}'.format(i)

        with open(prism_file, 'r') as fin, open(in_prism, 'w') as fout:
            fout.write(fin.read())
            fout.write('''
prism_main :-
    random_set_seed({}),
    set_prism_flag(learn_mode,both),
    set_prism_flag(max_iterate,{}),
    go,
    learn_statistics(free_energy, V),
    open('{}',write, Stream), write(Stream,V), close(Stream),
    save_phi,
    save_theta.

save_phi :-
    findall((V,Param), get_sw(phi(V), [_,_,Param]), Params),
    open('{}',write, Stream), forall(member((V,Param), Params), (write(Stream,V), write(Stream,'|'), write(Stream,Param), nl(Stream))), close(Stream).

save_theta :-
    findall((V,Param), get_sw(theta(V), [_,_,Param]), Params),
    open('{}',write, Stream), forall(member((V,Param), Params), (write(Stream,V), write(Stream,'|'), write(Stream,Param), nl(Stream))), close(Stream).
'''.format(seed, i, vfe_prism, phi_prism, theta_prism))

        start = time.time()
        call_cmd('upprism {}'.format(in_prism))
        time_prism.append(time.time()-start)
        
        with open(phi_prism, 'r') as fin:
            phi_idx = []
            for line in fin:
                idx, rest = line.strip().split('|')
                idx = int(idx)-1
                topic = np.array(ast.literal_eval(rest))
                phi_idx.append((idx, topic))
            phi = [0 for i in range(len(phi_idx))]
            for idx, topic in phi_idx:
                phi[idx] = topic
            phi = np.array(phi)    
        
        with open(theta_prism, 'r') as fin:
            theta_idx = []
            for line in fin:
                idx, rest = line.strip().split('|')
                idx = int(idx)-1
                doc_mix = np.array(ast.literal_eval(rest))
                theta_idx.append((idx, doc_mix))
            mu = [0 for i in range(len(theta_idx))]
            for idx, doc_mix in theta_idx:
                mu[idx] = doc_mix
            mu = np.array(mu)
        
        ll = ull(mu, phi, T, counts)
        metric_prism.append(ll)
    return time_prism, metric_prism

def do_tm_gibbs(data_dir, j, seed, T, counts):
    print 'Doing topicmodels-Gibbs.'

    gibbs_file = join(data_dir, 'gibbs_lda.R')
    final_gibbs = join(data_dir, 'gibbs_lda_{}.R'.format(j))
    gibbs_theta = join(data_dir, 'gibbs_theta_{}.csv'.format(j))
    gibbs_phi = join(data_dir, 'gibbs_phi_{}.csv'.format(j))
    
    time_gibbs = []
    metric_gibbs = []

    for i in [50,100,150,200]:
        print '\t{}'.format(i)

        with open(gibbs_file, 'r') as fin, open(final_gibbs, 'w') as fout:
            fout.write('''
iter = {}
seed = {}
            '''.format(i, seed))
            fout.write(fin.read())
            fout.write('''
gibbs_models = lapply(list.files("{}", pattern='lda_0.txt', full.names=T), run_gibbs)
write.table(gibbs_models[[1]][[1]]@gamma, "{}", sep=" ", row.names=F, col.names=F)
write.table(exp(gibbs_models[[1]][[1]]@beta), "{}", sep=" ", row.names=F, col.names=F)
'''.format(data_dir, gibbs_theta, gibbs_phi))

        start = time.time()
        call_cmd('Rscript {}'.format(final_gibbs))
        time_gibbs.append(time.time()-start)
        
        with open(gibbs_theta, 'r') as fin:
            mu = []
            for line in fin:
                mu.append([float(i) for i in line.strip().split()])
            mu = np.array(mu)
        
        with open(gibbs_phi, 'r') as fin:
            phi = []
            for line in fin:
                phi.append([float(i) for i in line.strip().split()])
            phi = np.array(phi)
        
        ll = ull(mu, phi, T, counts)
        metric_gibbs.append(ll)

    return time_gibbs, metric_gibbs

def do_tm_vem(data_dir, j, seed, T, counts):
    print 'Doing topicmodels-VEM.'

    vem_file = join(data_dir, 'vem_lda.R')
    final_vem = join(data_dir, 'vem_lda_{}.R'.format(j))
    vem_theta = join(data_dir, 'vem_theta_{}.csv'.format(j))
    vem_phi = join(data_dir, 'vem_phi_{}.csv'.format(j))

    time_vem = []
    metric_vem = []

    for i in [20,30,40,50]:
        print '\t{}'.format(i)

        with open(vem_file, 'r') as fin, open(final_vem, 'w') as fout:
            fout.write('''
iter = {}
seed = {}
            '''.format(i, seed))
            fout.write(fin.read())
            fout.write('''
vem_models = lapply(list.files("{}", pattern='lda_0.txt', full.names=T), run_vem)
write.table(vem_models[[1]][[1]]@gamma, "{}", sep=" ", row.names=F, col.names=F)
write.table(exp(vem_models[[1]][[1]]@beta), "{}", sep=" ", row.names=F, col.names=F)
'''.format(data_dir, vem_theta, vem_phi))

        start = time.time()
        call_cmd('Rscript {}'.format(final_vem))
        time_vem.append(time.time()-start)
        
        with open(vem_theta, 'r') as fin:
            mu = []
            for line in fin:
                mu.append([float(i) for i in line.strip().split()])
            mu = np.array(mu)
        
        with open(vem_phi, 'r') as fin:
            phi = []
            for line in fin:
                phi.append([float(i) for i in line.strip().split()])
            phi = np.array(phi)
        
        ll = ull(mu, phi, T, counts)
        metric_vem.append(ll)

    return time_vem, metric_vem

def main():
    run_experiment()

if __name__=='__main__':
    main()
