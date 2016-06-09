#!/usr/bin/env python2

import subprocess
import shlex
import os
import sys
import time
sys.path.append("..") # for utils
from utils import call_cmd

def main():
    use_seeds = False
    if len(sys.argv)==2 and sys.argv[1] == 'seeds':
        use_seeds = True
    if use_seeds:
        data_dir = 'data'
        cmd_str = 'peircebayes data/news20comp_c.pb -n 400 -d -t -o $PWD/data/peircebayes'
        #call_cmd(cmd_str)
        #call_cmd('cp data/peircebayes/avg_samples.npz data/avg_samples.npz')
        #call_cmd('upprism clda.psm')

    else:
        data_dir = 'data'
        cmd_str = 'peircebayes data/news20comp.pb -n 400 -d -t -o $PWD/data/peircebayes'
        seed = 1234
        n_stan_samples = 20

        # 1 tm
        #TODO tm
        start = time.time()

        #call_cmd('Rscript data/news20_tm.R')

        tm_time = time.time()-start

        # 2 pb
        start = time.time()

        #call_cmd(cmd_str)
        #call_cmd('cp data/peircebayes/avg_samples.npz data/avg_samples.npz')

        pb_time = time.time()-start

        # 3 prism
        start = time.time()

        call_cmd('upprism data/news20_prism.psm')

        prism_time = time.time()-start

        # 4 stan
        stan_file = 'data/lda_stan.py'
        with open(os.path.join(data_dir, 'news20_stan.py'), 'r') as fin, open(
                    stan_file, 'w') as fout:
                fout.write("kwargs = {{'seed': {}, 'iter':{} }}\n".format(seed, n_stan_samples))
                fout.write(fin.read())
        start = time.time()

        #call_cmd('python3 {}'.format(stan_file))

        stan_time = time.time()-start
        with open('data/times', 'w') as fout:
            fout.write('PB: {} seconds\n'.format(pb_time))
            fout.write('Prism: {} seconds\n'.format(prism_time))
            fout.write('Stan: {} seconds\n'.format(stan_time))
            fout.write('Topicmodels: {} seconds\n'.format(tm_time))

if __name__=='__main__':
    main()
