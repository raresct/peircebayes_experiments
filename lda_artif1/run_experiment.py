#!/usr/bin/env python2

import numpy as np
import subprocess
import shlex
import re
import sys
import time

from os import listdir
from os.path import isfile, join

sys.path.append("..") # for utils
from utils import call_cmd

def run_experiment():
    pattern = re.compile('lda_([0-9]+)\.pb')
    data_dir = 'data'
    files = [ (re.search(pattern, f).group(1), join(data_dir,f) )
        for f in listdir(data_dir)
        if isfile(join(data_dir,f)) and re.match(pattern, f)]
    cmd_str = 'peircebayes {} -n 100 -m lda -t -s {}'
    cmd_str2 = 'peircebayes {} -n 100 -m lda -t -s {} -a cgs'
    np.random.seed(1234)
    start = time.time()
    for i,f in files:
        print i
        # sample 10 times
        for j,seed in enumerate(np.random.choice(5000, 10, replace=False)+1):
            call_cmd(cmd_str.format(f, seed))
            phi = np.load('/tmp/peircebayes/avg_samples.npz')['arr_1']
            np.savez(join(data_dir, 'phi_{}_{}'.format(i,j)), **{'phi':phi})
            call_cmd('cp /tmp/peircebayes/lls.npz data/lls_{}_{}.npz'.format(i,j))
            call_cmd(cmd_str2.format(f, seed))
            call_cmd('cp /tmp/peircebayes/lls.npz data/lls_cgs_{}_{}.npz'.format(i,j))
    end = time.time()
    with open('data/time_pb', 'w') as f:
        f.write(str(end-start))
    cmd_str_r = 'Rscript run_lda.R'
    start = time.time()
    call_cmd(cmd_str_r)
    end = time.time()
    with open('data/time_r', 'w') as f:
        f.write(str(end-start))
def main():
    run_experiment()

if __name__=='__main__':
    main()
