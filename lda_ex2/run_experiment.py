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
    data_dir = 'data'
    pb_file = 'lda_0.pb'
    cmd_str  = 'peircebayes {} -n 300 -m lda_ll -d -t -s {}'
    np.random.seed(1234)
    times_pb = []
    times_church = []
    times_church2 = []
    for j,seed in enumerate(np.random.choice(5000, 10, replace=False)+1):
        start = time.time()
        call_cmd(cmd_str.format(join(data_dir, pb_file), seed))
        end = time.time()
        times_pb.append(end-start)
        call_cmd('cp /tmp/peircebayes/last_sample.npz data/last_sample_{}.npz'.format(j))
        call_cmd('cp /tmp/peircebayes/lls.npz data/lls_{}.npz'.format(j))
        church_str = 'church -s {} data/lda_church1_0.md'.format(seed)
        start = time.time()
        call_cmd(church_str)
        end = time.time()
        times_church.append(end-start)
        call_cmd('mv ll_church1.csv data/ll_church1_{}.csv'.format(j))
        church2_str = 'church -s {} data/lda_church2_0.md'.format(seed)
        start = time.time()
        call_cmd(church2_str)  
        end = time.time()
        times_church2.append(end-start)
        call_cmd('mv ll_church2.csv data/ll_church2_{}.csv'.format(j))
    with open('data/times', 'w') as f:
        f.write('pb: {}\n'.format(np.average(times_pb)))
        f.write('church: {}\n'.format(np.average(times_church)))
        f.write('church2: {}\n'.format(np.average(times_church2)))
def main():
    run_experiment()

if __name__=='__main__':
    main()
