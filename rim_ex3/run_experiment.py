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
    data_dir = 'data/sushi3'
    pb_file = 'sushi3.pb'
    cmd_str  = 'peircebayes {} -n 100 -t -s {}'
    np.random.seed(1234)
    start = time.time()
    for j,seed in enumerate(np.random.choice(1000, 10, replace=False)+1):
        call_cmd(cmd_str.format(join(data_dir, pb_file), seed))
        call_cmd('cp /tmp/peircebayes/avg_samples.npz data/avg_samples_{}.npz'.format(j))
        call_cmd('cp /tmp/peircebayes/lls.npz data/lls_{}.npz'.format(j))
    end = time.time()
    with open('data/rim_time', 'w') as f:
        f.write(str(end-start))
        
def main():
    run_experiment()

if __name__=='__main__':
    main()
