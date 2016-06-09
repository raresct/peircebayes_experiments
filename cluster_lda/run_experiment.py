#!/usr/bin/env python2

import subprocess
import shlex

import sys
sys.path.append("..") # for utils
from utils import call_cmd

def main():
    cmd_str = 'peircebayes data/lda2_arxiv.pb -d -t -n 50 -b 100'        
    call_cmd('cat data/pb_hlda3.pb data/arxiv_obs.pb', open('data/lda2_arxiv.pb', 'w'))
    call_cmd(cmd_str)
    call_cmd('cp /tmp/peircebayes/avg_samples.npz data/avg_samples.npz')
    call_cmd('rm data/lda2_arxiv.pb')
    
if __name__=='__main__':
    main()
