#!/usr/bin/env python2   

from data_gen import data_gen_plp15
from post_process import post_process
from run_experiment import run_experiment

def main():
    data_gen_plp15()
    run_experiment()
    post_process()

if __name__=="__main__":
    main()
