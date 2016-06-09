#!/usr/bin/env python2   

from data_gen import data_gen
from post_process import post_process

def run_experiments():
    pass

def main():
    data_gen()
    run_experiments()
    post_process()

if __name__=="__main__":
    main()
