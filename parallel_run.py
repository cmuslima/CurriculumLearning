import sys
import argparse
from main import run
import matplotlib.pyplot as plt
import numpy as np
import time 
import config
import multiprocessing
num_cores = 1


def multiprocessing_func(batchsize):
    print('{} is {} '.format(batchsize, run(batchsize)))
    
if __name__ == '__main__':
    starttime = time.time()
    pool = multiprocessing.Pool()
    
    batchsize = [64] #[1.25]#0, 1, 1.5, 1.75]
   
    pool.map(multiprocessing_func, batchsize)
    pool.close()
    print()
    print('Time taken = {} seconds'.format(time.time() - starttime))