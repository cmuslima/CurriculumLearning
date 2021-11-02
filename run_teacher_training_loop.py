
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from CL_loop import CL_loop
import pickle



def run(args):
    subdir = 'teacher-returns'
    all_teacher_returns =[]
    all_teacher_scores = []
    
   
    for seed in range(0, args.runs):
        print(f'run {seed}')
    
        teacher_return_list, teacher_scores  = CL_loop(seed, args)
        print(f'returns on run {r}, {teacher_return_list}')
        all_teacher_returns.append(teacher_return_list)
        all_teacher_scores.append(teacher_scores)
    averaged_returns = [np.mean(np.array(all_teacher_returns), axis = 0), np.std(np.array(all_teacher_returns), axis = 0)]
    average_teacher_scores = [np.mean(np.array(all_teacher_scores), axis = 0), np.std(np.array(all_teacher_scores), axis = 0)]
    
    with open(f'{args.rootdir}/{args.experiment_folder}/{subdir}/returns_list_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{args.alpha}', 'wb') as output:
        pickle.dump(averaged_returns, output)
    with open(f'{args.rootdir}/{args.experiment_folder}/{subdir}/teacher_score_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{args.alpha}', 'wb') as output:
        pickle.dump(averaged_returns, output)
