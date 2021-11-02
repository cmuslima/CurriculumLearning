
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from CL_loop import CL_loop
import pickle
import config

subdir = 'teacher-returns'
def run(batchsize):
    alpha = 1.25
    lr = .001
    all_teacher_returns =[]
    all_teacher_scores = []
    
   
    for r in range(0, config.runs):
        print(f'run {r}')
        print('BATCHSIZE', batchsize)

        teacher_return_list, teacher_scores  = CL_loop(lr, batchsize, num_student_episodes = config.student_episodes, num_teaching_episodes = config.teacher_episodes, game = config.env, SR = config.SR, seed = r, alpha = alpha)
        print(f'returns on run {r}, {teacher_return_list}')
        all_teacher_returns.append(teacher_return_list)
        all_teacher_scores.append(teacher_scores)
    averaged_returns = [np.mean(np.array(all_teacher_returns), axis = 0), np.std(np.array(all_teacher_returns), axis = 0)]
    average_teacher_scores = [np.mean(np.array(all_teacher_scores), axis = 0), np.std(np.array(all_teacher_scores), axis = 0)]
    
    with open(f'{config.rootdir}/{config.experiment_folder}/{subdir}/returns_list_{config.reward_function}_{batchsize}_{lr}_{config.SR}_{alpha}', 'wb') as output:
        pickle.dump(averaged_returns, output)
    with open(f'{config.rootdir}/{config.experiment_folder}/{subdir}/teacher_score_{config.reward_function}_{batchsize}_{lr}_{config.SR}_{alpha}', 'wb') as output:
        pickle.dump(averaged_returns, output)
