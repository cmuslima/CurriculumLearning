import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
from student_agent import agent, grid, state_encoding, evaluate_source_task, evaluate_target_task
import numpy as np
from dqn_teacher import DQNAgent




def dqn(eps_start=.01, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    eps = eps_start                    # initialize epsilon
    start_states = [np.array([0,4]), np.array([0,3]), np.array([0,2]), np.array([0,1]), np.array([0,0])] 
    mystudentagent = agent() 
    env = grid()
    mystudentagent.initalize_q_matrix()
    curriculum_returns = list()

    for ss in start_states: 
        #for each curriculum start state, train for 2 episodes
        for i in range(3):
            print('this is the start state:', ss)
            state = mystudentagent.reset(start_state = ss) # this will be given the start state from the teacher.. 

            total_rewards = 0
            num_time_steps=0

            #r = evaluate_only(mystudentagent, env)
            #print('this is the evaluation return', r)
            trajectory_prime = []
            while(True): # for k to num time steps loop
                    
                action_movement, action_index = mystudentagent.act(state, env)
                next_state, reward, done = mystudentagent.step(state, action_index, action_movement, trajectory_prime, env)
                
                total_rewards+=reward
                
                mystudentagent.learning_update(reward, next_state, state, action_index)
                            
                if done:
                    num_time_steps+=1 
                    #print("Episode finished after {} timesteps".format(num_time_steps))
                    break
                state=next_state
                num_time_steps+=1
                #end loop
            
        print('q matrix after training on start state', ss)
        print('students q_matrix at start state:', ss, mystudentagent.q_matrix)

        score_source_task = evaluate_source_task(mystudentagent, env, ss)
        print('score on source task', ss, score_source_task)
        score_target_task = evaluate_target_task(mystudentagent, env)
        print('score on target task,', score_target_task)
    #end loop
        curriculum_returns.append((ss, score_source_task,score_target_task))

    return curriculum_returns
            
    

cc = dqn()

print('cc', cc)
