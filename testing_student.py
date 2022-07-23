
from dqn_student import DQNAgent
import student_env_setup
from evaluation import train_evaluate_protocol
from env2 import basic_grids
import numpy as np


env = basic_grids('empty_grid', 7, 7)

goals = [np.array([1,1]),np.array([2,1]),np.array([3,2]),np.array([4,2]),np.array([4,3]),np.array([5,4])]
# start_state = np.array([0,0])   
# goal = goals[4]
# state = env.reset(goal, start_state)  
# print('start', state)
# grid_state = start_state

testing_goals = [np.array([1,1]), np.array([3,1])]
#testing_goals = [np.array([1,1]), np.array([2,2]), np.array([3,1]), np.array([3,3]), np.array([4,2]), np.array([4,4]), np.array([5,4])]
testing_goals = [np.array([5,4])]
holdout_set = [np.array([1,2]), np.array([2,1]),np.array([3,2]), np.array([5,3]), np.array([4,3]), np.array([4,1])]
eps = 1
total_time_steps = 0

batchsizes = [32,64,128]
lrs = [.01, .001, .0001]
all_wins_from_curriculum = dict()
all_wins_no_curriculum = dict()
for b in batchsizes:
    for lr in lrs:
        # print(b, lr)
        # all_wins = []
        # agent = DQNAgent(b, lr)
        # for idx, g in enumerate(testing_goals):
        #     start_state = np.array([0,0])   
        #     goal = g
        #     print('goal', g)
        #     tot_wins =0
        #     eps = .5
        #     for i in range(50):
        #         state = env.reset(goal, start_state)  
                
        #         grid_state = start_state
        #         for j in range(100):
        #             #print(j)
        #             action = agent.act(state, goal, eps = eps)
        #             #print('action', action)
        #             next_state, grid_next_state, reward, done = env.step(grid_state, action)
        #             agent.step(state, goal, action, reward, next_state, done)
        #             #
        #             eps*=.999
        #             eps = max(.1,eps )
        #             state = next_state
        #             grid_state = grid_next_state
        #             total_time_steps+=1
        #             if done or j == 99:
        #                 #print(f' episode = {i} finished at time step {j} with reward {reward}')
        #                 if done:
        #                     tot_wins+=1
        #                     #print(tot_wins)
        #                 break 
            
        # for idx, g in enumerate(holdout_set):
        #     start_state = np.array([0,0])   
        #     goal = g
        #     print('goal', g)
        #     tot_wins =0
        #     eps = .1
        #     for i in range(50):
        #         state = env.reset(goal, start_state)  
                
        #         grid_state = start_state
        #         for j in range(100):
        #             #print(j)
        #             action = agent.act(state, goal, eps = eps)
        #             #print('action', action)
        #             next_state, grid_next_state, reward, done = env.step(grid_state, action)                
        #             state = next_state
        #             grid_state = grid_next_state
        #             total_time_steps+=1
        #             if done or j == 99:
        #                 #print(f' episode = {i} finished at time step {j} with reward {reward}')
        #                 if done:
        #                     tot_wins+=1
        #                     #print(tot_wins)
        #                 break 
        #     all_wins.append(tot_wins)
        # all_wins_from_curriculum[(lr, b)] = all_wins

        # print('number of wins with curriculum', all_wins_from_curriculum)

        for idx, g in enumerate(testing_goals):
            agent = DQNAgent(b, lr)
            start_state = np.array([0,0])   
            goal = g
            
            tot_wins =0
            eps = .5
            for i in range(50):
                state = env.reset(goal, start_state)  
                
                grid_state = start_state
                for j in range(100):
                    #print(j)
                    action = agent.act(state, goal, eps = eps)
                    #print('action', action)
                    next_state, grid_next_state, reward, done = env.step(grid_state, action)
                    agent.step(state, goal, action, reward, next_state, done)
                    #if idx ==5:
                        #print(f'state = {grid_state}, action = {action}, reward = {reward}, next state = {grid_next_state}')
                        #print(f'next real state', next_state)
                        #print(eps)
                    eps*=.999
                    eps = max(.1,eps )
                    state = next_state
                    grid_state = grid_next_state
                    total_time_steps+=1
                    if done or j == 99:
                        #print(f' episode = {i} finished at time step {j} with reward {reward}')
                        if done:
                            tot_wins+=1
                            #print(tot_wins)
                        break 
        all_wins = []
        for idx, g in enumerate(holdout_set):
            start_state = np.array([0,0])   
            goal = g
            print('goal', g)
            tot_wins =0
            eps = .1
            for i in range(50):
                state = env.reset(goal, start_state)  
                
                grid_state = start_state
                for j in range(100):
                    #print(j)
                    action = agent.act(state, goal, eps = eps)
                    #print('action', action)
                    next_state, grid_next_state, reward, done = env.step(grid_state, action)                
                    state = next_state
                    grid_state = grid_next_state
                    total_time_steps+=1
                    if done or j == 99:
                        #print(f' episode = {i} finished at time step {j} with reward {reward}')
                        if done:
                            tot_wins+=1
                            #print(tot_wins)
                        break 
            all_wins.append(tot_wins)
        
        all_wins_no_curriculum[(lr, b)] = all_wins


        print('number of wins without curriculum', all_wins_no_curriculum)

