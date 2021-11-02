import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
from student_agent import agent
from evaluation import evaluate_task, train
import numpy as np
from teacher_agent import DQNAgent
from global_vars import env_variables

import torch
import env


subdir = 'teacher-checkpoints'

def CL_loop(seed, args):

   
    env_config = env_variables(args)
    eps = env_config.eps_start 
    return_list = [] # this records the teacher returns 
    teacher_scores = [] #this records the first student episode for which the student succeeds on the target task
    
    teacher_agent = DQNAgent(state_size=env_config.state_size, action_size = env_config.teacher_action_size, seed=seed, action_list= env_config.teacher_action_list, LR = args.lr, BATCHSIZE = args.batchsize) 
  
    print(f'configration: environment = {args.env}, reward function type: {args.reward_function}, number of tasks = {env_config.teacher_action_size}, alpha = {args.alpha}, batch size = {args.batchsize}, step size = {args.lr} ')
    
    for i_episode in range(1, args.teacher_episodes+1):
        print(f'teacher episode {i_episode}')
        teacher_score = 1 #counter to keep track of the first student episode for which the student succeeds on the target task
        teacher_return = 0
        scores = list()
        teacher_action_list = list()


        

        #initalizing the student agent 
        mystudentagent = agent(env_config.rows, env_config.columns, env_config.student_num_actions, .5, .99)
        
        if args.debug:
            print('initalized the student agent')
            print(f'env has {env_config.rows} rows and {env_config.columns} columns')
            print(f'there are {env_config.student_num_actions} actions')
    
        env = env_config.live_env
        env_config.initalize_ALP_dict()
        mystudentagent.initalize_q_matrix() 
       
                                                                       
        if args.easy_initialization: # easy_initialization sets the first task to an easy one
            if args.env == 'maze':
                teacher_action_int = 7
            elif args.env == 'four_rooms':
                teacher_action_int = 2
        else:
            teacher_action_int = random.randint(0, env_config.teacher_action_size-1) 
        
        teacher_action = env_config.teacher_action_list[teacher_action_int] #teacher_action is the array verson of the teacher_action_int
        
        

        first_score, last_score = train(mystudentagent, env, args.num_training_episodes, teacher_action, args)  
                                                                              
        print('completed first training session')


        #In order to calculate the LP, we need to get the first score
        # I could just have it be 0 by default, but instead I use the first score of the training loop  
        if str(teacher_action) not in list(env_config.returns_dict.keys()):
            update_entry = {str(teacher_action): first_score}
            env_config.returns_dict.update(update_entry)
            print(f'teacher action {teacher_action} is not in the returns dict, adding now with the first score {first_score}')
            print(env_config.returns_dict)

        source_task_score, _ = evaluate_task(mystudentagent, env, teacher_action, args)
        
        target_task_score, _ = evaluate_task(mystudentagent, env, env_config.live_env.start_state,args)
        
        if args.debug:
            print(f'source task score {source_task_score}')
            print(f'target_task_score {target_task_score}')
    
        ALP = env_config.get_ALP(source_task_score, teacher_action, args)
        
        #this updates the corresponding data structs with their values: Return + LP
        update_entry = {str(teacher_action): source_task_score}
        env_config.returns_dict.update(update_entry)
        env_config.update_ALP_dict(teacher_action, ALP)
        
        if args.debug:
            print(f'updating returns dict with new value')
            print(f'updated ALP dict {env_config.ALP_dict}')

        reward, SF , target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, args)
        
        if args.debug:
            print(f'teacher action: {teacher_action_int}. Raw ALP = {ALP}, Updated LP = {args.alpha*ALP } SF = {SF}, reward = {reward}')


        teacher_action_list.append(teacher_action_int)
        

        traj = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, 0, ALP*args.alpha, args)


        print('first teacher action', teacher_action)
        print('first traj', traj)
        #print('\n')
        for j_episode in range(1, args.student_episodes+1):
            teacher_score+=1
            print('student episode', j_episode)
            


            teacher_action = env_config.get_teacher_action(teacher_agent, traj, eps) #returns raw teacher action
            teacher_action_int = env_config.convert_teacher_action(teacher_action)
            
            if args.debug:
                print(f"raw teacher action {teacher_action}, indexed teacher action {teacher_action_int}")

            teacher_action_list.append(teacher_action_int)
            

            first_score, last_score = train(mystudentagent, env, args.num_training_episodes, teacher_action, args) #this trains the agent for 10 episodes 
                                                                                    #on the particular env then returns the score/# of time steps from the last episode
                                                                                    #to do: probably better to do an avg of some number of episodes

            if str(teacher_action) not in list(env_config.returns_dict.keys()):
                update_entry = {str(teacher_action): first_score}
                env_config.returns_dict.update(update_entry)
                
                if args.debug:
                    print(f'teacher action {teacher_action} is not in the returns dict, adding now with the first score {first_score}')

            
            source_task_score, _ = evaluate_task(mystudentagent, env, teacher_action, args)
            target_task_score, _ = evaluate_task(mystudentagent, env, env_config.live_env.start_state, args)
            if args.debug:
                print(f'source task score {source_task_score}')
                print(f'target_task_score {target_task_score}')

            ALP = env_config.get_ALP(source_task_score, teacher_action, args)
           
            update_entry = {str(teacher_action): source_task_score}
            env_config.returns_dict.update(update_entry)
            

            #updating the returns with the raw learning progress rewards.. 
            env_config.update_ALP_dict(teacher_action, ALP)
            
            
            reward, SF, target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, args)
                
            if args.debug:
                print(f'updated ALP dict = {env_config.ALP_dict}')
                print(f'teacher action: {teacher_action_int}. Raw ALP = {ALP}, Updated ALP = {args.alpha*ALP}, SF = {SF}, reward = {reward}')

        
            
            traj_prime = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, j_episode, ALP*args.alpha, args)
            if args.debug:
                print(f'traj prime = {traj_prime}')
            done, success_bonus = env_config.find_termination(target_task_score, j_episode, args.student_episodes)
            
            teacher_agent.step(traj, teacher_action_int, reward, traj_prime, done)
                

            # print('\n')
            eps = max(env_config.eps_end, env_config.eps_decay*eps) # decrease epsilon
        
            traj = traj_prime

            scores.append(target_task_score) #I want to collect data on the target task.. to see if over time, it approves on the target task.
            teacher_return+=reward
            
            if done:
                print(f'teacher return = {teacher_return}')
                teacher_scores.append(teacher_score)
                print(f'teacher score = {teacher_score}')
                print(f'teacher curriculum = {teacher_action_list}')
                print(f'student scores on target task = {scores}')
                
                break
        
            print('\n')
        print('end of episode')
 
        return_list.append(teacher_return)
        #print('\n')
        if i_episode % 10 == 0:
            model_name = f'{args.rootdir}/{args.experiment_folder}/{subdir}/teacher_agent_checkpoint_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{seed}_{args.alpha}.pth'


            torch.save(teacher_agent.qnetwork_local.state_dict(), model_name)
            print(f'configration: alpha = {args.alpha}, batch size = {args.batchsize}, step size = {args.lr} ')
            print(f'scores: {scores}, return {teacher_return}, task list {teacher_action_list}')

    

    
    print(np.mean(return_list), np.std(return_list))


    return return_list, teacher_scores
            
