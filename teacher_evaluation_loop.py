
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




def eval_loop(num_teaching_episodes, file, seed, args):
    final_done = True
    

    #initalizing environment
    env_config = env_variables(args)
    eps = 0.00
    scores = list()
    teacher_action_list = list()

    
    if args.random_curriculum == False and args.target_only == False:
        print('uploading teacher policy')
        teacher_agent = DQNAgent(state_size=env_config.state_size, action_size = env_config.teacher_action_size, seed=seed, action_list= env_config.teacher_action_list, LR = args.lr, BATCHSIZE = args.batchsize) 
        teacher_agent.qnetwork_local.load_state_dict(torch.load(file))
    
   

    for i_episode in range(1, num_teaching_episodes+1):
        teacher_score = 0
       
       
        scores = list()
        teacher_action_list = list()
    
        

        #initalizing the student agent 
        mystudentagent = agent(env_config.rows, env_config.columns, env_config.student_num_actions, .5, .99)
        
    
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
        
        
        teacher_action = env_config.teacher_action_list[teacher_action_int]
      
        first_score, last_score = train(mystudentagent, env, args.num_training_episodes, teacher_action, args) #this trains the agent for 10 episodes 
                                                                                #on the particular env then returns the score/# of time steps from the last episode
                                                                                #to do: probably better to do an avg of some number of episodes
        if args.random_curriculum == False and args.target_only == False:
            if str(teacher_action) not in list(env_config.returns_dict.keys()):
                update_entry = {str(teacher_action): first_score}
                env_config.returns_dict.update(update_entry)
            
            source_task_score,_ = evaluate_task(mystudentagent, env, teacher_action, args)
            
            target_task_score, _ = evaluate_task(mystudentagent, env, env_config.live_env.start_state, args)
            # print(f'source task score {source_task_score}, raw num steps {source_task_num_steps}')
            # print(f'target_task_score {target_task_score}, raw num steps {target_task_num_steps}')
        
            ALP = env_config.get_ALP(source_task_score, teacher_action, args)
            env_config.update_ALP_dict(teacher_action, ALP)
            update_entry = {str(teacher_action): source_task_score}
            env_config.returns_dict.update(update_entry)
            teacher_action_list.append(teacher_action_int)
            
            reward, SF , target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, args)
            
            #print(f'teacher action: {teacher_action_int}. Raw ALP = {ALP}, Updated LP = {alpha*ALP, }SF = {SF}, reward = {reward}')


        

            traj = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, 0, ALP*args.alpha, args)
            
            

    
        for j_episode in range(1, args.student_episodes+1):
           
            teacher_score+=1
          

            if args.random_curriculum and args.target_only == False:
                teacher_action_int = random.randint(0, env_config.action_size-1)
                teacher_action = env_config.action_list[teacher_action_int]

            elif args.target_only and args.random_curriculum == False:
                teacher_action_int = 0
                teacher_action = env_config.action_list[teacher_action_int]
            else:
                teacher_action = env_config.get_teacher_action(teacher_agent, traj, eps) #returns raw teacher action
                teacher_action_int = env_config.convert_teacher_action(teacher_action)
   
            teacher_action_list.append(teacher_action_int)
            
            first_score, last_score, = train(mystudentagent, env, args.num_training_episodes, teacher_action, args) #this trains the agent for 10 episodes 
                                                                                    #on the particular env then returns the score/# of time steps from the last episode
                                                                                    #to do: probably better to do an avg of some number of episodes

            if args.random_curriculum == False and args.target_only == False:
                if str(teacher_action) not in list(env_config.returns_dict.keys()):
                    update_entry = {str(teacher_action): first_score}
                    env_config.returns_dict.update(update_entry)
                
                source_task_score, _ = evaluate_task(mystudentagent, env, teacher_action, args)
        
            target_task_score, _ = evaluate_task(mystudentagent, env, env_config.live_env.start_state, args)

            if args.random_curriculum == False and args.target_only == False:
                ALP = env_config.get_ALP(source_task_score, teacher_action)
    
                update_entry = {str(teacher_action): source_task_score}
                env_config.returns_dict.update(update_entry)
            

                #updating the returns with the raw learning progress rewards.. 
                env_config.update_ALP_dict(teacher_action, ALP)

                reward, SF, target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, args)

            
                traj_prime = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, j_episode, ALP*args.alpha, args)
                traj = traj_prime

            done, success_bonus = env_config.find_termination(target_task_score, j_episode, args.student_episodes)
      
            scores.append(target_task_score) #I want to collect data on the target task.. to see if over time, it approves on the target task.
            

            if done and final_done:
                #The point of the final_done is so I can record the first student episode for which the student succeeded on the target task
                final_teacher_score = teacher_score
                final_done = False
                
            
       
       
        print('teacher actions', teacher_action_list)
        print('scores', scores)
        print('final_teacher_score', final_teacher_score)
        

    
    return np.array(scores), np.array(teacher_action_list), final_teacher_score
            
    






