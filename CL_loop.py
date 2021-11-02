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
import config

subdir = 'teacher-checkpoints'

def CL_loop(LR, BATCHSIZE,num_student_episodes,num_teaching_episodes, game, SR, seed, alpha):

    
    env_config = env_variables()
    env_config.alpha = alpha
    eps = env_config.eps_start
    teacherdata = list()               
    scoredata = list()   
    return_list = []
    teacher_scores = []
    
    teacher_agent = DQNAgent(state_size=env_config.state_size, action_size = env_config.teacher_action_size, seed=seed, action_list= env_config.teacher_action_list, LR = LR, BATCHSIZE = BATCHSIZE) 
    #teacher_agent.qnetwork_local.load_state_dict(torch.load(file))
    print(f'configration: environment = {game}, reward function type: {config.reward_function}, number of tasks = {env_config.teacher_action_size}, alpha = {alpha}, batch size = {BATCHSIZE}, step size = {LR} ')
    
    for i_episode in range(1, num_teaching_episodes+1):
        final_ALP = 0
        teacher_score = 1
        print(f'teacher episode {i_episode}')
        teacher_return = 0
        scores = list()
        teacher_action_list = list()


        

        #initalizing the student agent 
        mystudentagent = agent(env_config.rows, env_config.columns, env_config.student_num_actions, .5, .99)
        if config.debug:
            print('initalized the student agent')
            print(f'env has {env_config.rows} rows and {env_config.columns} columns')
            print(f'there are {env_config.student_num_actions} actions')
    
        env = env_config.live_env
        env_config.initalize_ALP_dict()
        mystudentagent.initalize_q_matrix() 
        print('initalized student q matrix')
                                                                       
        
        teacher_action_int =  random.randint(0, env_config.teacher_action_size-1) 
        teacher_action = env_config.teacher_action_list[teacher_action_int]
        print('first teacher action', teacher_action)

        first_score, last_score = train(mystudentagent, env, config.num_training_episodes, teacher_action) #this trains the agent for 10 episodes 
                                                                                #on the particular env then returns the score/# of time steps from the last episode
                                                                                #to do: probably better to do an avg of some number of episodes
        print('completed first training session')

        if str(teacher_action) not in list(env_config.returns_dict.keys()):
            update_entry = {str(teacher_action): first_score}
            env_config.returns_dict.update(update_entry)
            print(f'teacher action {teacher_action} is not in the returns dict, adding now with the first score {first_score}')
            print(env_config.returns_dict)

        source_task_score, source_task_cost, source_task_num_steps = evaluate_task(mystudentagent, env, teacher_action)
        
        target_task_score, _, _ = evaluate_task(mystudentagent, env, env_config.live_env.start_state)
        print(f'source task score {source_task_score}')
        print(f'target_task_score {target_task_score}')
    
        ALP = env_config.get_ALP(source_task_score, teacher_action)
        final_ALP+=ALP
        update_entry = {str(teacher_action): source_task_score}
        env_config.returns_dict.update(update_entry)
        print(f'updating returns dict with new value')
        env_config.update_ALP_dict(teacher_action, ALP)
        print(f'updated ALP dict {env_config.ALP_dict}')

        reward, SF , target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, source_task_cost)
        
        print(f'teacher action: {teacher_action_int}. Raw ALP = {ALP}, Updated LP = {alpha*ALP } SF = {SF}, reward = {reward}')


        teacher_action_list.append(teacher_action_int)
        

        traj = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, 0, ALP*env_config.alpha)
        print(f'first traj = {traj}')


        #print('\n')
        for j_episode in range(1, num_student_episodes+1):
            teacher_score+=1
            print('student episode', j_episode)
            


            teacher_action = env_config.get_teacher_action(teacher_agent, traj, eps) #returns raw teacher action
            teacher_action_int = env_config.convert_teacher_action(teacher_action)
            #teacher_action = env_config.teacher_action_list[teacher_action_int]
            if config.debug:
                print(f"raw teacher action {teacher_action}, indexed teacher action {teacher_action_int}")

            teacher_action_list.append(teacher_action_int)
            



            first_score, last_score = utils.train(mystudentagent, env, config.num_training_episodes, teacher_action) #this trains the agent for 10 episodes 
                                                                                    #on the particular env then returns the score/# of time steps from the last episode
                                                                                    #to do: probably better to do an avg of some number of episodes

            if str(teacher_action) not in list(env_config.returns_dict.keys()):
                update_entry = {str(teacher_action): first_score}
                env_config.returns_dict.update(update_entry)
                
                if config.debug:
                    print(f'teacher action {teacher_action} is not in the returns dict, adding now with the first score {first_score}')

            
            source_task_score, _, _ = evaluate_task(mystudentagent, env, teacher_action)
            target_task_score, _, _ = evaluate_task(mystudentagent, env, env_config.live_env.start_state)
            if config.debug:
                print(f'source task score {source_task_score}')
                print(f'target_task_score {target_task_score}')

            ALP = env_config.get_ALP(source_task_score, teacher_action)
            final_ALP+=ALP
            update_entry = {str(teacher_action): source_task_score}
            env_config.returns_dict.update(update_entry)
            

            #updating the returns with the raw learning progress rewards.. 
            env_config.update_ALP_dict(teacher_action, ALP)
            
            
            reward, SF, target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, source_task_cost)
                
            if config.debug:
                print(f'updated ALP dict = {env_config.ALP_dict}')
                print(f'teacher action: {teacher_action_int}. Raw ALP = {ALP}, Updated ALP = {1.5*ALP}, SF = {SF}, reward = {reward}')

        
            
            traj_prime = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, j_episode, ALP*env_config.alpha)
            if config.debug:
                print(f'traj prime = {traj_prime}')
            done, success_bonus = env_config.find_termination(target_task_score, j_episode, num_student_episodes)
            #print(f'student has solved the target task {done}')
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
                print('final ALP', final_ALP)
                break
        
            print('\n')
        print('end of episode')
        teacherdata.append(np.array(teacher_action_list))
        scoredata.append(np.array(scores)) 
        return_list.append(teacher_return)
        #print('\n')
        if i_episode % 10 == 0:
            model_name = f'{config.rootdir}/{config.experiment_folder}/{subdir}/teacher_agent_checkpoint_{config.reward_function}_{BATCHSIZE}_{LR}_{config.SR}_{seed}_{alpha}.pth'


            torch.save(teacher_agent.qnetwork_local.state_dict(), model_name)
            print(f'configration: alpha = {alpha}, batch size = {BATCHSIZE}, step size = {LR} ')
            print(f'scores: {scores}, return {teacher_return}, task list {teacher_action_list}')

    
    # print('teacher actions', teacherdata)
    # print('scores', scoredata)
    # print('teacher_return', return_list)
    
    print(np.mean(return_list), np.std(return_list))


    return return_list, teacher_scores
            
