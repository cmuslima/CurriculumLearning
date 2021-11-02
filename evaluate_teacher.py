
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
from student_agent import agent
from evaluation import evaluate_task
import numpy as np
from teacher_agent import DQNAgent
from global_vars import env_variables
import utils
import torch
import env
import config



def eval_loop(LR, batch_size,num_student_episodes,num_teaching_episodes, game, SR, file, seed, alpha):
    final_done = True
    

    #initalizing environment
    env_config = env_variables()
    env_config.alpha = alpha
    eps = 0.00
    scores = list()
    teacher_action_list = list()
    print('LR', LR, 'batchsize', batch_size, 'alpha', alpha)
    print(f'seed = {seed}')
    
    if config.random_curriculum == False and config.target_only == False:
        print('uploading teacher policy')
        teacher_agent = DQNAgent(state_size=env_config.state_size, action_size = env_config.teacher_action_size, seed=seed, action_list= env_config.teacher_action_list, LR = LR, BATCHSIZE = batch_size) 
        teacher_agent.qnetwork_local.load_state_dict(torch.load(file))
    
   

    for i_episode in range(1, num_teaching_episodes+1):
        teacher_score = 0
       
       
        scores = list()
        teacher_action_list = list()
        teacher_reward_list = list()
        

        #initalizing the student agent 
        mystudentagent = agent(env_config.rows, env_config.columns, env_config.student_num_actions, .5, .99)
        
    
        env = env_config.live_env
        env_config.initalize_ALP_dict()
        mystudentagent.initalize_q_matrix() 
                                                                       
        
        teacher_action_int =  random.randint(0, env_config.teacher_action_size-1) 
        teacher_action = env_config.teacher_action_list[teacher_action_int]
      
        first_score, last_score = utils.train(mystudentagent, env, config.num_training_episodes, teacher_action) #this trains the agent for 10 episodes 
                                                                                #on the particular env then returns the score/# of time steps from the last episode
                                                                                #to do: probably better to do an avg of some number of episodes
        if config.random_curriculum == False and config.target_only == False:
            if str(teacher_action) not in list(env_config.returns_dict.keys()):
                update_entry = {str(teacher_action): first_score}
                env_config.returns_dict.update(update_entry)
            
            source_task_score, source_task_cost, source_task_num_steps = evaluate_task(mystudentagent, env, teacher_action)
            
            target_task_score, _, _ = evaluate_task(mystudentagent, env, env_config.live_env.start_state)
            # print(f'source task score {source_task_score}, raw num steps {source_task_num_steps}')
            # print(f'target_task_score {target_task_score}, raw num steps {target_task_num_steps}')
        
            ALP = env_config.get_ALP(source_task_score, teacher_action)
            env_config.update_ALP_dict(teacher_action, ALP)
            update_entry = {str(teacher_action): source_task_score}
            env_config.returns_dict.update(update_entry)
            teacher_action_list.append(teacher_action_int)
            
            reward, SF , target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, source_task_cost)
            
            #print(f'teacher action: {teacher_action_int}. Raw ALP = {ALP}, Updated LP = {alpha*ALP, }SF = {SF}, reward = {reward}')


        

            traj = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, 0, ALP*env_config.alpha)
            
            

    
        for j_episode in range(1, num_student_episodes+1):
           
            teacher_score+=1
          

            if config.random_curriculum and config.target_only == False:
                teacher_action_int = random.randint(0, env_config.action_size-1)
                teacher_action = env_config.action_list[teacher_action_int]

            elif config.target_only and config.random_curriculum == False:
                teacher_action_int = 0
                teacher_action = env_config.action_list[teacher_action_int]
            else:
                teacher_action = env_config.get_teacher_action(teacher_agent, traj, eps) #returns raw teacher action
                teacher_action_int = env_config.convert_teacher_action(teacher_action)
   
            teacher_action_list.append(teacher_action_int)
            
            first_score, last_score, = utils.train(mystudentagent, env, config.num_training_episodes, teacher_action) #this trains the agent for 10 episodes 
                                                                                    #on the particular env then returns the score/# of time steps from the last episode
                                                                                    #to do: probably better to do an avg of some number of episodes

            if config.random_curriculum == False and config.target_only == False:
                if str(teacher_action) not in list(env_config.returns_dict.keys()):
                    update_entry = {str(teacher_action): first_score}
                    env_config.returns_dict.update(update_entry)
                
                source_task_score, source_task_cost, source_task_num_steps = evaluate_task(mystudentagent, env, teacher_action)
        
            target_task_score, _, target_task_num_steps = evaluate_task(mystudentagent, env, env_config.live_env.start_state)

            if config.random_curriculum == False and config.target_only == False:
                ALP = env_config.get_ALP(source_task_score, teacher_action)
    
                update_entry = {str(teacher_action): source_task_score}
                env_config.returns_dict.update(update_entry)
            

                #updating the returns with the raw learning progress rewards.. 
                env_config.update_ALP_dict(teacher_action, ALP)

                reward, SF, target_task_success  = env_config.get_teacher_reward(teacher_action, ALP, target_task_score, source_task_cost)

            
                traj_prime = env_config.get_traj_prime(teacher_action_int, source_task_score, mystudentagent, reward, target_task_success, j_episode, ALP*env_config.alpha)
                traj = traj_prime

            done, success_bonus = env_config.find_termination(target_task_score, j_episode, num_student_episodes)
      
            scores.append(target_task_score) #I want to collect data on the target task.. to see if over time, it approves on the target task.
            

            if done and final_done:
                final_teacher_score = teacher_score
                final_done = False
                
            
       
       
        print('teacher actions', teacher_action_list)
        print('scores', scores)
        print('final_teacher_score', final_teacher_score)
        

    
    return np.array(scores), np.array(teacher_action_list), final_teacher_score
            
    


    
if __name__ == '__main__':
    subdir1= 'teacher-checkpoints'
    subdir2= 'evaluation-data'
    num_student_episodes = config.student_episodes
    num_teaching_episodes = 1
    game = config.env
    SR = config.SR
    BATCHSIZE =  [64]
    learningrate = [.001]
    alpha_list = [1.25]
    for alpha in alpha_list:
        for batch_size in BATCHSIZE:
            for LR in learningrate:
                evaluation_scores_all_runs = []
                teacher_actions_all_runs = []
                teacher_score = []
                random.seed(0)
                
                for i in range(0,config.runs):
                    file = f'{config.rootdir}/{config.experiment_folder}/{subdir1}/teacher_agent_checkpoint_{config.reward_function}_{batch_size}_{LR}_{config.SR}_{i}_{alpha}.pth'
                    
                    
                    evaluation_average_score, evaluation_teacher_data, final_teacher_score = eval_loop(LR, batch_size,num_student_episodes,num_teaching_episodes, game, SR, file, i, alpha)
                    evaluation_scores_all_runs.append(evaluation_average_score)
                    teacher_actions_all_runs.append(evaluation_teacher_data)
                    teacher_score.append(final_teacher_score)
                averaged_eval_scores = [np.mean(np.array(evaluation_scores_all_runs), axis = 0), np.std(np.array(evaluation_scores_all_runs), axis = 0)]
                print('average scores', averaged_eval_scores[0])
                averaged_teacher_actions = [np.mean(np.array(teacher_actions_all_runs), axis = 0), np.std(np.array(teacher_actions_all_runs), axis = 0)]
                assert len(teacher_score) == config.runs

                if config.target_only == False and config.random_curriculum == False:
                    model_name1 = f'{config.rootdir}/{config.experiment_folder}/{subdir2}/evaluation_average_score_{config.reward_function}_{batch_size}_{LR}_{config.SR}_{alpha}'
                    model_name2 = f'{config.rootdir}/{config.experiment_folder}/{subdir2}/evaluation_teacher_data_{config.reward_function}_{batch_size}_{LR}_{config.SR}_{alpha}'
                    model_name3 = f'{config.rootdir}/{config.experiment_folder}/{subdir2}/all_teacher_actions_{config.reward_function}_{batch_size}_{LR}_{config.SR}_{alpha}'
                    model_name4 = f'{config.rootdir}/{config.experiment_folder}/{subdir2}/all_teacher_score_{config.reward_function}_{batch_size}_{LR}_{config.SR}_{alpha}'




                    
                with open(model_name1, 'wb') as output:
                    pickle.dump(averaged_eval_scores, output)
                with open(model_name2, 'wb') as output:
                    pickle.dump(averaged_teacher_actions, output)
                with open(model_name3, 'wb') as output:
                        pickle.dump(teacher_actions_all_runs, output)
                with open(model_name4, 'wb') as output:
                    pickle.dump(teacher_score, output)

