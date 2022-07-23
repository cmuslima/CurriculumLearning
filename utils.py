import numpy as np
import os
from scipy import stats
import math
import random
import pickle
import numpy as np
import shutil

#to do:
#collect all states during a single training episode in the tabular case. 
#This class creates the environment (four rooms, maze, etc) and has most of the 
#functions for getting the teaacher interaction (i.e getting reward, getting next trajectory)



def get_rootdir(args, SR):
    
    if args.random_student_seed:
        if args.MP:
            rootdir = f'./RT/{args.env}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}_MP'
        else:

            rootdir = f'./RT/{args.env}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}'
        
        if args.student_type == 'DDPG':
            rootdir = f'./RT/{args.env}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}'    

            if args.env == 'fetch_push':
                if args.HER:
                    rootdir = f'./RT/{args.env}_{args.total_timesteps}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}'    
                else:
                    if args.three_layer_network:
                        if args.clear_buffer:
                            rootdir = f'./RT/{args.env}_{args.total_timesteps}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}/3layernetwork/clear_buffer'    
                        else:
                            rootdir = f'./RT/{args.env}_{args.total_timesteps}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}/3layernetwork'    

                    else:
                        if args.clear_buffer:
                            rootdir = f'./RT/{args.env}_{args.total_timesteps}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}/clear_buffer'    
                        else:
                            rootdir = f'./RT/{args.env}_{args.total_timesteps}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}'    


            else:
                if args.clear_buffer:
                    rootdir = f'./RT/{args.env}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}/clear_buffer'    
                else:
                    rootdir = f'./RT/{args.env}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}'    

        if args.student_type == 'PPO':
            rootdir = f'./RT/{args.env}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/hidden_size_i_{args.hidden_size_input}_hidden_size_o_{args.hidden_size_output}_teacher_network_h_{args.teacher_network_hidden_size}'    


        # if args.student_lr_transfer:
        #     rootdir = f'./RT/{args.env}/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{SR}/student_lr_transfer'
        # if args.Narvekar2018:
        #     args.rootdir = f'./RT/{args.env}/Narvekar2018/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{args.SR}'
        # if args.Narvekar2017:
        #     args.rootdir = f'./RT/{args.env}/Narvekar2017random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{args.SR}'
        # if args.L2T:
        #     args.rootdir = f'./RT/{args.env}/L2T/random_student_seeds/{args.saving_method}/single_student/{args.student_type}/{args.SR}'
    if args.multi_students:
        if args.multi_controller:
            if args.two_buffer:
                print("Running experiment with two buffers + multiple controllers + multi students")
                rootdir = f'./RT/{args.env}_{args.num_training_episodes}/random_student_seeds/{args.saving_method}/multi_students/multi_controller/two_buffer'
            else:
                print("Running experiment with a combined buffer + multiple controllers + multi students")
                rootdir = f'./RT/{args.env}_{args.num_training_episodes}/random_student_seeds/{args.saving_method}/multi_students/multi_controller/combined_buffer'
        else:
            print("Running experiment with a combined buffer + single controllers + multi students")
            rootdir = f'./RT/{args.env}_{args.num_training_episodes}/random_student_seeds/{args.saving_method}/multi_students/single_controller/combined_buffer'

    return rootdir

def import_modules(args):
    if args.env == 'fetch_push':
        try:
            from baselines.baselines import run
            from baselines.baselines import build_env
            return run, build_env
        except:
            import baselines
            from baselines import run, build_env
            print('success')
            return run, build_env
    if args.env == 'fetch_reach_3D_outer':
        from gym_fetch_RT import RT_run
        return RT_run
    if args.env == 'four_rooms' and args.tabular == False:
        try:
            from rl_starter_files_master.student_training import student_train
            from rl_starter_files_master.evaluate import evaluate_task
            from rl_starter_files_master.visualize import visualize
            return student_train, evaluate_task, visualize
        except:
            from RT.rl_starter_files_master.student_training import student_train
            from RT.rl_starter_files_master.evaluate import evaluate_task
            from RT.rl_starter_files_master.visualize import visualize
            return student_train, evaluate_task, visualize

def set_student_seeds():
    np.random.seed(0)
    print(f'finished seeing np with seed {0}')
    return 
def set_global_seeds(seed, args, student_seed=None):
    
    if args.random_curriculum or args.target_task_only:
        seed = student_seed
    try:
        import tensorflow as tf
        if args.random_student_seed:
            if args.evaluation:
                tf.set_random_seed(student_seed)
                print(f'finished seeing tf with seed = {student_seed}')
            else:
                tf.set_random_seed(seed)
                print(f'finished seeing tf with seed = {seed}')
    except:
        pass

    try:
        import torch
        if args.evaluation and args.student_type == 'PPO':
            torch.manual_seed(student_seed)
            print(f'finished torch random with seed = {student_seed}')

        else:
            torch.manual_seed(seed)
            print(f'finished torch random with seed = {seed}')
    except:
        pass
    
    if args.random_student_seed:
        if args.evaluation:
            np.random.seed(student_seed)
            print(f'finished seeing np with seed {student_seed}')
        else:
            np.random.seed(seed)
            print(f'finished seeing np with seed {seed}')



    if args.evaluation and args.student_type == 'PPO':
        random.seed(student_seed)
        print(f'finished seeing random with seed = {student_seed}')
    else:
        random.seed(seed)
        print(f'finished seeing random with seed = {seed}')
def convert_array_to_tuple(array, args):

    if args.student_type == "PPO" and (args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_policy'):
        array = np.reshape(array, (args.student_input_size,))
        #print('state here', array)
        #print(np.shape(array))
    if type(array) == type(np.array([1])):
        array = array.tolist()
        array = tuple(array)   
    
    return array
# def set_seed(seed, args): #i'll try it this way at frist
#     if args.env == 'fetch_reach_2D': #with this env, I seed numpy using using the other files. 
#         return
#     else:
#         np.random.seed(seed)
def get_one_hot_action_vector(index, value, value2, size):
    one_hot_vector = [value2]*size
    one_hot_vector[index] = value
    return one_hot_vector

def get_data(file):
    #print('file', file)
    with open(file,'rb') as input:
        data = pickle.load(input)
    return data
def save_data(model_name, data):
    with open(model_name, 'wb') as output:
        pickle.dump(data, output)

def get_model_name(args, dir, file_details):
    start_of_file = file_details[0]
    seed = file_details[1]
    if 'buffer' not in args.SR:
        if args.student_lr_transfer:
            model_name = f'{dir}/{start_of_file}_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.student_lr}_student_lr_transfer_{seed}'
        else:
            model_name = f'{dir}/{start_of_file}_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}'

    else:
        if args.student_lr_transfer:
            model_name = f'{dir}/{start_of_file}_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{args.student_lr}_student_lr_transfer_{seed}'
        else:
            model_name = f'{dir}/{start_of_file}_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'

    return model_name
def remove_dir(args, folder_path):
    #folder_path = f'./storage/{args.model}'
    #print(f' removing folder_path = {folder_path}')
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def make_dir(args, folder_path):
    print('folder_path', folder_path)
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print('finished making dir')
    except:
        return

def get_q_values(states, actions, student_agent): #only used for tabular case
    #s is a tuple, so states = list of tuples
    #print(actions)
    q_values = []
    one_hot_q_value = False
    if one_hot_q_value:
        for s,a in zip(states, actions):
            #print(s,a)
            q_value = student_agent.q_matrix[s][0][a]
            q_values.append(q_value)
    else:
        for s in states:
            q_value = student_agent.q_matrix[s][0]
            q_values.append(q_value)
    return q_values
def relu(self, x, threshold):
    if x > threshold:
        return x
    if x <= threshold:
        return 0
def get_file_name(args, dir, seed):
    if args.trained_teacher:
        if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action':
            model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}.pth'
        else:
            model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}.pth'
    else:
        model_name = None
    
    return model_name
def get_return_file_name(args, dir, seed):
    if args.trained_teacher:
        if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action':
            model_name = f'{dir}/teacher_return_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        else:
            model_name = f'{dir}/teacher_return_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}'
    else:
        model_name = None
    
    return model_name
# def normalize(self, values):
#     try:
#         max_value = max(values)
#         min_value = min(values)
#     except:
#         return values
#     if max_value == min_value:
#         print(f'Unable to normalize because min and value value are identical.. Returning un-normalized values')
#         return values
            
#     normalized_values = []
#     for v in values:
#         nv = (v - min_value)/(max_value-min_value)
#         normalized_values.append(nv)
#     return normalized_values

def normalize(min_value, max_value, value):

    nv = (value - min_value)/(max_value-min_value)
       
    return nv

def calculate_zscore(self, value_list):
    zscore = stats.zscore(value_list)[-1]

    return zscore
def clip(self, x):
    if x>=1:
        return .99

    if x<= -1:
        return -.99

