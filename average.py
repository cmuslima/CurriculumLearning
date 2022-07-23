import pickle
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy import stats
from plotting_graphs_updated import get_data, calculate_area_under_curve
import pickle
import utils
import matplotlib.pyplot as plt


def get_model_name(args, dir, file_details, batchsize, lr, buffersize):
    start_of_file = file_details[0]
    seed = file_details[1]
    if args.SR == 'params':
        model_name = f'{dir}/{start_of_file}_{args.SR}_{args.reward_function}_{lr}_{batchsize}_{seed}'
    else:
        model_name = f'{dir}/{start_of_file}_{args.SR}_{args.reward_function}_{lr}_{batchsize}_{buffersize}_{seed}'
    return model_name

def normalize(values):
    max_value = max(values)
    min_value = min(values)
    print('max value', max_value)
    print('min value', min_value)
    normalized_list = []
    for v in values:
        normalized_value = (v-min_value)/(max_value-min_value)
        normalized_list.append(normalized_value)
        #print('before =', v, 'after =', normalized_value)

    return normalized_list

def average_data(args):


    student_D = args.comparing_scores
    all_runs = []
    all_AOC = []
    final_performances = []
    area_under_curve_list = dict()
    for i in range(0,args.num_runs):
        print('number of runs', args.num_runs, i)
        try:
            if 'buffer' in args.SR:
                print('here')
                if student_D:
                        
                    
                        if args.student_transfer:
                            subdir = 'transfer_q_learning_to_sarsa'
                            data = get_data(f'{args.rootdir}/evaluation-data/{subdir}/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_sarsa_{i}')
                        if args.student_lr_transfer:
                            subdir = f'lr_transfer_{args.student_lr}'
                            data = get_data(f'{args.rootdir}/evaluation-data/{subdir}/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{args.student_lr}_{i}')

                        if args.student_NN_transfer:
                            subdir = 'NN_large_64_64'
                            data = get_data(f'{args.rootdir}/evaluation-data/{subdir}/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{i}')
                        
                        if args.student_transfer == False and args.student_lr_transfer == False and args.student_NN_transfer == False and args.random_curriculum == False and args.target_task_only == False:
                            # try:
                            #     print('should be here')
                            #     data = get_data(f'{args.rootdir}/evaluation-data/{args.reward_function}_reward_data/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{i}')
                            #     print(data)
                            # except:
                                
                                data = get_data(f'{args.rootdir}/evaluation-data/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{i}')
                                print('did I get here')
                else:
                    try:
                        #print('getting teacher data')
                        data = get_data(f'{args.rootdir}/teacher-data/{args.reward_function}_teacher_data/teacher_return_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{i}')
                        print()
                    except:
                        print('getting teacher data', f'{args.rootdir}/teacher-data/teacher_return_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{i}')
                        data = get_data(f'{args.rootdir}/teacher-data/teacher_return_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{i}')
                        print('data length = ', len(data))
                        if len(data) < 50:
                            data = data[0]
                        print(data)
                        # if len(data) == 1:
                        #     data = data[0]
                        #     print(data)

            else:
                print('im not using PE ')
                if student_D:     
                    print('student D', student_D)   
                    print('file', f'{args.rootdir}/evaluation-data/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{i}')    
                    data = get_data(f'{args.rootdir}/evaluation-data/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{i}')
                    print(data)
                else:
                    if args.reward_function == 'LP' or args.reward_function == 'target_task_score' or args.reward_function == '0_target_task_score':
                        data = get_data(f'{args.rootdir}/teacher-data/{args.reward_function}_teacher_data/teacher_return_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{i}')
                    else:
                        data = get_data(f'{args.rootdir}/teacher-data/teacher_return_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{i}')
        except: 
            if args.random_curriculum:
                dir = f'./RT/{args.env}/random_curriculum'
                data = get_data(f'{dir}/random_student_score_{i}')
                print(f'getting data {dir}/random_student_score_{i}')

            if args.target_task_only:
                dir = f'./RT/{args.env}/target_task_only'
                data = get_data(f'{dir}/target_task_only_student_score_{i}')


        if args.env == 'four_rooms':
            data = data[0:90]
        all_runs.append(data)

        AOC = calculate_area_under_curve(data)
        final_performance = data[-1]
        all_AOC.append(AOC)
        final_performances.append(final_performance)

        
    #print(len(all_runs))
   # print(all_runs)
    print(f'len of list {len(all_runs)}')
    if len(all_runs) == args.num_runs:
      
        raw_averaged_returns = [np.mean(np.array(all_runs), axis = 0), np.std(np.array(all_runs), axis = 0)]
        #print('raw_averaged_return', raw_averaged_returns)



        if 'buffer' in args.SR:
            if student_D:
                model_name_return = f'{args.rootdir}/evaluation-data/raw_averaged_returns_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_'

                if args.student_transfer:
                    model_name_return = f'{args.rootdir}/evaluation-data/{subdir}/raw_averaged_returns_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_sarsa_'
                if args.student_lr_transfer:
                    model_name_return = f'{args.rootdir}/evaluation-data/{subdir}/raw_averaged_returns_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{args.student_lr}_'
                if args.student_NN_transfer:
                    model_name_return = f'{args.rootdir}/evaluation-data/{subdir}/raw_averaged_returns_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_'  

            else:
                model_name_return = f'{args.rootdir}/teacher-data/non_normalized_teacher_raw_averaged_returns_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_'


        else:
            if student_D:
                model_name_return = f'{args.rootdir}/evaluation-data/raw_averaged_returns_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_'
            
            else:
                model_name_return = f'{args.rootdir}/teacher-data/non_normalized_teacher_raw_averaged_returns_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_'
         
        name = f'{args.SR} + {args.teacher_lr} + {args.reward_function} + {args.teacher_batchsize} + {args.teacher_buffersize}'
        model_name_AOC = f'./RT/plots/{args.env}/area_under_curve_{args.SR}_{args.reward_function}_{args.env}'
        if args.random_curriculum:
            model_name_return = f'{dir}/raw_averaged_returns_random'
            model_name_AOC = f'./RT/plots/{args.env}/area_under_curve_random'
            name = f'random'
        if args.target_task_only:
            model_name_return = f'{dir}/raw_averaged_returns_target'
            model_name_AOC = f'./RT/plots/{args.env}/area_under_curve_target'   
            name = f'target task only'
        with open(model_name_return, 'wb') as output:
            pickle.dump(raw_averaged_returns, output)
        print(f'Finished averaged student scores {model_name_return} {raw_averaged_returns}')


        area_under_curve_list[name] = all_AOC
        if student_D and args.AUC:
            print('saving area under the curve value')
            print(area_under_curve_list, name, sum(all_AOC)/len(all_AOC))
            print(final_performances)
            utils.save_data(model_name_AOC,area_under_curve_list)
    return #area_under_curve

def quick_plot(args):
    batchsizes = [128]
    lr = .001
    rewards = ['target_task_score']
    SR = 'buffer_q_table'
    buffer = 100
    for idx,rf in enumerate(rewards):
        batchsize = batchsizes[idx]
        model_name = f'{args.rootdir}/teacher-data/non_normalized_teacher_raw_averaged_returns_{SR}_{rf}_{lr}_{batchsize}_{buffer}_'
        data = get_data(model_name)
        mean = data[0]
        std = data[1]
        caption = rf
        plt.plot(np.arange(300), mean, lw = 2, label = caption)
        CI = 1.96
        variance = (CI*std)/np.sqrt(args.num_runs)
        plt.fill_between(np.arange(300), mean+variance, mean-variance, alpha=0.2)
    
    plt.legend(loc='best')
    plt.show()