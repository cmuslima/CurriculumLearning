import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from rendering_envs import render_env
def get_data(file):

    with open(file,'rb') as input:
        data = pickle.load(input)
    return data
    



def get_all_data(files):
    all_data = dict()

    for f in files:
        file_name = f[0]
        file_description = f[1]
        try:
            
            with open(file_name,'rb') as input:
                data = pickle.load(input)
            print(f'Successfully retrieved {file_name}')
            #print(data)
        except:
            print(f'File {file_name} does not exist')
            continue
        name = file_description
        print(name)
        if name in list(all_data.keys()):
            continue
        all_data.update({name: data})
        
    return all_data

    


def plot_data(data,scores, env, args, buffer, rf):
    normalized = True
    
    subdir = 'plots'
    print(f'In the plot data function')

    if scores:
        if args.SR == 'q_matrix':
            model_name = f'student_performance_{args.SR}_{rf}'
        else:
            model_name = f'{env}_comp_SP2'
        y_label = 'Average Student \'s Return'
        x_label = 'Number of Student Episodes'
        title = 'Student Performance on Target Task'
    elif scores == False and normalized == False:
        if args.SR == 'q_matrix':
            model_name = f'raw_teacher_return_{args.SR}_{rf}'
        else:
            model_name = f'raw_teacher_return_{args.SR}_{buffer}_{rf}'
        y_label = 'Raw Teacher Return'
        x_label = 'Number of Teacher Episodes'
        title = 'Raw Teacher Return'
    elif scores == False and normalized == True:
        if args.SR == 'q_matrix':
            model_name = f'normalized_teacher_return_{args.SR}_{rf}'
        else:
            model_name = f'normalized_teacher_return_max'
        y_label = 'Normalized Teacher Return'
        x_label = 'Number of Teacher Episodes'
        title = 'Normalized Teacher Return'
    fig = plt.figure()
    num_runs = 5
    colors = ['orange', 'black', 'green', 'purple', 'red', 'blue', 'magenta', 'saddlebrown', 'aqua', 'magenta', 'darkcyan', 'peru','lightcoral', "indianred", 'firebrick', 'midnightblue', 'forestgreen', 'lime',\
         'navy', 'mediumblue', 'slateblue', 'silver','grey', 'blueviolet' ]
    #colors = ['lightcoral', "indianred", 'firebrick', 'brown', 'forestgreen', '√', 'green', 'lime', 'navy', 'mediumblue', 'slateblue', 'midnightblue']
   
    print(list(data.keys()))
    for idx, f in enumerate(list(data.keys())):
        file_description = f
        print(file_description, idx)
        try:
            
            if scores:
                mean = data[file_description][0][0:60]
                #print(data[file_description][0][0])
                std = data[file_description][1][0:60]
            else:
                if normalized:

                    mean = normalize(data[file_description][0][0:200])
                
                    #print(data[file_description][0][0])
                    std = normalize(data[file_description][1][0:200])
            
                else:
                    mean = data[file_description][0]
                    std = data[file_description][1]
            length = len(mean)

            plt.plot(np.arange(length), mean, lw = 2, color = colors[idx], label = file_description)
            variance = std/np.sqrt(num_runs)

            plt.fill_between(np.arange(length), mean+variance, mean-variance, facecolor=colors[idx], alpha=0.2)
        except:
            continue
    #plt.legend(loc = 'lower right')
    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.title(f'{title}')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # x_labels = ['0', '200', '400', '600', '800'] 
    # x_ticks = [0,20,40,60,80]
    # plt.xticks(ticks=x_ticks, labels=x_labels)
    fig.set_figwidth(12)
    fig.set_figheight(8)
    plt.savefig(f'{args.rootdir}/{args.experiment_folder}/{model_name}.png', bbox_extra_artists=(lg,), 
            bbox_inches='tight')

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



def get_files(args, scores, teacher_returns, teacher_actions, buffer, rf):
    normalized = False
    reward_functions = [rf]
    state_reps = ['buffer_policy']
    files = []
    batchsize_param = [32,64, 128,256]
    learningrate_param =  [.01, .001, .0001, .05, .005, .0005] #.0001, .0005] #for buffer size= 100, did 64 + .001
    buffer_sizes = [25,50,75]
    
    for buffer in buffer_sizes:
        for lr in learningrate_param:
            for b in batchsize_param:
                for rf in reward_functions:
                   
                    if rf == 'LP_log':
                        caption = f'R^(T) = -log(10+SF) + LP w/ bs = {b} and lr = {lr}'
                    if rf == 'simple_LP':
                        caption = f'Linear LP w/ bs = {b} and lr = {lr} + {buffer}' #f'R^(T) = -1+ LP w/ buffer state rep'
                    if rf == 'cost':
                        caption = f'Time-to-threshold w/ bs = {b} and lr = {lr} + {buffer}'
                    if scores:
                        subdir = 'evaluation-100'
                        if args.SR == 'buffer_q_table' or args.SR == 'buffer_policy' or args.SR == 'buffer_action':
                            base_file = f'{args.rootdir}/{args.experiment_folder}/{subdir}/evaluation_average_score_{rf}_{b}_{lr}_{args.SR}_{buffer}'           
                        else:
                            base_file = f'{args.rootdir}/{args.experiment_folder}/{subdir}/evaluation_average_score_{rf}_{b}_{lr}_{args.SR}'    
                    
                    if teacher_returns:
                        subdir = 'averaged-returns'
                        if normalized:
                            returns = 'z_score_normalized_averaged_returns'
                        else:
                            returns = 'raw_averaged_returns'
                        if args.SR == 'buffer_q_table' or args.SR == 'buffer_policy' or args.SR == 'buffer_action':
                            base_file = f'{args.rootdir}/{args.experiment_folder}/{subdir}/{returns}_{rf}_{b}_{lr}_{args.SR}_{buffer}'           
                        else:
                            base_file = f'{args.rootdir}/{args.experiment_folder}/{subdir}/{returns}_{rf}_{b}_{lr}_{args.SR}'    
                    if teacher_actions:
                        subdir = 'teacher-returns' 
                        if args.SR == 'buffer_q_table' or args.SR == 'buffer_policy' or args.SR == 'buffer_action':
                            base_file = f'{args.rootdir}/{args.experiment_folder}/{subdir}/all_teacher_actions_{rf}_{b}_{lr}_{args.SR}_{buffer}'           
                        else:
                            base_file = f'{args.rootdir}/{args.experiment_folder}/{subdir}/all_teacher_actions_{rf}_{b}_{lr}_{args.SR}'           

                    
                    
                    caption = f'Reward Function: {caption}'
                    print(f'Base file = {base_file}, caption = {caption}')
                    files.append((base_file, caption))
    return files

def get_best_files(args, scores, teacher_returns, teacher_actions, buffer, rf):
    normalized = False
    
    files = []

    best_params = [('cost', 'buffer_policy', 256, .001, 75), ('cost', 'buffer_q_table', 256, .005, 50), ('cost', 'q_matrix', 128, .001, 50),\
        ('simple_LP', 'buffer_policy', 128, .001, 75), ('simple_LP', 'buffer_q_table', 128, .01, 75), ('simple_LP', 'q_matrix', 256, .0001, 50),\
        ] #(None, "Random", None, None, None), (None, 'Target-Only',None, None, None)
    best_params = [('simple_LP', 'buffer_policy', 128, .001, 75)]#,(None, "Random", None, None, None), (None, 'Target-Only',None, None, None)]
    print(len(best_params))
    for param in best_params:
        rf = param[0]
        state_rep = param[1]
        b = param[2]
        lr = param[3]
        buffer = param[4]

        if state_rep == 'buffer_policy':
            s = 'PE-OneHotPolicy'
        if state_rep == 'buffer_q_table':
            s = 'PE-QValues'
        if state_rep == 'q_matrix':
            s = "Parameters"       
        if rf == 'simple_LP':
            caption = f'LP + {s}' #f'R^(T) = -1+ LP w/ buffer state rep'
        if rf == 'cost':
            caption = f'Time-to-threshold + {s}'
        if rf == None:
            caption = state_rep
        if scores:
            subdir = 'evaluation-200'
            print('state rep', state_rep)
            if state_rep == 'buffer_q_table' or state_rep == 'buffer_policy':
                base_file = f'{args.rootdir}/{state_rep}/{subdir}/evaluation_average_score_{rf}_{b}_{lr}_{state_rep}_{buffer}'
                print('basefile', base_file)     
                print('here')      
            if state_rep == 'Random' or state_rep == "Target-Only":
                base_file = f'{args.rootdir}/{state_rep}/evaluation-data/evaluation_average_score'
                print('here 2')
            if state_rep == 'q_matrix':
                print('here3')
                base_file = f'{args.rootdir}/{state_rep}/{subdir}/evaluation_average_score_{rf}_{b}_{lr}_{state_rep}'    


        if teacher_returns:
            subdir = 'averaged-returns'
            returns = 'raw_averaged_returns'
            if state_rep == 'buffer_q_table' or state_rep == 'buffer_policy':
                base_file = f'{args.rootdir}/{state_rep}/{subdir}/{returns}_{rf}_{b}_{lr}_{state_rep}_{buffer}'           
            else:
                base_file = f'{args.rootdir}/{state_rep}/{subdir}/{returns}_{rf}_{b}_{lr}_{state_rep}'    
        if teacher_actions:
            subdir = 'teacher-returns'
            for seed in range(0,5):
                if state_rep == 'buffer_q_table' or state_rep == 'buffer_policy':

                    base_file = f'{args.rootdir}/{state_rep}/{subdir}/all_teacher_actions_{rf}_{b}_{lr}_{state_rep}_{buffer}_{seed}'           
                else:
                    base_file = f'{args.rootdir}/{state_rep}/{subdir}/all_teacher_actions_{rf}_{b}_{lr}_{state_rep}_50_{seed}'           

                if rf == 'simple_LP':
                    caption = f'LP + {s} + {seed}' #f'R^(T) = -1+ LP w/ buffer state rep'
                if rf == 'cost':
                    caption = f'Time-to-threshold + {s} + {seed}'
                files.append((base_file, caption))    
        if teacher_actions == False:
            print(f'Base file = {base_file}, caption = {caption}')
            files.append((base_file, caption))
    return files
def calculate_area_under_curve(data_list):
    area = np.sum(data_list)
    return area

def get_area_under_curve_all_data(data):
    hyper_param_dict = {}
    for idx, f in enumerate(list(data.keys())):
        file_description = f
        print(file_description, idx)
        area = calculate_area_under_curve(data[file_description][0])
        hyper_param_dict[file_description] = area
        print(area, f)
    #print(hyper_param_dict)
    return hyper_param_dict

def get_max_hyperparam(hyper_param_dict):
    max_key = max(hyper_param_dict, key=hyper_param_dict.get)
    return max_key

def plotting(args):
    scores = True
    teacher_returns = False
    teacher_actions = False    
    buffer =75
    rf = 'simple_LP'
    #files = get_files(args, scores, teacher_returns, teacher_actions, buffer, rf)
    files = get_best_files(args, scores, teacher_returns, teacher_actions, buffer, rf)
    #print(f'Got the files')
    all_data = get_all_data(files)
    # hyper_param_dict = get_area_under_curve_all_data(all_data)
    # print('here')
    # max_key = get_max_hyperparam(hyper_param_dict)
    # print('max key',max_key)
    #files = get_best_files(args, scores, teacher_returns, teacher_actions, buffer, rf)
    #all_data = get_all_data(files)
    
    env = 'maze'
    
    plot_data(all_data,scores, env, args, buffer, rf)



def plot_actions(args):
    scores = False
    teacher_returns = False
    teacher_actions = True
    buffer =75
    rf = 'simple_LP'
    files = get_best_files(args, scores, teacher_returns, teacher_actions, buffer, rf)
    print(files)
    all_data = get_all_data(files)
    colors = ['black', 'blue', 'red', 'green', 'orange', 'purple']
    #for exp in list(all_data.keys())

    interval1 = []
    interval2 = []
    interval3 = []
    for seed in range(0,5):
        x = all_data[f'LP + PE-OneHotPolicy + {seed}']
        print()
        curr = x[0][200]
        print(curr)
        interval1.append(curr[0:3])
        interval2.append(curr[3:6])
        interval3.append(curr[6:])
        
    flat_interval1 = [item for sublist in interval1 for item in sublist]
    flat_interval2 = [item for sublist in interval2 for item in sublist]
    flat_interval3 = [item for sublist in interval3 for item in sublist]
    all_intervals = [flat_interval1, flat_interval2, flat_interval3]
    
    for i in all_intervals:
        raw_freqs = np.bincount(i)
        probability = raw_freqs/sum(raw_freqs)
        #print(raw_freqs, probability)
        render_env(probability)
    return probability
    #print(all_data)
    # for i in actions:
    #     plt.scatter(np.arange(0, len(actions[0])), i)
    # plt.show()
    

    #     actions = get_data('./action_return/evaluation-data/all_teacher_scores_simple_LP_256_0.005_buffer_policy_75')
    #     print(actions[0])
    
    #     for i in actions[0]:
    #         plt.scatter(np.arange(0, len(actions[0])), i)
    #     plt.show()



def action(args):
    subdir = 'teacher-returns'
    base_file = f'{args.rootdir}/{args.experiment_folder}/{subdir}/all_teacher_actions_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{args.buffer_size}_4'

    actions =get_data(base_file)
    print('len of actions', len(actions[0]))
    assert len(actions[0]) == 300
    print(actions[0][200])
