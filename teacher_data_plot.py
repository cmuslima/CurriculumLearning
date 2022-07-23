import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import utils
from scipy import stats
from rendering_envs import render_env
from collections import Counter
def get_data(file):

    with open(file,'rb') as input:
        data = pickle.load(input)
    return data
    



def get_all_data(files):
    print('in this function')
    all_data = dict()
    print(files)
    for f in files:
        file_name = f[0]
        file_description = f[1]
        try:
            print('using file', file_name)
            with open(file_name,'rb') as input:
                data = pickle.load(input)
            print(f'Successfully retrieved {file_name}')
            print(data)
        except:
            print(f'File {file_name} does not exist')
            continue
        name = file_description
        print(name)
        if name in list(all_data.keys()):
            continue
        all_data.update({name: data})
        
    return all_data

    
def get_title(args):
    if args.comparing_scores:
        if args.env == 'maze':
            title = 'Student Performance in Maze'
        elif args.env == 'four_rooms':
            title = 'Student Performance in Four Rooms'
        elif args.env == 'fetch_reach_3D_outer':
            title = 'Student Performance in Fetch Reach'

    else:
        if args.env == 'maze':
            title = 'Maze'
        elif args.env == 'four_rooms':
            title = 'Four Rooms'
        elif args.env == 'fetch_reach_3D_outer':
            title = 'Fetch Reach'
    if args.student_transfer:
        title = f'Generalization from Q Learning Student to SARSA student'
    if args.student_lr_transfer:
        title = f'Generalization from Student w/ LR = {args.student_lr}' 
    return title
def get_axis_tiles(args):
    if args.comparing_scores:
        
        if args.env == 'fetch_reach_3D_outer':
            x_label = 'Number of Episodes (x6)'
            y_label = 'Student Success Rate'
        if args.env == 'fetch_push':
            x_label = '# of Student Time Steps(10000)'
        if args.env == 'four_rooms':
            x_label = 'Number of Episodes (x25)'
            y_label = 'Student \'s Return'
        if args.env == 'maze':
            x_label = 'Number of Episodes (x10)'
            y_label = 'Student \'s Return'
        model_name = f'student_scores'
        title = 'Student Performance on Target Task'
    else:
        y_label = 'Teacher Return'
        x_label = 'Number of Teacher Episodes'
        model_name = f'teacher_returns'
        title = ''
    return x_label, y_label, model_name 


def save_image(args, model_name, fig):
    
    if args.single_baseline_comp:
        if args.student_transfer == False and args.student_lr_transfer == False:
            dir = f'{args.rootdir}/evaluation-data/plots'
    
        if args.student_transfer:
            folder = 'q_learning_to_sarsa_transfer'
            dir = f'{args.rootdir}/{folder}/plots'
        
        if args.student_lr_transfer:
            folder = f'lr_{args.student_lr}'
            dir = f'{args.rootdir}/{folder}/plots'


    else:
        if args.student_transfer == False and args.student_lr_transfer == False:
            dir = f'./RT/{args.env}/thesis_plots'
    
        if args.student_transfer:
            folder = 'q_learning_to_sarsa_transfer'
            dir = f'./RT/{args.env}/single_student/{args.student_type}/{folder}/plots'
        if args.student_lr_transfer:
            folder = f'lr_{args.student_lr}'
            dir = f'./RT/{args.env}/single_student/{args.student_type}/{folder}/plots'

    utils.make_dir(args, dir)
    file_name = f'{dir}/{model_name}.png'        
    print('file name', file_name)
    fig.savefig(file_name)

def plot_data(data,scores,args):
    normalized = True
    
    subdir = 'plots'
    print(f'In the plot data function')
    title = get_title(args)
    x_label, y_label, model_name = get_axis_tiles(args)
    #fig = plt.figure()
    
    colors = ['green', 'black', 'lawngreen', 'purple', 'red', 'blue', 'saddlebrown', 'saddlebrown', 'aqua', 'magenta', 'darkcyan', 'peru','lightcoral', "indianred", 'firebrick', 'midnightblue', 'forestgreen', 'lime',\
         'navy', 'mediumblue', 'slateblue', 'silver','grey', 'blueviolet' ]
    #colors = ['lightcoral', "indianred", 'firebrick', 'brown', 'forestgreen', '√', 'green', 'lime', 'navy', 'mediumblue', 'slateblue', 'midnightblue']
   
    #order = [0,2, 1, 3, 6, 5, 7,8, 4]
    #print(list(data.keys()))
    #ax = f'ax_{idx}'
    #fig, ax = plt.subplots() 
    d = []
    for idx, f in enumerate(list(data.keys())):
        file_description = f
        print(file_description, idx)
       
        
        if scores:
            mean = data[file_description][0]
            
            std = data[file_description][1]
        else:
            # if idx == 1:
            #     mean = data[file_description][0][0]
            
            #     std = data[file_description][1][0]
            #     print(mean,std)
            # else:
            mean = data[file_description][0]
            
            std = data[file_description][1]
            # mean = data[file_description][0]
            # std = data[file_description][1]
        length = len(mean)

        color = colors[idx]
        marker = ''
        #marker = 'o'
        # if 'Time-to-threshold' in file_description:
        #     color = 'blue'
        #     #marker = 'x'
        # if 'LP' in file_description:
        #     color = 'green'
        #     #marker = 'o'
        #     print('here')
        # # if 'L2T reward' in file_description:
        # #     color = 'blue'

        # if 'PE-OneHotPolicy' in file_description:
        #     #color = 'green'
        #     marker = 'D'
        # if 'PE-QValues' in file_description:
        #     marker = "x"
        #     #color = 'orange'
            
        # if 'Parameters' in file_description:
        #     #color = 'blue'
        #     marker = '*'
        # if 'L2T state' in file_description:
        #     #color = 'blue'
        #     marker = 'o'
        marker = ''

        rewards = ['Time-to-threshold', 'LP', 'Huang(2019) reward', 'Fan(2018) reward', 'Matiisen(2017) reward', 'Ruiz(2019) reward', 'Sparse Ruiz(2019) reward']
        states = ['PE-Actions', 'Fan(2018) state', 'Huang(2019) state', 'Parameters', 'PE-QValues']
        #print(f'file description', file_description)
        for r in rewards:       
            for s in states:
                if s in file_description and r in file_description:
                    print(f'file description', file_description)
                    caption =  f'{s} + {r}'
                    print('caption', caption)
                
                if 'PE-Actions' in file_description and 'LP' in file_description:
                    caption = 'PE-Actions + LP (Ours)'
                    #color = 'purple'
                if 'PE-QValues' in file_description and 'LP' in file_description:
                    caption = 'PE-QValues + LP (Ours)'
                    color = 'orange'
                if 'Fan(2018) state' in file_description and 'Fan(2018) reward' in file_description:
                    caption = 'Fan(2018)'
                    print('using caption fan', caption)
                    color = 'black'
                if 'Huang(2019) state' in file_description and 'Huang(2019) reward' in file_description:
                    caption = 'Huang(2019)'
                    print('using caption huang', caption)

                if 'Parameters' in file_description and 'Time-to-threshold' in file_description:
                    caption = 'Narvekar(2017)'
                    print('using caption', caption, file_description)
                    color = 'green'
                if 'Random' in file_description:
                    color = 'red'
                if "Target" in file_description:
                    color = 'blue'
                

                    
        if "Random" in file_description:
            caption = "Random"
        if 'Target' in file_description:
            caption = 'Target'
        #ax = f'ax_{idx}'
        d.append((mean, std, caption, color))
    
    CI = 1.96
    num_runs = 10    
    mean1 = d[0][0][0:90]
    std1 = d[0][1][0:90]
    caption1 = d[0][2]
    color1 = d[0][3]
    mean2 = d[1][0][0:90]
    std2 = d[1][1][0:90]
    caption2 = d[1][2]
    color2 = d[1][3]
      
    variance1 = (CI*std1)/np.sqrt(num_runs)
    variance2 = (CI*std2)/np.sqrt(num_runs)

    length = len(mean1)
    print(length)
    print(len(mean1))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of episodes')
    ax1.set_ylabel(f'Teacher Return: {caption1}', color=color1, fontsize = 20)
    
    
    plot1 =ax1.plot(np.arange(length), mean1, lw = 2, color = color1, marker = marker, label = caption1)
    plt.fill_between(np.arange(length), mean1+variance1, mean1-variance1, facecolor=color1, alpha=0.2)
    plt.legend( loc='best')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel(f'Teacher Return: {caption1}', color=color2, fontsize = 20)
    plot2 =ax2.plot(np.arange(length), mean2, lw = 2, color = color2, marker = marker, label = caption2)
    plt.fill_between(np.arange(length), mean2+variance2, mean2-variance2, facecolor=color2, alpha=0.2)


    #handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend( loc='best')
    #plt.legend(loc='best')
    #plt.tight_layout()
    #lg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.05, 1.0), fontsize = 20)
    #plt.tight_layout()
    #lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.title(title, fontsize=25)
    #plt.ylabel(y_label, fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    # x_labels = ['0', '200', '400', '600', '800'] 
    # x_ticks = [0,20,40,60,80]
    # plt.xticks(ticks=x_ticks, labels=x_labels)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    plt.show()
    #lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
   
    # fig.savefig('example.png',
    #         format='png', 
    #         bbox_extra_artists=(lg,), 
    #         bbox_inches='tight')
            
    #save_image(args, model_name, fig)
    #fig.savefig(file_name)

def normalize(values, min_value, max_value):
    # max_value = max(values)
    # min_value = min(values)
    print('max value', max_value)
    print('min value', min_value)
    normalized_list = []
    for v in values:
        normalized_value = (v-min_value)/(max_value-min_value)
        normalized_list.append(normalized_value)
        #print('before =', v, 'after =', normalized_value)

    return normalized_list



def get_subdir(args,rootdir):
    prev_dir =f'{rootdir}/evaluation-data'
    if args.comparing_scores:
        new_dir = f'{rootdir}/evaluation-data'
    else:
        new_dir = f'{rootdir}/teacher-data'

    if args.student_transfer: 
        folder = 'q_learning_to_sarsa_transfer'
        new_dir = f'{args.rootdir}/evaluation-data/averaged-student-scores/{folder}'
        prev_dir =f'{args.rootdir}/evaluation-data/{folder}'
    # if args.student_lr_transfer:
    #     folder = f'lr_{args.student_lr}'
    #     new_dir = f'{args.rootdir}/evaluation-data'
    #     prev_dir =f'{args.rootdir}/evaluation-data'

    return prev_dir, new_dir


def check_data(args, rootdir):
    folder = 'teacher-data'
 


    total = 0
    for i in range(args.num_runs):
        try:
            try:
                base_file = f'{args.rootdir}/teacher-data/teacher_scores_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{i}'

                data = get_data(base_file)
                len_data = len(data)
                #assert len_data == args.student_episodes or len_data == args.student_episodes+1 or len_data == args.student_episodes-1
                print(f'{base_file} is GOOD', len_data)
                total+=1
            except:
                base_file = f'{args.rootdir}/teacher-data/teacher_scores_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{i}'

                data = get_data(base_file)
                len_data = len(data)
                #assert len_data == args.student_episodes or len_data == args.student_episodes+1 or len_data == args.student_episodes-1
                print(f'{base_file} is GOOD', len_data)
                total+=1
        except:
            print(f'{base_file} does not exist')

    if total == args.num_runs:
        print(f'{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize} is CLEAR')
def get_files_single_baseline(args, comp_baselines, scores, teacher_returns, teacher_actions, student_lr_list, learning_rates, buffer_sizes, batch_sizes, SR, reward_function):

    print('SR', SR)
    
    files = []
    
    for student_lr in student_lr_list:
        for buffer in buffer_sizes:
            for lr in learning_rates:
                for batch in batch_sizes:
                    add_to_files = True
                    if SR == 'buffer_policy':
                        s = f'PE-Actions {buffer} + {lr} + {batch}'
                    if SR == 'buffer_action':
                        print('do I ever get here')
                        s = f'PE-Actions + {buffer} + {lr} + {batch}'
                    if SR == 'buffer_q_table':
                        s = f'PE-QValues {buffer} + {lr} + {batch}'
                    if SR == 'params':
                        s = f"Parameters  + {lr} + {batch}"    
        
                    if SR == 'L2T':
                        s = f'Fan(2018) state + {lr} + {batch}'

                    if SR == 'loss_mismatch':
                        s = f'Huang(2019) state + {lr} + {batch}'




                    if reward_function == 'simple_LP':
                        reward = 'LP'
                        #caption = f'LP + {s}' #f'R^(T) = -1+ LP w/ buffer state rep'
                        #print('caption ahhaha', caption)
                    if reward_function == 'cost':
                        reward = 'Time-to-threshold'
                        #caption = f'Time-to-threshold + {s}'
                        #print('caption', caption)
                    if reward_function == 'L2T':
                        reward = 'Fan(2018) reward'
                    if reward_function == 'binary_LP':
                        reward = 'Huang(2019) reward'
                    if reward_function == 'LP':
                        reward = f'Matiisen(2017) reward'
                    if reward_function == 'target_task_score':
                        reward = f'Ruiz(2019) reward'
                    if reward_function == '0_target_task_score':
                        reward = f'Sparse Ruiz(2019) reward'
                    caption = f'{s} + {reward}'
                    if comp_baselines:
                        add_to_files = False
                    if reward_function == 'simple_LP' and SR == 'buffer_policy':
                    #     caption = f'{caption} (Ours)'
                        if comp_baselines:
                            add_to_files = True
                    if reward_function == 'simple_LP' and SR == 'buffer_q_table':
                        if comp_baselines:
                            add_to_files = True
                    if reward_function == 'simple_LP' and SR == 'buffer_action':
                        if comp_baselines:
                            add_to_files = True
                    if reward_function == 'L2T' and SR == 'L2T':
                        if comp_baselines:
                            add_to_files = True
                    if reward_function == 'cost' and SR == 'params':
                          if comp_baselines:
                            add_to_files = True
                    if reward_function == 'binary_LP' and SR == 'loss_mismatch':
                        if comp_baselines:
                            add_to_files = True






                    #caption = f'{s} + {reward}'
                    



                    if scores:
                        if add_to_files:
                            rootdir = utils.get_rootdir(args, SR)
                            _, subdir = get_subdir(args, rootdir)
                            print('caption found', caption)
                            print('got subdir')
                            if args.student_lr_transfer == False:
                                if 'buffer' in SR:
                                    
                                    #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}_{buffer}'
                                    base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{buffer}_'

                                    print('basefile', base_file)     
                                        
                
                                else:
                                    print('SR is not buffer')
                                    #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}'    
                                    base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_'  
                                    print(base_file)
                            else:
                                if 'buffer' in SR:
                                    
                                    #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}_{buffer}'
                                    base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{buffer}_{student_lr}_student_lr_transfer_'

                                    print('basefile', base_file)     
                                        
                
                                else:
                                    
                                    #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}'    
                                    base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{student_lr}_student_lr_transfer_'                          



                    if teacher_returns:
                        if add_to_files:
                            rootdir = utils.get_rootdir(args, SR)
                            _, subdir = get_subdir(args, rootdir)
                            if 'buffer' in SR:
                                base_file = f'{subdir}/non_normalized_teacher_raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{buffer}_'
                            else:
                                base_file = f'{subdir}/teacher_raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_'
                    files.append((base_file, caption))
    return files

def calculate_area_under_curve(data_list):
    area = np.sum(data_list)
    return area

def get_area_under_curve_all_data(data):
    print('in the get area function')
    hyper_param_dict = {}
    for idx, f in enumerate(list(data.keys())):
        file_description = f
        print(file_description, idx)
        area = calculate_area_under_curve(data[file_description][0])
        hyper_param_dict[file_description] = area
        print('area',area, f)
    #print(hyper_param_dict)
    return hyper_param_dict

def get_max_hyperparam(hyper_param_dict, data):
    print('hyper param dict')
    print(hyper_param_dict)
    max_dict = dict(Counter(hyper_param_dict).most_common(5))
    print('max dict')
    print(max_dict)
    max_dict2 = {}
    print(list(max_dict.keys()))
    for idx, f in enumerate(list(max_dict.keys())):
        print('inside this loop')
        file_description = f
        print(file_description, idx)
        print(data[file_description][0][-20:])
        sum_last_scores = np.sum(data[file_description][0][-80:])
        print('sum_last_scores',sum_last_scores)
        max_dict2[file_description] = sum_last_scores
        
    max_key1 = max(hyper_param_dict, key=hyper_param_dict.get)
    max_key2 = max(max_dict2, key=max_dict2.get)
    print('max_key1', max_key1)
    print('max_key2', max_key2)
    return max_key1

def get_best_files(data): #this is a single baseline as of right now
    hyper_param_dict = get_area_under_curve_all_data(data)
    print(hyper_param_dict)
    max_key = get_max_hyperparam(hyper_param_dict,data)
    print('max_key', max_key)
    best_data = data[max_key]
    return best_data, max_key

def get_best_data_all_baselines(args, scores, teacher_returns, teacher_actions):
    comp_baselines = False
    learning_rates = [.01, .05, .001, .005, .0001, .0005]
    #learning_rates = [.001]
    buffer_sizes = [50,75,100,200,300]
    batch_sizes = [256, 128,64] 

    # state_reps = ['buffer_policy', 'L2T', 'params', 'buffer_q_table', 'buffer_action'] #, 'params', 'tile_coded_params', 'buffer_action'
    # rf = ['simple_LP', 'cost', 'L2T']
    
    state_reps = ['buffer_q_table'] #, 'params', 'tile_coded_params', 'buffer_action'
    rf = ['simple_LP', 'target_task_score']#, 'L2T', 'target_task_score']# 'L2T', 'LP', 'target_task_score', '0_target_task_score']
    envs = ['maze']
    # state_reps = ['buffer_action', 'L2T', 'params']
    # rf = ['simple_LP']
    student_lr_list = [.5]
    #rf = ['cost']
    best_data_dict = {}
    for env in envs:
        for SR in state_reps:
            for reward_function in rf:
                try:
                    files = get_files_single_baseline(args, comp_baselines, scores, teacher_returns, teacher_actions, student_lr_list, learning_rates, buffer_sizes, batch_sizes, SR, reward_function)
                    print(f'Got the files ccc')
                    all_data = get_all_data(files)
                    print(all_data)
                    temp_dict = {}
                    if args.plot_best_data:
                        best_data_values, max_key = get_best_files(all_data)
                        
                        temp_dict[max_key] = best_data_values
                        
                    else:
                        temp_dict = all_data

                    best_data_dict.update(temp_dict)
                except:
                    continue

    return best_data_dict
def plot_single_baseline(args):
    scores = args.comparing_scores
    teacher_returns =not args.comparing_scores
    teacher_actions = False    
    best_data_dict = get_best_data_all_baselines(args, scores, teacher_returns, teacher_actions)

    
    if args.random_curriculum:
        random_file_name = [(f'./RT/{args.env}/random_curriculum/raw_averaged_returns_random', 'Random')]
        random_data = get_all_data(random_file_name)
        best_data_dict['Random'] =random_data.get('Random')
    if args.target_task_only:
        target_file_name = [(f'./RT/{args.env}/target_task_only/raw_averaged_returns_target', 'Target')]
        target_data = get_all_data(target_file_name)
        best_data_dict["Target"] =target_data.get('Target')
    if args.HER:
        HER_file = [(f'{args.rootdir}/evaluation-data/averaged-student-scores/H_average_student_scores', 'HER')]
        HER_data = get_all_data(HER_file)
        best_data_dict["HER"] =HER_data.get('HER')
    #If I want to get the files for each baseline, run get_files in config.py changing the state_rep and just add files into a big ass list
    #files = get_best_files(args, scores, teacher_returns, teacher_actions, buffer, rf)
    print(f'Got the files yyy')
   
    
    # hyper_param_dict = get_area_under_curve_all_data(all_data)
    # # print('here')
    # max_key = get_max_hyperparam(hyper_param_dict)
    # print('max key',max_key)
    #files = get_best_files(args, scores, teacher_returns, teacher_actions, buffer, rf)
    #all_data = get_all_data(files)
        
    
    plot_data(best_data_dict,scores,args)

# 



