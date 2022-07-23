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
        elif args.env == 'fetch_push':
            title = 'Student Performance in Fetch Push'

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
        title = f'Generalization to Differing Student LRs.' 
    if args.student_NN_transfer:
        title = f'Generalization to Students with Differing NN' 
    return title
def get_axis_tiles(args):
    if args.comparing_scores:
        
        if args.env == 'fetch_reach_3D_outer':
            x_label = 'Number of Episodes (x6)'
            y_label = 'Student Success Rate'
        if args.env == 'fetch_push':
            x_label = 'Number of Episodes'
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
        


    # else:
    #     if args.student_transfer == False and args.student_lr_transfer == False:
    #         dir = f'./RT/{args.env}/thesis_plots'
    
    #     if args.student_transfer:
    #         folder = 'q_learning_to_sarsa_transfer'
    #         dir = f'./RT/{args.env}/single_student/{args.student_type}/{folder}/plots'
    #     if args.student_lr_transfer:
    #         folder = f'lr_{args.student_lr}'
    #         dir = f'./RT/{args.env}/single_student/{args.student_type}/{folder}/plots'

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
    fig = plt.figure()
    
    colors = ['magenta', 'black', 'green', 'aqua', 'red', 'blue', 'saddlebrown', 'saddlebrown', 'aqua', 'magenta', 'darkcyan', 'peru','lightcoral', "indianred", 'firebrick', 'midnightblue', 'lime',\
         'navy', 'mediumblue', 'slateblue', 'silver','grey', 'blueviolet' ]
    #colors = ['lightcoral', "indianred", 'firebrick', 'brown', 'forestgreen', '√', 'green', 'lime', 'navy', 'mediumblue', 'slateblue', 'midnightblue']
   
    #order = [0,2, 1, 3, 6, 5, 7,8, 4]
    print('data list')
    print(list(data.keys()))
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
            # if args.env == 'fetch_reach_3D_outer':
            #     mean = data[file_description][0][0]
                
            #     std = data[file_description][1][0]
            # else:
            mean = data[file_description][0]
            std = data[file_description][1]
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
                    #marker = 'o'
                    color = 'purple'
                    print('using', caption, file_description)
                    if 'student_lr_transfer' in file_description:
                        print(file_description)
                        student_lr = file_description[0:8]
                        print('student_lr in plot data', student_lr)
                        caption = f'PE-Actions + LP (Ours) + {student_lr}'
                        print('caption', caption)
                        if '0.25' in student_lr: 
                            print('0.25', student_lr)
                            marker = '*'
                        else:
                            print('0.01', student_lr)
                            marker = '+'
                    if 'sarsa_transfer' in file_description:
                        caption = f'PE-Actions + LP (Ours): SARSA student'
                        
                if 'PE-QValues' in file_description and 'LP' in file_description:
                    caption = 'PE-QValues + LP (Ours)'
                    
                    color = 'orange'
                    #marker = 'o'
                    print('using caption', caption, file_description)
                    if 'student_lr_transfer' in file_description:
                        student_lr = file_description[0:8]
                        print('student_lr in plot data', student_lr)
                        caption = f'PE-QValues + LP (Ours) + {student_lr}'
                        print('caption', caption)
                        if '0.25' in student_lr: 
                            marker = '*'
                        else:
                            marker = "+"
                    if 'sarsa_transfer' in file_description:
                        caption = f'PE-QValues + LP (Ours): SARSA student'
                    
                if 'Fan(2018) state' in file_description and 'Fan(2018) reward' in file_description:
                    caption = 'Fan(2018)'
                    print('using caption fan', caption, file_description)
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
        if 'Diff. NN' in file_description:
            caption = 'PE-QValues + LP (Ours): Diff. NN'
            marker = 'o'
            color = 'orange'



        plt.plot(np.arange(length), mean, lw = 2, color = color, marker = marker, label = caption)
        CI = 1.96
        num_runs = args.num_runs


        
        variance = (CI*std)/np.sqrt(num_runs)

        plt.fill_between(np.arange(length), mean+variance, mean-variance, facecolor=color, alpha=0.2)

    if scores:
        if args.env == 'maze':
            plt.axhline(y=0.77, color='black', linestyle='--', label = 'Performance Threshold')
        if args.env == 'four_rooms':
            plt.axhline(y=0.6, color='black', linestyle='--', label = 'Performance Threshold')
        if args.env == 'fetch_reach_3D_outer':
            plt.axhline(y=0.9, color='black', linestyle='--', label = 'Performance Threshold')
    if scores == False:
        
        if args.env == 'maze':
            plt.vlines(x=150, ymin = -35, ymax = -10, color='black', linestyle='--', label = 'Approx. convergence point')
        if args.env == 'four_rooms':
            plt.vlines(x=40, ymin = -10, ymax = -18, color='black', linestyle='--', label = 'Approx. convergence point')
        if args.env == 'fetch_reach_3D_outer':
            plt.vlines(x=25, ymin = -23, ymax = -13, color='black', linestyle='--', label = 'Approx. convergence point')
    #handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(loc='upper left')#
    #plt.legend(loc='best', fontsize=40)#

    #plt.tight_layout()
    #lg = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.05, 1.0), fontsize = 20)
    #plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1., 1.0), loc='upper left')

    plt.title(title, fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    plt.xlabel(x_label, fontsize=30)
    # x_labels = ['0', '200', '400', '600', '800'] 
    # x_ticks = [0,20,40,60,80]
    # plt.xticks(ticks=x_ticks, labels=x_labels)
    fig.set_figwidth(13)
    fig.set_figheight(10)
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



def get_subdir(args,rootdir, student_lr = None):
    print('inside get_subdir')
    prev_dir =f'{rootdir}/evaluation-data'
    print('prev dir', prev_dir)
    if args.comparing_scores:
        new_dir = f'{rootdir}/evaluation-data'
    else:
        new_dir = f'{rootdir}/teacher-data'

    if args.student_transfer: 
        folder = 'transfer_q_learning_to_sarsa'
        new_dir = f'{rootdir}/evaluation-data/{folder}'
        #prev_dir =f'{rootdir}/evaluation-data/{folder}'
    if args.student_lr_transfer:
        folder = f'lr_transfer_{student_lr}'
        new_dir = f'{rootdir}/evaluation-data/{folder}'
        #prev_dir =f'{rootdir}/evaluation-data'
    print('prev dir',prev_dir, 'new dir', new_dir)
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
def get_files_single_baseline(args, comp_baselines, scores, teacher_returns, teacher_actions, learning_rates, buffer_sizes, batch_sizes, SR, reward_function):
    print('inside the get_files_single_baseline')
    print('SR', SR)
    
    files = []
    
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

                caption = f'{reward} + {s}'
                print('caption', caption)
                if scores:
                    if add_to_files:
                        print('im here')
                        rootdir = utils.get_rootdir(args, SR)
                        print('rootdir', rootdir)
                        old_dir, subdir = get_subdir(args, rootdir)
                        print('old dir', subdir)
                        print('blah')
                    
                        print('got subdir', old_dir)
                        #if args.student_lr_transfer == False and args.student_transfer == False:
                        if 'buffer' in SR:
                            
                            #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}_{buffer}'
                            base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{buffer}_'

                            print('basefile', base_file)     
                                
        
                        else:
                            print('SR is not buffer')
                            #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}'    
                            base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_'  
                            print(base_file)
    

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
def get_transfer_files(args, comp_baselines, scores, teacher_returns, teacher_actions, student_lr, learning_rates, buffer_sizes, batch_sizes, SR, reward_function):

    print('SR', SR)
    
    files = []
    
    
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

                
                if args.student_lr_transfer:
                    caption = f'{student_lr}       + student_lr_transfer + {s}+ {reward} + {buffer} + {lr} + {batch}'
                    caption_dict = {'type': 'student_lr_transfer', 'state': s, 'reward': reward, 'student_lr': student_lr, 'buffer': buffer, 'lr':lr, 'batch':batch}
                if args.student_transfer:
                    caption = f'student_sarsa_transfer + {s}+ {reward} + {buffer} + {lr} + {batch}'
                    caption_dict = {'type': 'sarsa_transfer', 'state': s, 'reward': reward, 'buffer': buffer, 'lr':lr, 'batch':batch}
                    print(caption, 'caption')

                if scores:
                    if add_to_files:
                        print('im here')
                        rootdir = utils.get_rootdir(args, SR)
                        print('rootdir', rootdir)
                        old_dir, subdir = get_subdir(args, rootdir, student_lr)
                        print('caption found', caption)
                        print('got subdir', old_dir)
                        if args.student_lr_transfer == True and args.student_transfer == False:
                            if 'buffer' in SR:
                                
                                #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}_{buffer}'
                                base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{buffer}_{student_lr}_'

                                print('transfer lr basefile', base_file)     
                                    
            
                            else:
                                
                                #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}'    
                                base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{student_lr}_student_lr_transfer_'                          
                        if args.student_lr_transfer == False and args.student_transfer == True:
                            if 'buffer' in SR:
                                
                                #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}_{buffer}'
                                base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{buffer}_sarsa_'

                                print('basefile', base_file)     
                                    
            
                            else:
                                
                                #base_file = f'{subdir}/average_student_scores_{reward_function}_{batch}_{lr}_{S}'    
                                base_file = f'{subdir}/raw_averaged_returns_{SR}_{reward_function}_{lr}_{batch}_{student_lr}_student_lr_transfer_'   



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
    comp_baselines = True
    learning_rates = [.001, .005]
    #learning_rates = [.001]
    buffer_sizes = [75, 100,200,300]
    batch_sizes = [64,256] 

    state_reps = ['buffer_action'] #, 'params', 'tile_coded_params', 'buffer_action'
    #rf = ['simple_LP']
    
    # state_reps = ['buffer_action']
    rf = [ 'simple_LP'] #'target_task_score', '0_target_task_score', 'L2T', 'cost', "LP",
    envs = ['maze']
    # state_reps = [ 'L2T']
    # rf = ['L2T']

    best_data_dict = {}
    all_data = {}
    for env in envs:
        for SR in state_reps:
            for reward_function in rf:
                
                try:
                    baseline_files = get_files_single_baseline(args, comp_baselines, scores, teacher_returns, teacher_actions, learning_rates, buffer_sizes, batch_sizes, SR, reward_function)
                    print(f'Got the files ccc')
                    baseline_data = get_all_data(baseline_files)
                    print(baseline_data)

                    all_data = baseline_data 
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
def get_best_transfer_data(args, scores, teacher_returns, teacher_actions):
    comp_baselines = True
    learning_rates = [.01, .05, .001, .005, .0001, .0005]
    #learning_rates = [.001]
    buffer_sizes = [75, 100,200,300]
    batch_sizes = [256,128,64] 

    state_reps = ['buffer_action', "L2T", 'params'] #'buffer_q_table', 'buffer_policy'#, 'params', 'tile_coded_params', 'buffer_action'
    rf = ['simple_LP', 'L2T', 'cost']
    

    envs = ['maze']
  
    student_lr_list = [.0001, .25]
    
    best_data_dict = {}
    all_data = {}
    plot_best_data = True
    for student_lr in student_lr_list:
        for SR in state_reps:
            for reward_function in rf:
                
                # try:
                if args.student_lr_transfer or args.student_transfer:
                    
                    transfer_files = get_transfer_files(args, comp_baselines, scores, teacher_returns, teacher_actions, student_lr, learning_rates, buffer_sizes, batch_sizes, SR, reward_function)
                    transfer_data = get_all_data(transfer_files)
                    print('transfer data', transfer_data)
                temp_dict = {}
                if plot_best_data:
                    best_data_values, max_key = get_best_files(transfer_data)
                    
                    temp_dict[max_key] = best_data_values
                    
                else:
                    temp_dict = transfer_data

                best_data_dict.update(temp_dict)
                # except:
                #     continue

    return best_data_dict
def plot_single_baseline(args):
    scores = args.comparing_scores
    teacher_returns =not args.comparing_scores
    teacher_actions = False    
    best_data_dict = {}
    best_data_dict = get_best_data_all_baselines(args, scores, teacher_returns, teacher_actions)
    #print(best_data_dict)
    if args.student_lr_transfer or args.student_transfer:
        best_transfer_data = get_best_transfer_data(args, scores, teacher_returns, teacher_actions)
        best_data_dict.update(best_transfer_data)
    
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
    if args.student_NN_transfer:

        new_dir = f'{args.rootdir}/evaluation-data/NN_large_64_64'
        print('args.rootdir', args.rootdir, new_dir)
        transfer_file = [(f'{new_dir}/raw_averaged_returns_buffer_q_table_simple_LP_0.001_128_100_', 'Diff. NN')]
        
        transfer_data = get_all_data(transfer_file)
        best_data_dict["Diff. NN"] =transfer_data.get('Diff. NN')
    print(best_data_dict)
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
def determine_normal_dis(args):
    import statsmodels.api as sm
    from scipy.stats import norm
    import pylab
    import math
    model_name = f'./RT/plots/{args.env}/area_under_curve_{args.SR}_{args.reward_function}_{args.env}'
    mean_data = get_data(model_name)
    keys = list(mean_data.keys())
    mean_data = mean_data[keys[0]]
    log_mean_data = []
    for v in mean_data:
        if v == 0:
            v = 0.01
        log_mean_data.append(math.log(v))
    #print(keys)
    plt.hist(mean_data, label = args.reward_function)
    plt.show()
    shapiro_test1 = stats.shapiro(log_mean_data)
    shapiro_test2 = stats.shapiro(mean_data)
    
    p_value_log = shapiro_test1.pvalue
    p_value_reg = shapiro_test2.pvalue
    
    if p_value_log >= .05:
        print(f'{args.reward_function} log is normal')
    else:
        print(f'{args.reward_function} log is NOT normal')
        print(f'p value = {p_value_log}')
    if p_value_reg >= .05:
        print(f'{args.reward_function} reg is normal')
    else:
        print(f'{args.reward_function} reg is NOT normal')
        print(f'p value = {p_value_reg}')
    #sm.qqplot(np.array(mean_data), line = "s")
    #pylab.show()

    #plt.boxplot(mean_data)
    #plt.show()
def plot_actions(args):
    scores = False
    teacher_returns = False
    teacher_actions = True

    
    #files = get_best_files(args, scores, teacher_returns, teacher_actions, buffer, rf)
    #print(files)
    #all_data = get_all_data(files)
    #colors = ['black', 'blue', 'red', 'green', 'orange', 'purple']
    #for exp in list(all_data.keys())
    file = f'{args.rootdir}/evaluation-data/teacher_actions_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_'
    print(file)
    data = get_data(file)
    print(len(data))
    print(len(data[0]))
    interval1 = []
    interval2 = []
    interval3 = []
    bad_count = 0
    total_count = 0 
    for seed in range(0,30):
        curr = data[seed]
    
        interval1.append(curr[0:5])
        interval2.append(curr[5:10])
        interval3.append(curr[10:])
        print(curr )
        for c in curr:
            total_count+=1
            if c == 1 or c == 2 or c == 3 or c == 8 or c == 8 or c == 10:
                bad_count+=1
    print('% of irrelevant/impossible tasks', bad_count/total_count)
    # flat_interval1 = [item for sublist in interval1 for item in sublist]
    # flat_interval2 = [item for sublist in interval2 for item in sublist]
    # flat_interval3 = [item for sublist in interval3 for item in sublist]
    # all_intervals = [flat_interval1, flat_interval2, flat_interval3]
    
    # for i in all_intervals:
    #     raw_freqs = np.bincount(i)
    #     probability = raw_freqs/sum(raw_freqs)
    #     #print(raw_freqs, probability)
    #     render_env(probability)
    return probability


def t_testing(args):
    teacher = False
    from scipy import stats
    baseline_1 = ['buffer_q_table', 'simple_LP']
    baseline_2 = ['target_task_only', 'target_task_only']
    baselines = [baseline_1, baseline_2]
    all_data = []
    for baseline in baselines:
        SR = baseline[0]
        reward = baseline[1]
        if teacher:
            model_name = f'./RT/plots/{args.env}/teacher_area_under_curve_{SR}_{reward}_{args.env}'
        else:
            model_name = f'./RT/plots/{args.env}/area_under_curve_{SR}_{reward}_{args.env}'
            if reward == 'random':
                model_name = f'./RT/plots/{args.env}/area_under_curve_random'
            if reward == 'target_task_only':
                model_name = f'./RT/plots/{args.env}/area_under_curve_target'
            area_under_curve = utils.get_data(model_name)
            key = list(area_under_curve.keys())
            key = key[0]
            all_data.append(area_under_curve[key])

    mean_baseline_1 = np.mean(all_data[0])
    mean_baseline_2 = np.mean(all_data[1])
    std_baseline_1 = np.std(all_data[0])
    std_baseline_2 = np.std(all_data[1])
    print(baseline_1,'mean =', mean_baseline_1, baseline_2,'mean =', mean_baseline_2)
    print(baseline_1,'std =',std_baseline_1, baseline_2, 'std =', std_baseline_2)
    print(stats.ttest_ind(all_data[0], all_data[1], equal_var=False, alternative = 'greater'))
    calculate_standard_error(std_baseline_1, args, baseline_1)
    calculate_standard_error(std_baseline_2, args, baseline_2)

def calculate_standard_error(std, args, name):

    n = args.num_runs
    print(f'using n = {n}')
    standard_error = std/np.sqrt(n)
    print(f'standard_error for {name} = {standard_error}')



def action(args):
    SR = args.SR
    reward_function = args.reward_function
    lr = args.teacher_lr
    batch = args.teacher_batchsize
    buffer = args.teacher_buffersize

    rootdir = utils.get_rootdir(args, SR)
    _, subdir = get_subdir(args, rootdir)
    
    for seed in range(0,10):
        print(f'run = {seed}')
        base_file = f'{subdir}/teacher_action_list_{SR}_{reward_function}_{lr}_{batch}_{buffer}_{seed}'
        actions =get_data(base_file)
        print('len of actions', len(actions))
        print('actions', actions[0:25])

