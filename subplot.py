import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from rendering_envs import render_env
from plotting_graphs import normalize

from scipy import stats
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

    


def plot_data(all_data, scores, env):
    print('here')
    normalized = False
    
    subdir = 'plots'
    print(f'In the plot data function')

    if scores:
        model_name = f'all_env_comp_SP1'
        y_label = 'Average Student \'s Return'

        title = 'Student Performance on Target Task'
  
    else:
        model_name = f'normalized_teacher_return_max'
        y_label = 'Teacher Return'
        x_label = 'Number of Teacher Episodes'
        title = 'Normalized Teacher Return'
    
    fig, axs = plt.subplots(1, 2)
    num_runs = 5
    colors = ['blue', 'indianred', 'green', 'purple', 'red', 'blue', 'magenta', 'saddlebrown', 'aqua', 'magenta', 'darkcyan', 'peru','lightcoral', "indianred", 'firebrick', 'midnightblue', 'forestgreen', 'lime',\
         'navy', 'mediumblue', 'slateblue', 'silver','grey', 'blueviolet' ]
    #colors = ['lightcoral', "indianred", 'firebrick', 'brown', 'forestgreen', '√', 'green', 'lime', 'navy', 'mediumblue', 'slateblue', 'midnightblue']
   
    count = 0
    for i in range(0, len(all_data)):
        for idx, f in enumerate(list(all_data[i].keys())):
            file_description = f
        
            try:
                
                if scores:
                    if i == 1:
                        mean = all_data[i][file_description][0][0:60]
                        std = all_data[i][file_description][1][0:60]
                    else:
                        mean = all_data[i][file_description][0]
                        std = all_data[i][file_description][1]
                else:
                    if normalized:

                        mean = stats.zscore((all_data[i][file_description][0]))
                    
                        #print(data[file_description][0][0])
                        std = stats.zscore((all_data[i][file_description][1]))
                
                    else:
                        mean = all_data[i][file_description][0][0:200]
                        std = all_data[i][file_description][1][0:200]
                length = len(mean)

                axs[i].plot(np.arange(length), mean, lw = 2, color = colors[idx], label = file_description)
                variance = std/np.sqrt(num_runs)

                axs[i].fill_between(np.arange(length), mean+variance, mean-variance, facecolor=colors[idx], alpha=0.2)
                count+=1
            except:
        
                continue
            # if i == 0:
            #     axs[0].plot(np.arange(length), [.6]*length,color='black', linestyle='dashed')
            # else:
            #     axs[1].plot(np.arange(length), [(.99)**35]*length,color='black', linestyle='dashed')

    #plt.legend(loc = 'lower right')
    if scores:
        axs[0].set_xlabel('Number of Student Episodes (x25)', fontsize = 18)
        axs[1].set_xlabel(xlabel='Number of Student Episodes (x10)', fontsize = 18)
        axs[0].set_ylabel(y_label, fontsize = 18)
    else:
        axs[0].set_xlabel('Number of Teacher Episodes', fontsize = 18)
        axs[1].set_xlabel(xlabel='Number of Teacher Episodes', fontsize = 18)
        axs[0].set_ylabel(y_label, fontsize = 18)   
    axs[0].set_title('LP',fontsize = 25)
    axs[1].set_title('Time-to-threshold', fontsize = 25)  
    #axs[2].set_title('Parameters', fontsize = 25)  
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    lg = plt.legend(lines[0:3], labels[0:3], bbox_to_anchor=(1, 1.0), loc='best', fontsize = 'large')
    
    fig.set_figwidth(20)
    fig.set_figheight(6)
    
    #plt.title(f'{title}')
    #plt.ylabel(y_label)
    #plt.xlabel(x_label)
    # x_labels = ['0', '200', '400', '600', '800'] 
    # x_ticks = [0,20,40,60,80]
    # plt.xticks(ticks=x_ticks, labels=x_labels)
    fig.set_figwidth(22)
    fig.set_figheight(8)
    fig.savefig(f'comp_return.png', bbox_extra_artists=(lg,), bbox_inches='tight')





def get_best_files(scores, teacher_returns, teacher_actions, subdir, rootdir, best_params):
    normalized = False
    
    files = []

    # best_params = [('cost', 'buffer_policy', 256, .001, 75), ('cost', 'buffer_q_table', 256, .005, 50), ('cost', 'q_matrix', 128, .001, 50),\
    #     ('simple_LP', 'buffer_policy', 128, .001, 75), ('simple_LP', 'buffer_q_table', 128, .01, 75), ('simple_LP', 'q_matrix', 256, .0001, 50),\
    #     ] #(None, "Random", None, None, None), (None, 'Target-Only',None, None, None)
    # best_params = [('simple_LP', 'buffer_policy', 128, .001, 75),(None, "Random", None, None, None), (None, 'Target-Only',None, None, None)]

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
        if state_rep == 'q_matrix' or state_rep == 'params':
            s = "Parameters"       
        if rf == 'simple_LP':
            caption = f'LP + {s}' #f'R^(T) = -1+ LP w/ buffer state rep' #LP +\n
        if rf == 'cost':
            caption = f'Time-to-threshold + {s}'#Time-to-threshold +
            print(caption)
        if rf == None:
            caption = state_rep
        if scores:
           
            #print('state rep', state_rep)
            if state_rep == 'buffer_q_table' or state_rep == 'buffer_policy':
                base_file = f'{rootdir}/{state_rep}/{subdir}/evaluation_average_score_{rf}_{b}_{lr}_{state_rep}_{buffer}'
                #print('basefile', base_file)     
                #print('here')      
            if state_rep == 'Random' or state_rep == "Target-Only":
                base_file = f'{rootdir}/{state_rep}/evaluation-data/evaluation_average_score'
                print('here 2')
            if state_rep == 'q_matrix' or state_rep == 'params':
                print('here3')
                base_file = f'{rootdir}/{state_rep}/{subdir}/evaluation_average_score_{rf}_{b}_{lr}_{state_rep}'    


        if teacher_returns:
            subdir2 = 'averaged-returns'
            returns = 'raw_averaged_returns'
            if state_rep == 'buffer_q_table' or state_rep == 'buffer_policy':
                base_file = f'{rootdir}/{state_rep}/{subdir2}/{returns}_{rf}_{b}_{lr}_{state_rep}_{buffer}'           
            else:
                base_file = f'{rootdir}/{state_rep}/{subdir2}/{returns}_{rf}_{b}_{lr}_{state_rep}'    
        if teacher_actions:
            subdir = 'evaluation-data'
            buffer = 25
            
            if state_rep == 'buffer_q_table' or state_rep == 'buffer_policy':

                base_file = f'{rootdir}/{state_rep}/{subdir}/teacher_actions_{rf}_{b}_{lr}_{state_rep}_{buffer}'           
            else:
                base_file = f'{rootdir}/{state_rep}/{subdir}/teacher_actions_{rf}_{b}_{lr}_{state_rep}'           

          
    
        print(f'Base file = {base_file}, caption = {caption}')
        files.append((base_file, caption))
    return files


def plotting():
    scores = False
    teacher_returns = True
    teacher_actions = False    
    rootdir = './experiment-data/maze'
    subdir = 'teacher-returns'
    #subdir = 'evaluation-data'
    #rootdir = './rl-starter-files-master/four_rooms/easy-initalization'

    # best_params = [('simple_LP', 'buffer_q_table', 128, .01, 75), ('cost', 'buffer_q_table', 256, .005, 50), ('simple_LP', 'buffer_policy', 128, .001, 75), ('cost', 'buffer_policy', 256, .001, 75),
    #      ('simple_LP', 'q_matrix', 256, .0001, 50),('cost', 'q_matrix', 128, .001, 50)] 
    # subdir = 'evaluation-200'
    # rootdir = './experiment-data/maze'

    # tabular_files = get_best_files(scores, teacher_returns, teacher_actions,subdir, rootdir, best_params)

    #tabular_results = get_all_data(tabular_files)

    #best_params = [('simple_LP', 'buffer_q_table', 128, .01, 25),('simple_LP', 'buffer_policy', 64, .01, 25), \
        #('simple_LP', 'params', 128, .01, None)]
    #('cost', 'buffer_q_table', 128, .01, 25)]
    #('simple_LP', 'buffer_policy', 64, .01, 25)
    

    #this param is for maze
    best_params = [('simple_LP', 'buffer_q_table', 128, .01, 75), ('simple_LP', 'buffer_policy', 128, .001, 75),('simple_LP', 'q_matrix', 256, .0001, 50)] 

    #best_params= [('simple_LP', 'buffer_q_table', 128, .01, 25), ('simple_LP', 'buffer_policy', 64, .01, 25),('simple_LP', 'params', 128, .01, None)]


    FA_files = get_best_files(scores, teacher_returns, teacher_actions,subdir, rootdir, best_params)
    FA_results = get_all_data(FA_files)
   
    #this is for for rooms
    #best_params= [('cost', 'buffer_q_table', 128, .01, 25), ('cost', 'buffer_policy', 64, .01, 25),('cost', 'params', 128, .01, None)]
    
    #this param is for maze
    best_params = [('cost', 'buffer_q_table', 256, .005, 50), ('cost', 'buffer_policy', 256, .001, 75),('cost', 'q_matrix', 128, .001, 50)]


    
    FA_files2 = get_best_files(scores, teacher_returns, teacher_actions,subdir, rootdir, best_params)
    FA_results2 = get_all_data(FA_files2)

    # best_params= [('simple_LP', 'params', 128, .01, None), ('cost', 'params', 128, .01, None)]

    # subdir = 'evaluation-data'
    # rootdir = './rl-starter-files-master/four_rooms/easy-initalization'

    # FA_files3 = get_best_files(scores, teacher_returns, teacher_actions,subdir, rootdir, best_params)
    # FA_results3 = get_all_data(FA_files3)
    env = 'maze'
    all_results = [FA_results, FA_results2]
    plot_data(all_results, scores, env)



def plot_actions():
    scores = False
    teacher_returns = False
    teacher_actions = True

    #subdir = 'evaluation-data'
    #rootdir = './rl-starter-files-master/four_rooms/easy-initalization'
    best_params= [('simple_LP', 'params', 128, .01, 25)]

    files = get_best_files(scores, teacher_returns, teacher_actions, subdir, rootdir, best_params)
    print(files)
    all_data = get_all_data(files)
    
    #for exp in list(all_data.keys())

    interval1 = []
    interval2 = []
    interval3 = []
    curr = all_data[f'LP + Parameters']
    print(curr[0])
    print(curr[1])
    print(curr[2])
    print(curr[3])
    print(curr[4])
    for seed in range(0,5):

        interval1.append(curr[seed][0:4])
        interval2.append(curr[seed][4:8])
        interval3.append(curr[seed][8:50])
        
    print(interval1)
    flat_interval1 = [item for sublist in interval1 for item in sublist]
    flat_interval2 = [item for sublist in interval2 for item in sublist]
    flat_interval3 = [item for sublist in interval3 for item in sublist]
    all_intervals = [flat_interval1, flat_interval2, flat_interval3]
    #print()
    for i in all_intervals:
        raw_freqs = np.bincount(i)
        probability = raw_freqs/sum(raw_freqs)
        #print(raw_freqs, probability)
        render_env(probability)
    # return probability
    #print(all_data)
    # for i in actions:
    #     plt.scatter(np.arange(0, len(actions[0])), i)
    # plt.show()
    

    #     actions = get_data('./action_return/evaluation-data/all_teacher_scores_simple_LP_256_0.005_buffer_policy_75')
    #     print(actions[0])
    
    #     for i in actions[0]:
    #         plt.scatter(np.arange(0, len(actions[0])), i)
    #     plt.show()


plotting()
#plot_actions()