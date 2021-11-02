import matplotlib.pyplot as plt
import pickle
import numpy as np

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
            print('trying')
            with open(file_name,'rb') as input:
                data = pickle.load(input)
        except:
            print(f)
            continue
        name = file_description
        all_data.update({name: data})
    return all_data

    


def plot_data(data,scores, files, env, reward_function):

    if scores:
        model_name = 'student_performance'
        y_label = 'Average Discounted Return'
        x_label = 'Student Episode #'
    else:
        model_name = 'teacher_return'
        y_label = 'Average Teacher Return'
        x_label = 'Teacher Training Episode #'
        
    fig = plt.figure()
    num_runs = 15
    colors = ['blue', 'red', 'green', 'purple', 'black', 'orange', 'lawngreen', 'saddlebrown', 'aqua', 'magenta', 'darkcyan', 'peru']
    #colors = ['lightcoral', "indianred", 'firebrick', 'brown', 'forestgreen', 'limegreen', 'green', 'lime', 'navy', 'mediumblue', 'slateblue', 'midnightblue']
    fig = plt.figure()
    
    for idx, f in enumerate(files):
        file_description = f[1]
        print(file_description)
        try:
            mean = data[file_description][0]
            std = data[file_description][1]
            
            length = len(mean)
            plt.plot(np.arange(length), mean, lw = 2, color = colors[idx], label = file_description)
            variance = std/np.sqrt(num_runs)

            plt.fill_between(np.arange(length), mean+variance, mean-variance, facecolor=colors[idx], alpha=0.2)
        except:
            continue
    plt.legend()

    plt.title(f'LP Reward Function')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    
    fig.set_figwidth(8)
    fig.set_figheight(5)
    plt.savefig(f'{model_name}_comp_LP_four_rooms.png')


def get_files(file_type, state_reps, reward_function_types):
    files = []
    rootdir = '/Users/cmuslimani/Projects/Curriculum_MDP/Tabular/debug/four_rooms/new-reward-function/log-based/optimal/easy-initalization/fixed-student-seed'
    batch_size = 64
    lr = 0.001
    alpha = 1.25
    for reward_type in reward_function_types:

        for i in range(0, len(state_reps)):
            if file_type == 'teacher_return':
                file = f'{rootdir}/{state_reps[i]}/teacher-returns/returns_list_{reward_type}_{batch_size}_{lr}_{state_reps[i]}_{alpha}'
                print(file)
                print('\n')
            else:
                file = f'{rootdir}/{state_reps[i]}/evaluation-data/evaluation_average_score_{reward_type}_{batch_size}_{lr}_{state_reps[i]}_{alpha}'
            files.append((file,state_reps[i] +':' +reward_type))

    print(files)
    return files

#state_reps_LP_cost = ['action_return','policy_table', 'policy_table_SF',  'q_matrix_SF', 'q_matrix']
#state_reps_cost= ['q_matrix', 'q_matrix_action_return', 'policy_table', "policy_table_action_return"]
state_reps = ['policy_table', 'q_matrix', 'action_return', 'action_return_reward_time_step_SF'] #'action_return','policy_table_SF',  'q_matrix_SF', 'q_matrix',
reward_function_types = [ "LP"] #,'cost
files = get_files('teacher_return', state_reps,reward_function_types)
all_data = get_all_data(files)
print(all_data.keys())
scores = False
env = 'Four Rooms'
plot_data(all_data,scores, files, env, reward_function_types[0])

# rootdir = '/Users/cmuslimani/Projects/Curriculum_MDP/Tabular/debug/maze/new-reward-function/log-based/optimal/easy-initalization/fixed-student-seed'

# actions = get_data(f'{rootdir}/q_matrix/evaluation-data/all_teacher_actions_LP_cost_64_0.001_q_matrix_1.25')
# teacher_scores = get_data(f'{rootdir}/q_matrix/evaluation-data/all_teacher_score_LP_cost_64_0.001_q_matrix_1.25')
# print(teacher_scores)
# print(actions[10][0:10])