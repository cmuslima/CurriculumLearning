import matplotlib.pyplot as plt
import pickle
import numpy as np

def get_data(file):

    with open(file,'rb') as input:
        data = pickle.load(input)
    return data
    



def get_score_data(files):
    scores_data = dict()
  

    for f in files:
        file_name = f[0]
        file_description = f[1]
        with open(file_name,'rb') as input:
            data = pickle.load(input)
        name = file_description
        scores_data.update({name: data})
    return scores_data

    


def plot_data(data,scores, files):

    if scores:
        model_name = 'teacher return'
        
    fig = plt.figure()
    num_runs = 15
    colors = ['blue', 'red', 'green', 'purple', 'black', 'orange']

    LR = 0.05
    batch = 20
    num_files = 3
    fig = plt.figure()
    
    for idx, f in enumerate(files):
        file_description = f[1]
        print(file_description)
        print('mean')
        mean = data[file_description][0]
       
        print(mean)
        std = data[file_description][1]
        print('std')
        
     
       
        print('\n')
       
        length = len(mean)
        plt.plot(np.arange(length), mean, lw = 2, color = colors[idx], label = file_description)


        variance = std/np.sqrt(num_runs)

        plt.fill_between(np.arange(length), mean+variance, mean-variance, facecolor=colors[idx], alpha=0.2)
    plt.legend()
    plt.title('Maze')
    plt.ylabel('Average Teacher Return')
    plt.xlabel('Teacher Training Episode')
    fig.set_figwidth(11)
    fig.set_figheight(8)
    plt.savefig(f'{model_name}_comp.png')



score_files = []
rootdir = '/Users/cmuslimani/Projects/Curriculum_MDP/Tabular/debug/maze/old-reward-function/optimal/easy-initalization'
state_reps = ['policy_table', 'q_matrix', 'policy_table_action_return', 'q_matrix_action_return']
for i in range(0, 4):
    file = f'{rootdir}/{state_reps[i]}/teacher-returns/returns_list_64_0.001_{state_reps[i]}'
    score_files.append((file,state_reps[i]))


scores_data = get_score_data(score_files)

scores = True
plot_data(scores_data,scores, score_files)


