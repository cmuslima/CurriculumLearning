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
        model_name = 'scores'
        y_axis_label = 'Average Return'
  
    fig = plt.figure()
    num_runs = 9
    colors = ['blue', 'red', 'green', 'purple', 'black', 'orange', 'pink', 'yellow', 'grey']

    LR = 0.05
    batch = 20
    num_files = 3
    fig = plt.figure()
    
    for idx, f in enumerate(files):
        file_description = f[1]
        print(file_description)
        #print('mean')
        mean = data[file_description][0]
       
        std = data[file_description][1]
        #print('std')
        
        
        length = len(mean)
        plt.plot(np.arange(length), mean, lw = 2, color = colors[idx], label = file_description)


        variance = std/np.sqrt(num_runs)

        plt.fill_between(np.arange(length), mean+variance, mean-variance, facecolor=colors[idx], alpha=0.2)
    plt.legend()
    #plt.title('Larger Action Set')
    plt.ylabel('Average Student\'s Return')
    plt.xlabel('Student Trial #')
    fig.set_figwidth(8)
    fig.set_figheight(6)
    plt.savefig(f'{model_name}_expanded_fourrooms2.png')

# ('evaluation_average_score_new_64_0.001_SF', 'LP-SF Reward Function (Ours)')
#('target_evaluation_average_score_final', 'No Teacher')
score_files  = [('evaluation_average_score_old_64_0.001_action_return', '-1 signal action_return'),('evaluation_average_score_new_128_0.001_SF_0', '-1 Reward Signal Only'), ('evaluation_average_score_new_128_0.001_SF_1.5', 'ALP and Cost signals'), ('random_evaluation_average_score_final', 'Random'), ('target_evaluation_average_score_final', 'No Teacher')]
# score_files = []
# alpha_list = [0, 1, 2, 1.75, 1.5, 1.25]
# BATCHSIZE = [128] #, 64, 128]
# for batch in BATCHSIZE:
#     for alpha in alpha_list:
#         file_name = f'evaluation_average_score_new_{batch}_0.001_SF_{alpha}'
#         file_key = f'{batch} {alpha}'
#         score_files.append((file_name, file_key))
# results = list()

actions = get_data('all_teacher_actions_new_128_0.001_SF_0')
print(actions)
# all = []

# num_irre = 0
# tot = 0

# for i in range(0, 9):
#     for value in actions[i][0:20]:
#         tot+=1
#         if value >=5:
#             num_irre+=1
    
# print(num_irre/tot)

# plt.hist(all)
# plt.ylabel('Frequency Count')
# plt.xlabel('Selected Tasks during student episodes 15-20')
# plt.savefig(f'expanded_4rooms_15_20.png')
# plt.show()
scores_data = get_score_data(score_files)

scores = True
plot_data(scores_data,scores, score_files)

# LR = [.05]
# BATCH = [20,30,40,50]
# colors = ['red', 'blue', 'orange', 'black']
# for lr in LR:
#     for idx, batch in enumerate(BATCH):
#         returns = get_data(f'returns_list_old_reward_function_{batch}_{lr}')
#         plt.plot(returns[0], color = colors[idx])
# plt.show()


# BATCHSIZE = [64]
# LR = [.001]
# colors = ['orange', 'blue', 'red', 'green','black', 'purple', 'pink', 'grey']
# i = 0
# for batch in BATCHSIZE:
#     for lr in LR:
#         returns1 = get_data(f'evaluation_average_score_old_{batch}_{lr}_action_return2')

#         label = f'batch = {batch} lr = {lr}'
#         plt.plot(returns1[0], color = colors[i], label = label)

#         variance = returns1[1]/np.sqrt(30)

#         plt.fill_between(np.arange(100), returns1[0]+variance, returns1[0]-variance, facecolor=colors[i], alpha=0.2)
#         i+=1
#         plt.legend()
# returns1 = get_data(f'evaluation_average_score_new_64_0.001_SF')
# plt.plot(returns1[0], color = 'grey')
# variance = returns1[1]/np.sqrt(30)

# plt.fill_between(np.arange(100), returns1[0]+variance, returns1[0]-variance, facecolor='grey', alpha=0.2)
# plt.show()

#bad performance: (.05 and 16) (.05, 32) (16, .01) ()

