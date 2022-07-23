import pickle
def get_data(file):
    #print('file', file)
    with open(file,'rb') as input:
        data = pickle.load(input)
    return data


num_runs = 29

learning_rates = [.005, .001]
buffer_sizes = [75,100]
batch_sizes = [64,128,256] 
SR = ['buffer_q_table'] #'L2T' #'buffer_policy', 'buffer_q_table', 'params' need to do these with L2T
rf = ['L2T', 'LP', 'target_task_score', '0_target_task_score', 'cost']

for buffer in buffer_list:
    for batch_size in batch_sizes:
        for i in range(num_runs):

            if 'buffer' in state:
                file = f'teacher_return_list_{state}_{rf}_{lr}_{batch_size}_{buffer}_{i}'
            else:
                file = f'teacher_return_list_{state}_{rf}_{lr}_{batch_size}_{i}'

            try:
                data = get_data(file)
                if len(data) < 300:
                    print('file', file)
            except:
                print(f'{file} doesn not exist')
                print(f'state = {state}')
                print(f'rf = {rf}')
                print(f'buffer = {buffer}, batch = {batch_size}, lr = {lr}')
        
                   
