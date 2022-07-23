import pickle
def get_data(file):
    with open(file,'rb') as input:
        data = pickle.load(input)
    return data


SR = 'buffer_q_table'
rf = 'simple_LP'
batchsize = 64
lr = .001
buffer = 75
for seed in range(0,10):
    rootdir = f'./{SR}/teacher-data/teacher_scores_{SR}_simple_LP_{lr}_{batchsize}_{buffer}_{seed}'
    data = get_data(rootdir)
    print('len of data', len(data))
    if len(data) == 300:
        print("GOOD")
    else:
        print('BAD')