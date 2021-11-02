import random
import numpy as np

import pickle
import numpy as np

from teacher_evaluation_loop import eval_loop


def run_evaluate_teacher(args):
    subdir1= 'teacher-checkpoints'
    subdir2= 'evaluation-data'
    num_teaching_episodes = 1

    evaluation_scores_all_runs = []
    teacher_actions_all_runs = []
    teacher_score = []
    random.seed(0)
    
    for seed in range(0,args.runs):
        file = f'{args.rootdir}/{args.experiment_folder}/{subdir1}/teacher_agent_checkpoint_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{seed}_{args.alpha}.pth'
        
        
        evaluation_average_score, evaluation_teacher_data, final_teacher_score = eval_loop(num_teaching_episodes, file, seed)
        evaluation_scores_all_runs.append(evaluation_average_score)
        teacher_actions_all_runs.append(evaluation_teacher_data)
        teacher_score.append(final_teacher_score)
    averaged_eval_scores = [np.mean(np.array(evaluation_scores_all_runs), axis = 0), np.std(np.array(evaluation_scores_all_runs), axis = 0)]
    print('average scores', averaged_eval_scores[0])
    averaged_teacher_actions = [np.mean(np.array(teacher_actions_all_runs), axis = 0), np.std(np.array(teacher_actions_all_runs), axis = 0)]
    assert len(teacher_score) == args.runs

    

    model_name = f'{args.rootdir}/{args.experiment_folder}/{subdir2}/evaluation_average_score_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{args.alpha}'
    with open(model_name, 'wb') as output:
        pickle.dump(averaged_eval_scores, output)
    
    model_name = f'{args.rootdir}/{args.experiment_folder}/{subdir2}/evaluation_teacher_data_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{args.alpha}'
    with open(model_name, 'wb') as output:
        pickle.dump(averaged_teacher_actions, output)
    
    
    model_name = f'{args.rootdir}/{args.experiment_folder}/{subdir2}/all_teacher_actions_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{args.alpha}'
    with open(model_name, 'wb') as output:
        pickle.dump(teacher_actions_all_runs, output)
    
    model_name = f'{args.rootdir}/{args.experiment_folder}/{subdir2}/all_teacher_score_{args.reward_function}_{args.batchsize}_{args.lr}_{args.SR}_{args.alpha}'
    with open(model_name, 'wb') as output:
        pickle.dump(teacher_score, output)