from evalution_loop import teacher_evaluation
import utils
import numpy as np
def run_evaluation_loop(args):
    
    teacher_scores = []
    teacher_actions = []
    student_scores = []
    save_all_runs = False
    save_single_run = True 
    
    print('START', args.num_runs_start)
    print('END',args.num_runs_end)
    curr_data = []
    
    #try:
    if save_single_run:
        end = args.num_runs_start+1
    else:
        end = args.num_runs
    for seed in range(args.num_runs_start, end):

        #print('NN_large_128_128')
        print(f'run {seed} with student lr = {args.student_lr} with batch size = {args.teacher_batchsize} and learning rate = {args.teacher_lr} buffer = {args.teacher_buffersize}, reward {args.reward_function}')


        dir = f'{args.rootdir}/teacher-data'
        file = utils.get_file_name(args, dir, seed)        
        print(f'file being used = {file}')
      
        teacher_score, teacher_action_list, student_score = teacher_evaluation(args, file, seed)
        print('completed teacher eval')
        teacher_scores.append(teacher_score)
        teacher_actions.append(teacher_action_list)
        student_scores.append(student_score)
        
        print(f'teacher_action_list = {teacher_action_list}')

        print(f'student_score', student_score)
        dir = f'{args.rootdir}/evaluation-data'
        if 'buffer' in args.SR:
            model_name = f'{dir}/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        else:
            model_name = f'{dir}/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}'

        if args.random_curriculum:
            utils.make_dir(args, f'./RT/{args.env}/random_curriculum')
            dir = f'./RT/{args.env}/random_curriculum'
            print('should be here', dir)
            model_name = f'{dir}/random_student_score_{seed}'

        if args.target_task_only:
            utils.make_dir(args, f'./RT/{args.env}/target_task_only')
            dir =  f'./RT/{args.env}/target_task_only'
            model_name = f'{dir}/target_task_only_student_score_{seed}'

        if args.student_transfer:
            utils.make_dir(args, f'{args.rootdir}/evaluation-data/transfer_q_learning_to_sarsa')
            model_name = f'{args.rootdir}/evaluation-data/transfer_q_learning_to_sarsa/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_sarsa_{seed}'
        if args.student_lr_transfer:
            utils.make_dir(args, f'{args.rootdir}/evaluation-data/lr_transfer_{args.student_lr}')
            model_name = f'{args.rootdir}/evaluation-data/lr_transfer_{args.student_lr}/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{args.student_lr}_{seed}'
        if args.student_NN_transfer:
            utils.make_dir(args, f'{args.rootdir}/evaluation-data/NN_large_128_128')
            model_name = f'{args.rootdir}/evaluation-data/NN_large_128_128/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        utils.save_data(model_name,student_score)
        

    try:
        raw_averaged_returns = [np.mean(np.array(student_scores), axis = 0), np.std(np.array(student_scores), axis = 0)]
        print('raw_averaged_return', raw_averaged_returns)
    except:
        raw_averaged_returns = 'all returns were 0'
    print('raw_averaged_return', raw_averaged_returns)


    collected_data = [teacher_scores, teacher_actions]
    data_names = ['teacher_scores', 'teacher_actions']
    for idx, data in enumerate(collected_data):

        dir = f'{args.rootdir}/evaluation-data'
        if save_single_run:
            index = str(args.num_runs_start)
        else:
            index = ''
        file_details = [data_names[idx], index]
        model_name = utils.get_model_name(args, dir, file_details)
        if args.random_curriculum:
            dir = f'./RT/{args.env}/random_curriculum'
            print('should be here', dir)
            utils.make_dir(args, dir)
            model_name = f'{dir}/random_{data_names[idx]}_{index}'

        if args.target_task_only:
            dir =  f'./RT/{args.env}/target_task_only'
            utils.make_dir(args, dir)
            model_name = f'{dir}/target_only_{data_names[idx]}_{index}'
        # if args.HER:
        #     dir = dir + '/HER'
        #     utils.make_dir(args, dir)
        #     print('should be here b/c I am using HER')
        # if args.multi_students:
        #     dir = f'{dir}/{args.student_type}'
        #     utils.make_dir(args, dir)
        #     model_name = utils.get_model_name(args, dir, file_details)
        if args.student_transfer:
            utils.make_dir(args, f'{args.rootdir}/evaluation-data/transfer_q_learning_to_sarsa')
            model_name = f'{args.rootdir}/evaluation-data/transfer_q_learning_to_sarsa/{data_names[idx]}_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_sarsa_{index}'
        if args.student_lr_transfer:
            utils.make_dir(args, f'{args.rootdir}/evaluation-data/lr_transfer_{args.student_lr}')
            model_name = f'{args.rootdir}/evaluation-data/lr_transfer_{args.student_lr}/{data_names[idx]}_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{args.student_lr}_{index}'
        if args.student_NN_transfer:
            utils.make_dir(args, f'{args.rootdir}/evaluation-data/NN_large_128_128')
            model_name = f'{args.rootdir}/evaluation-data/NN_large_128_128/{data_names[idx]}_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{index}'
        
        utils.save_data(model_name,data)

