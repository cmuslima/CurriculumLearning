from training_loop import teaching_training

import utils
import numpy as np
def run_train_loop(args):
    save_single_run = True
    if save_single_run:
        end = args.num_runs_start+1
    else:
        end = args.num_runs_end
  
    dir = f'{args.rootdir}/teacher-data'

    for seed in range(args.num_runs_start, end):


        print(f'\n\n\n\nrun {seed} with teacher SR = {args.SR} and batch size = {args.teacher_batchsize} and learning rate = {args.teacher_lr} buffer = {args.teacher_buffersize}, reward {args.reward_function}\n\n\n\n')
        print(f'start seed = {args.num_runs_start}, ending seed = {end}')
        if args.trained_teacher:
            file = utils.get_file_name(args, dir, seed)
            return_file = utils.get_return_file_name(args, dir, seed)
            return_list = utils.get_data(return_file)
            print(f'file being used = {file}, return_file = {return_file}')
        else:
            file = None
            return_list = None
        teacher_return_list, teacher_scores, teacher_actions, eval_student_score  = teaching_training(seed, args, file, return_list)
        print(f'Teacher returns on run {seed}, {teacher_return_list}')
    
    
        collected_data = [teacher_return_list, teacher_scores, teacher_actions, eval_student_score]
        data_names = ['teacher_return_list', 'teacher_scores', 'teacher_actions', 'eval_student_score']

        for idx, data in enumerate(collected_data):
            dir = f'{args.rootdir}/teacher-data'
            utils.make_dir(args, dir)
            file_details = [data_names[idx], seed]
            model_name = utils.get_model_name(args, dir, file_details)
            utils.save_data(model_name,data)

        
    print(f'run {seed} complete')
        
