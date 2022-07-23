
import numpy as np
from init_teacher import teacher_utils
import init_student


#hi

def teacher_evaluation(args, file, student_seed):

    eps = 0     
    teacher_help_fns = teacher_utils(args)
    teacher_evaluation_seed= student_seed+1
    print(f'args.evaluation_seed = {teacher_evaluation_seed}')
    teacher_agent, student_env = teacher_help_fns.initalize_teacher(args, teacher_evaluation_seed, student_seed, file) #args.teacher_evaluation_seed will be a value outside of 0-5
    total_return = 0
    for i_episode in range(1, 2):
        print('\n\n\n i episode in teacer eval\n\n\n')
        print('teacher episode', i_episode)
        student_agent = init_student.initalize_single_student(args)
        
        #print('student type before', student_type)

        if teacher_agent == 'DQN':
            teacher_agent.student_type = args.student_type
            teacher_agent.update_student_type()
            print('teacher_agent.student_type', teacher_agent.student_type)
        if args.trained_teacher:
            teacher_help_fns.reset_teacher_params(args, args.student_type) #not sure if I should change this or not.. I'll try changing it and keeping it the same. 
        else:
            teacher_help_fns.reset_not_trained_teacher_params(args, args.student_type)
            print('used the not trained teacher params function')

        student_scores = list()
        teacher_action_list = list()

        if args.trained_teacher:                                                            
            task_index, traj = teacher_help_fns.start_teacher_episode(args, student_agent, 0) 
        else:
            task_index = teacher_help_fns.start_non_trained_teacher_episode(args, student_agent, 0)
            traj = None
            print('used the non_trained start teacher episode function')
        
        teacher_action_list.append(task_index)
    
        #print(f'Task = {task_index}  traj = {traj}')
        for current_student_episode in range(1, args.student_episodes):
            print('max student episodes', args.student_episodes)
            print('current student episode',current_student_episode)
                
    

            task_index, task_name = teacher_help_fns.get_teacher_action(teacher_agent, traj, eps, args)
            
            print('action = ', task_index)
            #print('Teacher action', task_index)
            teacher_action_list.append(task_index)
            #print(teacher_action_list)
            if args.trained_teacher:
                teacher_help_fns.first_occurence_task_check(task_index, task_name,student_agent, args)               
        


            #to do: will need to generalize the train function

            source_task_score, target_task_score, source_task_training_score = teacher_help_fns.student_train_protocol(student_agent, task_name, args)

            if args.trained_teacher:
                LP = teacher_help_fns.get_LP(source_task_score, task_index, args)
                teacher_help_fns.update_teacher_dicts(task_index, source_task_score, target_task_score, LP, current_student_episode, args)
                reward = teacher_help_fns.get_teacher_reward(task_index, LP, target_task_score,source_task_training_score, current_student_episode, args)
                total_return+=reward
                traj_prime = teacher_help_fns.get_traj_prime(task_index, source_task_score, target_task_score, current_student_episode, student_agent, args)

        
            
            student_success = teacher_help_fns.find_termination(target_task_score, current_student_episode, args.student_episodes, args)
            teacher_help_fns.update_teacher_score(args)
            student_scores.append(target_task_score)

            print(f'student score on targe task on episode {current_student_episode} = {target_task_score}')
        
            if args.trained_teacher:
                traj = traj_prime

        
                            
                    
    if args.student_type == 'DDPG':
        teacher_help_fns.close_run(args)
    print(f'Student Scores and Teacher Action List from Teaching with {args.teacher_batchsize} and {args.teacher_lr}')    
    print(f' student scores ={student_scores}')
    print(f'teacher_action_list={teacher_action_list}')
    print(f'total return =', total_return)
    return teacher_help_fns.teacher_score, teacher_action_list, student_scores
            
