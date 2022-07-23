
import numpy as np
from init_teacher import teacher_utils
import init_student




def teaching_training(seed, args, file, return_list):

    #env_config = env_variables(args)
  
    eps = args.teacher_eps_start     
    #print('eps',eps)
    if return_list == None:
        return_list = [] # this records the teacher returns
        max_teacher_episode =  args.teacher_episodes+1
    else:
        return_list = return_list
        max_teacher_episode = (args.teacher_episodes+1) - len(return_list)
        print('number of teacher episodes left',max_teacher_episode)
    teacher_scores = [] #this records the first student episode for which the student succeeds on the target task
    all_teacher_actions = []

    teacher_help_fns = teacher_utils(args)
    teacher_agent, student_env = teacher_help_fns.initalize_teacher(args, seed, None, file)
    #train_evalute_protocol(student_env)
    #student_env will be None for non-tabular student agents

    #print(f'configration: environment = {args.env}, reward function type: {rf}, number of tasks = {env_config.teacher_action_size}, alpha = {args.alpha}, batch size = {b}, step size = {lr} buffer = {buffer}')
    
    print('run = ', seed)
    #one reset here which initalizes the env, teacher state space, etc
    for i_episode in range(1, max_teacher_episode):
      
        print('teacher episode', i_episode)
        student_agent = init_student.initalize_single_student(args)
     
        #This is only needed if I want to train on different students during one teacher training procedue
        teacher_agent.student_type = args.student_type
        teacher_agent.update_student_type()

       
        teacher_help_fns.reset_teacher_params(args, args.student_type) 
      
        teacher_score = 0 #counter to keep track of the first student episode for which the student succeeds on the target task
        teacher_return = 0
        student_scores = list()
        teacher_action_list = list()

                                                                       
        task_index, traj = teacher_help_fns.start_teacher_episode(args, student_agent, 0)  # loop N times
        #print('COMPLETELY FINISHED WITH MY START TEACHER EPISODE FUNCTION')
        teacher_action_list.append(task_index)
    
        #print(f'Task = {task_index}  traj = {traj}')
        for current_student_episode in range(1, args.student_episodes+1):
          
            print('student episode', current_student_episode)
                
            teacher_score+=1
    
            task_index, task_name = teacher_help_fns.get_teacher_action(teacher_agent, traj, eps, args) # # loop N times
           
            teacher_action_list.append(task_index)
           
            teacher_help_fns.first_occurence_task_check(task_index, task_name,student_agent, args)  #  # loop N times            


            source_task_score, target_task_score, source_task_training_score = teacher_help_fns.student_train_protocol(student_agent, task_name, args) # loop 1 times, with N processes
            #print('just finished student_train_protocol')

            if args.debug:
                print(f'source task score {source_task_score}')
                print(f'target_task_score {target_task_score}')

            LP = teacher_help_fns.get_LP(source_task_score, task_index, args) #loop N times 

            teacher_help_fns.update_teacher_dicts(task_index, source_task_score, target_task_score, LP, current_student_episode, args) #loop N times 
            #print('just finished updating teacher dicts')

            reward = teacher_help_fns.get_teacher_reward(task_index, LP, target_task_score,source_task_training_score, current_student_episode, args) ##loop N times 
            #print('just finished getting reward')
            print(f'TEACHER ACTION = {task_index}, TEACHER REWARD = {reward}')
            if args.debug:
                print(f'updated ALP dict = {teacher_help_fns.LP_dict}')
                print(f'teacher action: {task_index}. Raw ALP = {LP}, Updated ALP = {args.alpha*LP} reward = {reward}')

        

            #This wierd update is only needed when the task_id doesn't go 0,1,2 ...
            if args.student_type == 'DDPG':
                task_index = teacher_help_fns.teacher_action_list.index(task_index)
            traj_prime = teacher_help_fns.get_traj_prime(task_index, source_task_score, target_task_score, current_student_episode, student_agent, args)
            
            if args.clear_buffer:
                teacher_help_fns.policy_table = dict()
                teacher_help_fns.student_state_buffer = []         
            if args.debug:
                print(f'traj prime = {traj_prime}')
        
            
            done = teacher_help_fns.find_termination(target_task_score, current_student_episode, args.student_episodes, args)
            print('done', done)
        
           
            teacher_agent.step(traj, task_index, reward, traj_prime, done, args) ##loop N times
            #print('just finished taking a techer step')
            teacher_help_fns.evaluation_true(args, i_episode)
            eps = teacher_help_fns.get_eps(args, eps)
            print('using eps', eps)
            print(teacher_help_fns.eval_flag)

            traj = traj_prime
            student_scores.append(target_task_score) #I want to collect data on the target task.. to see if over time, it approves on the target task.
            teacher_return+=reward
            print(f'teacher reward at teacher time step {current_student_episode} = {reward}')
            if done:
                if args.debug:
                    print(f'teacher return = {teacher_return}')
                    print(f'teacher score = {teacher_score}')
                #print(f'teacher curriculum = {teacher_action_list}')
                print(f'student scores on target task = {student_scores}')
                print(f'teacher return = {teacher_return}')
                print(teacher_action_list)
                teacher_scores.append(teacher_score)
                all_teacher_actions.append(teacher_action_list)
                if args.student_type == 'DDPG':
                    teacher_help_fns.close_run(args)
                break
            #print('finished a student episode \n')
                
                
        return_list.append(teacher_return)
        print('q learning should only contain numbers 0, 5,6,7')
        if teacher_help_fns.eval_flag:
            print('eps', eps)
            print('teacher action list eval', teacher_action_list)
        teacher_help_fns.save_teacher_data(args, teacher_agent, return_list, teacher_return, seed)



    print(f'Teacher return list = {return_list}')
    return return_list, teacher_scores, all_teacher_actions, student_scores


 
            
