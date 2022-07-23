from student_agent import tabular_agent
#from gym_fetch_RT import RT_run
import numpy as np
import random

   

def get_student_type(episode, args):
    student_list = [0,1]
    student_list = student_list*500
    if args.multi_students:
        type = student_list[episode]
        if type == 0:
            student_type = 'sarsa'
            print('getting student type', student_type)
        else:
            student_type = 'q_learning'
        print('getting student type', student_type)
        # if episode < (args.teacher_episodes+1)/2:
        #     student_type = 'sarsa'
        # else:
        #     student_type = 'q_learning'
        # print('getting student type', student_type)
        #student_type = 'sarsa'
        student_type = 'q_learning'
        print('student_type', student_type)
    else:
        
        student_type = args.student_type
    
    return student_type

def initalize_students(args):

    student_agents = dict()

    for i in range(0, args.num_student_processes):

        mystudentagent = initalize_single_student(args)

        student_agents[i] = mystudentagent


    
    return student_agents

def initalize_single_student(args):


    if args.tabular:
        mystudentagent = tabular_agent(args.rows, args.columns, args.student_num_actions, args.student_lr, args.student_discount, args.student_eps)
        mystudentagent.initalize_q_matrix() 
        print(mystudentagent.q_matrix)
    else:
        mystudentagent = None

    
    return mystudentagent