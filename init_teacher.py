import student_env_setup
from evaluation import train_evaluate_protocol
import init_student
import utils
import torch
import numpy as np
import math
from actionvalue_tilecoder import ACTileCoder


# 
import random
class teacher_utils():
    def __init__(self, args):
        self.LP_dict = dict()
        self.student_returns_dict = dict()
     
        self.student_success = False
        self.teacher_score = 0
        #teacher decay information
        self.target_task_success_threshold = None
        self.student_model = None
        self.teacher_total_steps = 0
        self.curriculum = [3]*14 + [5]*8 + [7]*8 + [1]*1000 #[6]*1 + [5]*1 + [4]*1 + [0]*2
        self.config_params = None
        self.dims = None 
        self.past_teacher_action = None
        print('len of curriculum', len(self.curriculum))
        self.AC_TC = ACTileCoder()
        self.num_tiles=8
        self.num_tilings = 176
        #self.action_count=0
        if args.env == 'fetch_push':
            self.run, self.build_env = utils.import_modules(args)
        if args.env == 'fetch_reach_3D_outer':
            self.RT_run  = utils.import_modules(args)
        # if args.env == 'four_rooms' and args.tabular == False:
        #     self.student_train, self.evaluate_task = utils.import_modules(args)

    def visual_student(self, args):
        student_seed = 0
        seed = 0
        student_env = self.reset_teacher_run(args, seed, student_seed)

        self.train_evaluate_protocol = train_evaluate_protocol(student_env, args)
        self.train_evaluate_protocol.visualize(args)
    def initalize_teacher(self, args, seed, student_seed = None, file=None):
        if args.trained_teacher == False and args.evaluation:
            seed = student_seed


        if args.evaluation and args.student_type == 'PPO' and args.trained_teacher:
            seed = student_seed 
        print(f'seed in init teacher', seed)

        if args.trained_teacher == False and args.evaluation: #this is for the random curriculum and learning from stratch
            student_env = self.reset_not_trained_teacher_run(args, seed, student_seed)
        else:
            student_env = self.reset_teacher_run(args, seed, student_seed)

        self.train_evaluate_protocol = train_evaluate_protocol(student_env, args)
    
        
        if args.teacher_agent == 'DQN':
            
            
            if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action':
                #print('am I in the right place')
                from buffer_teacher_agent import DQNAgent
                
                teacher_agent = DQNAgent(state_size=self.teacher_state_size, action_size = self.teacher_action_size, seed=seed, args= args) 
                #print('here')
                if args.trained_teacher:
                    print('using file', file)
                    teacher_agent.qnetwork_local.load_state_dict(torch.load(file))
            else:
                #print('am I ever here')
                from teacher_agent import DQNAgent
                #print('seed here', seed)
                teacher_agent = DQNAgent(state_size=self.teacher_state_size, action_size = self.teacher_action_size, seed=seed, args = args) 

                if args.trained_teacher:
                    print('using trained teacher', file)
                    teacher_agent.qnetwork_local.load_state_dict(torch.load(file))

        #example of what I will include once I have different kinds of teacher agents.  
        
        else:
            teacher_agent = None
        return teacher_agent, student_env

    def step(task_index, task_name,student_agent, current_student_episode, args):
        self.first_occurence_task_check(task_index, task_name,student_agent, args)               
        
        source_task_score, target_task_score = self.student_train_protocol(student_agent, task_name, args)

        LP = self.get_LP(source_task_score, task_index, args)

        self.update_teacher_dicts(task_index, source_task_score, target_task_score, LP, current_student_episode, args)

        reward = self.get_teacher_reward(task_index, LP, target_task_score, args)

        if args.student_type == 'DDPG':
            task_index = self.teacher_action_list.index(task_index)
        traj_prime = self.get_traj_prime(task_index, source_task_score, target_task_score, student_episode, student_agent, args)
        done = self.find_termination(target_task_score, current_student_episode, args.student_episodes, args)
        return traj_prime, reward, done
    def close_run(self, args):
        if args.env == 'fetch_push':
            self.run.student_training(self.student_model, self.config_params, self.dims, True, self.env_dict, self.target_task, args)
        else:
            print('inside close run')
            self.RT_run.student_training(self.student_model, self.target_task, args, self.env_dict, True, self.config_params, self.dims)

    def get_teacher_first_action(self, args):
        if args.easy_initialization: # easy_initialization sets the first task to an easy one
            if args.tabular:
                if args.env == 'maze':
                    task_index = 7
                if args.env == 'big_room':
                    task_index = 1
                elif args.env == 'four_rooms':
                    task_index = 2
                elif args.env == 'cliff_world':
                    task_index = len(self.teacher_action_list) - 1
                elif args.env == 'expanded_fourrooms':
                    task_index = 2
                
                elif args.env == 'combination_lock':
                    task_index = len(self.teacher_action_list) - 1
                    

            else:
                if args.env == 'four_rooms':
                    task_index = 5
                if args.env == 'fetch_push' or args.env == 'fetch_reach_2D' or args.env == 'fetch_reach_2D_outer' or args.env == 'fetch_reach_3D_outer' or args.env == 'fetch_reach_3D':
                    task_index = 2 #len(self.teacher_action_list)
                    self.past_teacher_action = 0
                
        
        if args.random_curriculum:
            print('choosing random task')
            if args.tabular:
                task_index = random.randint(0, self.teacher_action_size-1)
            else:
                
                task_index = random.choice(self.teacher_action_list)
                #task_index = random.choice([2])
                print('task index', task_index)
                

        task_name = self.get_gym_env(task_index, args)

        if args.evaluation and args.target_task_only:
            task_index = self.target_task_index
            task_name = self.target_task
            print('using target task only',task_index )
        
        return task_name,task_index

    def start_non_trained_teacher_episode(self, args, student_agent, student_episode):
        student_id = None
        task_name, task_index = self.get_teacher_first_action(args)
        print('just finished getting my first teacher action')
        print(f'Task = {task_name}, {task_index}')
        assert self.student_model == None and self.student_params == 0
        source_task_training_score, obss, q_values, actions, self.student_model, self.student_params, self.config_params, self.dims = self.train_evaluate_protocol.train(student_agent, self.student_type, task_name, args, self.student_model, self.env_dict, None, None)

        if args.student_type == 'DDPG':
            source_task_score, target_task_score, _ = self.train_evaluate_protocol.source_target_evaluation(student_id, student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)
        else:
            source_task_score, target_task_score, self.student_params = self.train_evaluate_protocol.source_target_evaluation(student_id, student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)
         
        return task_index
    def start_teacher_episode(self, args, student_agent, student_episode):
        
        student_id = None
        task_name, task_index = self.get_teacher_first_action(args)
        print('just finished getting my first teacher action')
        #print(f'Task = {task_name}, {task_index}')
        self.first_occurence_task_check(task_index, task_name,student_agent, args)
        print('just finished first_occurence_task_check in the start teacher episode function ')
        assert self.student_model == None and self.student_params == 0
        #print('inside start teacher episode')
        source_task_training_score, obss, q_values, actions, self.student_model, self.student_params, self.config_params, self.dims = self.train_evaluate_protocol.train(student_agent, self.student_type, task_name, args, self.student_model, self.env_dict, None, None)


        if args.SR == 'buffer_action' or args.SR == 'buffer_policy' or args.SR == 'buffer_q_table':
            self.add_obs_to_buffer(obss, q_values, actions, args)

        #print('just finished adding to buffer')
        if args.student_type == 'DDPG':
            source_task_score, target_task_score, _ = self.train_evaluate_protocol.source_target_evaluation(student_id, student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)
        else:
            source_task_score, target_task_score, self.student_params = self.train_evaluate_protocol.source_target_evaluation(student_id, student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)


        print(f'source_task_score = {source_task_score}')
        LP = self.get_LP(source_task_score, task_index, args)

        self.update_teacher_dicts(task_index, source_task_score, target_task_score, LP, student_episode, args)
        reward  = self.get_teacher_reward(task_index, LP, target_task_score, source_task_training_score, 0, args)

        traj = self.get_traj_prime(task_index, source_task_score, target_task_score, student_episode, student_agent, args)

        return task_index, traj

    # def reset(self, args, student_agent):
             
    #     task_name, task_index = self.get_teacher_first_action(args)
    #     print('just finished getting my first teacher action')
    #     #print(f'Task = {task_name}, {task_index}')
    #     self.first_occurence_task_check(task_index, task_name,student_agent, args)
    #     #print('just finished first_occurence_task_check in the start teacher episode function ')
    #     assert self.student_model == None and self.student_params == 0
    #     print('inside start teacher episode')
    #     _, obss, q_values, actions, self.student_model, self.student_params, self.config_params, self.dims = self.train_evaluate_protocol.train(student_agent, task_name, args, self.student_model, self.env_dict, None, None)
    #     #print('after train', self.config_params, self.dims)
    #     #print('just finished train_evaluate_protocol.train(student_agent, task_name, args, self.student_type)')
    #     #print(f'score on the first task {score}')



    #     #before
    #     # if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action':
    #     #     self.add_obs_to_buffer(obss, q_values, actions, args)
    #     if args.SR == 'buffer_action' or args.SR == 'buffer_policy' or args.SR == 'buffer_q_table':
    #         self.add_obs_to_buffer(obss, q_values, actions, args)

    #     #print('just finished adding to buffer')
    #     if args.student_type == 'DDPG':
    #         source_task_score, target_task_score, _, q_values, actions, obss = self.train_evaluate_protocol.source_target_evaluation(student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)
    #     else:
    #         source_task_score, target_task_score, self.student_params, q_values, actions, obss = self.train_evaluate_protocol.source_target_evaluation(student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)

    #     if args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action':
        
    #         self.add_obs_to_buffer(obss, q_values, actions, args)
    #     #print('finished self.train_evaluate_protocol.source_target_evaluation')
    #     #if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table':
    #         #self.add_obs_to_buffer(obss, q_values, actions, args)
    #     #print(f'source task score {source_task_score}')
    #     #print(f' {self.target_task} target_task_score {target_task_score}')
            
    #     LP = self.get_LP(source_task_score, task_index, args)
    #     print('finished getting LP')
    #     print(f'LP = {LP}')
    #     self.update_teacher_dicts(task_index, source_task_score, LP, args)
    #     print('finished updating teacher dicts')
    #     reward  = self.get_teacher_reward(task_index, LP, target_task_score, args)
    #     print('finished getting reward')
    #     print(f'reward = {reward}')
    #     traj = self.get_traj_prime(task_index, source_task_score, student_agent, args)
    #     print('traj shape', np.shape(traj))
    #     print('finished getting traj')
    #     return task_index, traj            
    def first_occurence_task_check(self, task_index, task_name, student_agent, args):
        if task_index not in list(self.student_returns_dict.keys()):
            print(f'task index = {task_index} not in returns dict')
            #evaluate to get the first return in the LP signal
            average_score, _ = self.train_evaluate_protocol.evaluate(student_agent, task_name, args, self.student_model, self.env_dict, self.config_params, self.dims) #should check on this, so that I don't have to use env
            #I made an update to the line above, I dont think I need to save the student params here
            
            #print('got average score', average_score)
            
            #if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table':
                #self.add_obs_to_buffer(obss, q_values, actions, args)
            if args.normalize:
                average_score = utils.normalize(-(args.max_time_step+1), self.get_max_value(task_index), average_score)
                
            update_entry = {task_index: average_score}
            #print(update_entry)
            self.student_returns_dict.update(update_entry)
            print('returns dict', self.student_returns_dict)
            #print('student returns dict', self.student_returns_dict)
            #print('should be here')
        

    def update_teacher_dicts(self, task_index, source_task_score, target_task_score, LP, current_student_episode, args):

        if args.normalize:
            source_task_score = utils.normalize(-(args.max_time_step+1), self.get_max_value(task_index), source_task_score)
            
        #print(f'Updating teacher dicts')
        update_entry = {task_index: source_task_score}
        #print(f'update entry = {update_entry}')
        

        self.student_returns_dict.update(update_entry)
        #print(self.student_returns_dict)
        self.update_LP_dict(task_index, LP)

        if current_student_episode == 0:
            self.average_target_task_score = 0
        else:
            self.average_target_task_score+= (1/current_student_episode)*(target_task_score-self.average_target_task_score)

        self.average_target_task_LP = target_task_score - self.average_target_task_score
        #print(self.LP_dict)

    def student_train_protocol(self, student_agent, task_name, args):
        student_id = None
        #self.student_model, self.student_params this is only for ddpg student
        source_task_training_score, obss, q_values, actions, self.student_model, self.student_params, _, _ = self.train_evaluate_protocol.train(student_agent, self.student_type, task_name, args, self.student_model, self.env_dict, self.config_params, self.dims)
        
        if args.random_curriculum == False and args.target_task_only == False and args.handcrafted == False:
            if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action':
                self.add_obs_to_buffer(obss, q_values, actions, args)
       
        if args.student_type == 'DDPG':
            source_task_score, target_task_score, _ = self.train_evaluate_protocol.source_target_evaluation(student_id, student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)
       
        else:
            source_task_score, target_task_score, self.student_params= self.train_evaluate_protocol.source_target_evaluation(student_id, student_agent, task_name, self.target_task, args, self.student_model, self.env_dict, self.config_params, self.dims)
        
    
        return source_task_score, target_task_score, source_task_training_score




    def save_teacher_data(self, args, teacher_agent, return_list, teacher_return, seed):
        dir = f'{args.rootdir}/teacher-checkpoints' #will need to check whether its necessary to make 2 directories seperately
        utils.make_dir(args, dir)
        teacher_agent.save_memory(args, seed)

        dir = f'{args.rootdir}/teacher-data'
        utils.make_dir(args, dir)
        file_details = ['teacher_return_list', seed]
        model_name = utils.get_model_name(args, dir, file_details)
        utils.save_data(model_name,return_list)
        if args.saving_method == 'every_episode': #need to add s

            if args.teacher_agent == 'DQN':
                if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action':
                    model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}.pth'
                else:
                    model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}.pth'
                
                print(f'Saving {model_name}')
                torch.save(teacher_agent.qnetwork_local.state_dict(), model_name)
                
        elif args.saving_method == 'exceeds_average':
            
            N = 5
            if len(return_list) > N and teacher_return > sum(return_list[(-1*N):])/N:
                if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action':
                    model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}.pth'
                else:
                    model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}.pth'
                
                print(f'Saving {model_name}')
                torch.save(teacher_agent.qnetwork_local.state_dict(), model_name)
                


    def set_teacher_action_list(self, args):
    #we make the assumption that the first task in the teacher's action list is the target task
        if args.env == 'four_rooms':
            if args.tabular:
                self.teacher_action_list = [np.array([0,0]), np.array([3,1]), np.array([5,3]), np.array([1,3]),np.array([3,5])]
            else:
                self.teacher_action_list = []
                for i in range(0, 10):
                    self.teacher_action_list.append(i)
            
        elif args.env == 'expanded_fourrooms':
            self.teacher_action_list = [np.array([0,0]),np.array([3,1]), np.array([5,3]), np.array([5,5]),np.array([3,4]),np.array([1,4]), np.array([0,6]), np.array([0,9]), np.array([6,6]), np.array([3,7]), np.array([5,9])]

        elif args.env == 'maze':
            self.teacher_action_list = [np.array([10,4]),np.array([1,1]), np.array([5,1]), np.array([9,1]),np.array([7,5]), np.array([3,6]), np.array([5,10]), np.array([2,12]), np.array([10,8]),  np.array([10,14]),  np.array([7,13])]
            print('teacher_action_list', self.teacher_action_list)
        elif args.env == 'big_room':
            self.teacher_action_list = [np.array([9,0]),np.array([5,4]),np.array([6,5]), np.array([5,3]), np.array([7,4]),np.array([5,1]), np.array([7,2]), np.array([8,1]), np.array([3,3]), np.array([1,2]),  np.array([1,5]),  np.array([3,5]), np.array([5,7]), np.array([3,7]), np.array([1,7]), np.array([7,7])]
            print('teacher_action_list', self.teacher_action_list)
        elif args.env == 'cliff_world':
            self.teacher_action_list = [np.array([3,0]),np.array([0,0]), np.array([0,3]), np.array([0,6]),np.array([0,9]),np.array([0,11]),np.array([2,2]),np.array([2,5]),np.array([2,8]), np.array([2,11])] #,np.array([1,2]), np.array([1,5]), np.array([1,9]) ]




        elif args.env == 'combination_lock':
            
            self.teacher_action_list = []
            for i in range(0, args.columns):
                print('here')
                if i*3 >= args.columns:
                    break
                else:
                    print(i, i+3)
                    self.teacher_action_list.append(np.array([0,i*3]))
            print('self.teacher_action_list', self.teacher_action_list)
        elif args.env == 'fetch_reach_2D' or args.env == 'fetch_reach_2D_outer' or  args.env == 'fetch_reach_3D':
            self.teacher_action_list = []
            for i in range(2, 11):
                self.teacher_action_list.append(i)
        elif args.env == 'fetch_reach_3D_outer':
            self.teacher_action_list = []
            for i in range(2, 11):
                self.teacher_action_list.append(i)
        elif args.env == 'fetch_push':
            self.teacher_action_list = []
            for i in range(1,10):
                self.teacher_action_list.append(i)    

        self.teacher_action_size = len(self.teacher_action_list)

        # if args.increase_decrease:
        #     self.teacher_action_size = 3
        self.get_target_task(args)

    def change_task_difficulty(self, previous_action, action):
        if action == 0:
            return self.teacher_action_list[previous_action]
        if action == 1:
            try:
                return self.teacher_action_list[previous_action+1]
            except:
                return self.teacher_action_list[previous_action]
        if action == 2:
            try:
                return self.teacher_action_list[previous_action-1]
            except:
                return self.teacher_action_list[previous_action]
    def get_target_task(self, args):
        self.target_task = self.get_gym_env(0, args)
        self.target_task_index = 0
        if args.env == 'fetch_reach_2D_outer':
            self.target_task_index = 6
            self.target_task= 'FetchReachSparse-v6'
        if args.env == 'fetch_reach_2D':
            self.target_task = self.get_gym_env(6, args)
            self.target_task_index = 6
        if args.env == 'fetch_reach_3D_outer':
            self.target_task_index = 6
            self.target_task= 'FetchReach3DSparse-v6'
        if args.env == 'fetch_reach_3D':
            self.target_task = self.get_gym_env(6, args)
            self.target_task_index = 6
        if args.env == 'fetch_push':
            self.target_task = self.get_gym_env(1, args)
            self.target_task_index = 1
    def initalize_teacher_state_space(self, args):

        if args.SR == 'action_return':
            if args.one_hot_action_vector:
                self.teacher_state_size =  1 + self.teacher_action_size #return + one hot encoding of the action
            else:
                self.teacher_state_size = 2 #action, return

        elif args.SR == 'L2T':
            print('teacher_action_size', self.teacher_action_size)
            self.teacher_state_size = self.student_input_size + self.teacher_action_size
            print()
        elif args.SR == 'loss_mismatch':
            print('teacher_action_size', self.teacher_action_size)
            self.teacher_state_size = self.student_input_size + self.teacher_action_size
            #assert self.teacher_state_size == 13
    

        elif args.SR == 'tile_coded_params':
            if args.tabular:
                self.teacher_state_size = self.student_input_size
            else:
                self.teacher_state_size = self.student_input_size #this will need to get the total params of the student network. 
        elif args.SR == 'params':
            if args.tabular:
                self.teacher_state_size = self.student_input_size*args.student_num_actions
            else:
                self.teacher_state_size = self.student_input_size #this will need to get the total params of the student network. 
        elif args.SR == 'params_student_type':
            if args.tabular:
                self.teacher_state_size = self.student_input_size*args.student_num_actions+2
                print('self.student_input_size', self.student_input_size)
            else:
                self.teacher_state_size = self.student_input_size #this will need to get the total params of the student network. 
        elif args.SR == 'policy_table':
            self.teacher_state_size = self.student_input_size*args.student_num_actions

    
        #for ppo student, this is really the sampled action.
        elif args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table':

            #print('student_input_size', self.student_input_size)
            #print('args.student_num_actions', args.student_num_actions)
            
            self.teacher_state_size = args.teacher_buffersize*(self.student_input_size+args.student_num_actions)
            #print('self.teacher_state_size', self.teacher_state_size)

        elif args.SR == 'buffer_action' or args.SR == 'buffer_max_action':
            self.teacher_state_size = args.teacher_buffersize*(self.student_input_size+args.student_output_size)
    #The purpose of this is to keep incrementing the curriculum length until the stuent solves the target task. Once the student solves the target task, the teacher will just select the target task only. 
    
    def evaluation_true(self, args, teacher_episode):
        if teacher_episode == args.teacher_episodes:
            self.eval_flag = True
    def get_eps(self, args, eps):
        self.teacher_total_steps+=1
        print('self.teacher_total_steps', self.teacher_total_steps)
        eps = eps
        if self.teacher_total_steps > args.teacher_batchsize:
            eps = max(args.teacher_eps_end, args.teacher_eps_decay*eps) 

        if self.eval_flag:
            print('ok')
            #eps = 0

        return eps
    
    def update_teacher_score(self, args):
        if args.evaluation and self.student_success == False:
            self.teacher_score+=1
        return self.teacher_score
    
    def get_teacher_action(self, teacher_agent, traj, eps, args):
        #print(traj)
        #print('shape of traj', np.shape(traj))
        if args.random_curriculum == False and  args.target_task_only == False and args.handcrafted == False:
            task_index = teacher_agent.act(traj, args, eps)
            task_name = self.get_gym_env(task_index, args)

        if args.random_curriculum:
            print('choosing random task')
            if args.tabular:
                task_index = random.randint(0, self.teacher_action_size-1)
            else:
                
                task_index = random.choice(self.teacher_action_list)
                #task_index = random.choice([2])
                print('task index', task_index)
            task_name = self.get_gym_env(task_index, args)


        if (args.evaluation and self.student_success) or args.target_task_only:
            print('using target tsk')
            task_index = self.target_task_index
            task_name = self.target_task
        
        if args.handcrafted:
            print('self.action_count', self.action_count)
            task_index = self.curriculum[self.action_count]
            print('task index', task_index)
            task_name = self.get_gym_env(task_index, args)

            
        self.action_count+=1
        #print('task_index', task_index, 'task_name', task_name)
        return task_index, task_name

    def get_traj_prime(self, task_index, source_task_score, target_task_score, student_episode, student_agent, args):
        
        if args.SR == 'tile_coded_params':
            all_active_tiles = []
            for key in list(student_agent.q_matrix.keys()):
                q_values = student_agent.q_matrix.get(key)
                #print('q_values',q_values)
                active_tiles=self.AC_TC.get_tiles(q_values[0])
                #print('active tiles', active_tiles)
                #print(len(active_tiles))
                all_active_tiles.append(active_tiles)

            traj_prime = np.array(all_active_tiles)
            traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
            #print('traj_prime', traj_prime)
            #print(student_agent.q_matrix)
        elif args.SR == 'action_return':
            if args.one_hot_action_vector:
                one_hot_vector = self.get_one_hot_action_vector(task_index, 1, self.teacher_action_size)
                if args.debug:
                    print(f'one hot vector for action {task_index} = {one_hot_vector}')
                traj_prime = one_hot_vector + [source_task_score]
            else:
                traj_prime = [task_index, source_task_score] 

            traj_prime = np.array(traj_prime)
        
        elif args.SR == 'L2T':
        # We collect several simple features, such as passed mini-batch number (i.e., iteration), the average historical training loss and historical validation accuracy. They are
        # proven to be effective enough to represent the status of current student mode
            print('task index', task_index)
            print('teacher action size', self.teacher_action_size)
            if args.env == 'fetch_push':
                task_index-=1
            if args.env == 'fetch_reach_3D_outer':
                task_index-=2
            one_hot_vector = utils.get_one_hot_action_vector(task_index, 1,  0, self.teacher_action_size)

            traj_prime = one_hot_vector + [source_task_score, target_task_score, student_episode] 
            traj_prime = np.array(traj_prime)
            print(traj_prime)
            traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
        elif args.SR == 'loss_mismatch':
        # We collect several simple features, such as passed mini-batch number (i.e., iteration), the average historical training loss and historical validation accuracy. They are
        # proven to be effective enough to represent the status of current student mode
            print('task index', task_index)
            print('teacher action size', self.teacher_action_size)
            if args.env == 'fetch_push':
                task_index-=1
            if args.env == 'fetch_reach_3D_outer':
                task_index-=2
            one_hot_vector = utils.get_one_hot_action_vector(task_index, 1,  0, self.teacher_action_size)

            traj_prime = one_hot_vector + [source_task_score, target_task_score, self.average_target_task_score, self.average_target_task_LP, student_episode/args.student_episodes] 
            traj_prime = np.array(traj_prime)
            print(traj_prime)
            traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
        elif args.SR == 'params':
            if args.tabular:
                q_values = list(student_agent.q_matrix.values())
                traj_prime = np.array(q_values)
                traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
            else:
                traj_prime = np.array(self.student_params)
            
        elif args.SR == 'params_student_type':
            if args.tabular:
                #print('in get trah')
                q_values = list(student_agent.q_matrix.values()) 
                traj_prime = np.array(q_values)
                #print('teacher_state_size', self.teacher_state_size)
                traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size-2))
                if self.student_type == 'q_learning':
                    index = 0
                else:
                    index = 1
                one_hot_student_type = utils.get_one_hot_action_vector(index, 1, 0, 2)
                #print('one_hot_student_type', one_hot_student_type)
                student_type = np.reshape(one_hot_student_type, (1,2))
                
                traj_prime = np.concatenate((traj_prime, student_type), axis = 1)
                #print('self.student_type', self.student_type)
                #print('traj prime', traj_prime)
            else:
                traj_prime = np.array(self.student_params)
        elif args.SR == 'policy_table':
            policy_table_values = self.get_policy_table(student_agent, args)
            traj_prime = np.array(policy_table_values) 
            traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size)) 
        
    
        elif args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action': #this buffer q table will now just look at the q value of the action chosen, not all q values 
            mini_state = []
            vals = []
            #print('buffer',self.student_state_buffer )
            #print(len(self.student_state_buffer))
            for i in range(0, args.teacher_buffersize):
                s = random.randint(0, len(self.student_state_buffer)-1)
                #print('s',s)
                #print('len of buffer', len(self.student_state_buffer))
                state = self.student_state_buffer[s]
                policy = self.policy_table[state] #policy or q-value
    
                value = np.concatenate((state, policy))
                #print(f'index = {s} state = {state}')
                mini_state.append(value)

            mini_state = np.array(mini_state)

            traj_prime = mini_state
        #print('traj prime', traj_prime)
        return traj_prime


    def get_policy_table(self, student_agent, args):
        for row_num in range(args.rows):
            for col_num in range(args.columns):
                self.policy_table.update({(row_num,col_num): np.zeros([1,args.student_num_actions])})
                self.argmax(student_agent.q_matrix[(row_num,col_num)], row_num, col_num, args)
                
                if args.debug:
                    print(student_agent.q_matrix[(row_num, col_num)])
                    print(f'policy table {row_num},{col_num} = {self.policy_table[(row_num,col_num)]}')
        return list(self.policy_table.values())
        
    def initalize_all_dicts(self):
        #print('initalizing ALP and student return data structs')
        try:
            for action in self.teacher_action_list: 
                emptylist = list()
                self.LP_dict.update({action:emptylist})
        except:
            for action in range(0, len(self.teacher_action_list)): 
                emptylist = list()
                self.LP_dict.update({action:emptylist})   
        self.student_returns_dict = dict()
        self.policy_table= dict()
        self.student_state_buffer = []
        
       




    def update_LP_dict(self, task_index, LP_value):
        self.LP_dict[task_index].append(LP_value)


        

    def determine_target_task_success_threshold(self, args):

        if args.env == 'expanded_fourrooms' or args.env == 'four_rooms':
            if args.tabular:
                self.target_task_success_threshold = (.99)**13 #15
            else:
                self.target_task_success_threshold = .6
        if args.env == 'maze':
            self.target_task_success_threshold = (.99)**35
        if args.env == 'big_room':
            self.target_task_success_threshold = (.99)**10
        if args.env == 'combination_lock':
            self.target_task_success_threshold = (.99)**(args.columns+int(args.columns*.10))
        if args.env == 'cliff_world':
            self.target_task_success_threshold = (-1)*20
        if args.env == 'fetch_reach_2D' or args.env == 'fetch_reach_2D_outer' or args.env == 'fetch_reach_3D' or args.env == 'fetch_reach_3D_outer':
            self.target_task_success_threshold = .9

        if args.env == 'fetch_push':
            self.target_task_success_threshold = .85
        if args.debug:
            print(f'self.target_task_success_threshold is {self.target_task_success_threshold}')
    
    

            
    def check_for_success_on_target_task(self, target_task_score, args):
        if target_task_score >= self.target_task_success_threshold:
            target_task_reward = 0
            if args.debug:
                print('agent success on target task')
            success = True
        else:
            if args.debug:
                print('agent failure at target task')
            target_task_reward = -1
            success = False
        
        return target_task_reward, success

   



    def get_SF(self, task_index, args):
       
        length = len(self.LP_dict[task_index]) #what
        LP_value_list = list(self.LP_dict[task_index])
        if args.debug:
            print(f'ALP_value_list for action {task_index} = {LP_value_list}')
        SF = 0 #stagnation factor
        
        for LP in LP_value_list:
            if LP < args.LP_threshold and LP > -args.LP_threshold:
                SF+=1
            else:
                SF = 0
       
        self.state_SF[task_index] = SF
        if args.debug:
            print(f'SF for action {task_index} = {SF}') 
            print(f'all SF = {self.state_SF}')
        return SF

    def get_teacher_reward(self, task_index, LP, target_task_score, source_task_training_score, student_sucesss_episode, args):
        target_task_reward, success = self.check_for_success_on_target_task(target_task_score, args)

        #SF = self.get_SF(task_index, args)

        #print('USING REWARD FUNCTION', args.reward_function)
        
        if args.reward_function == 'binary_LP':
            if LP > 0:
                reward = 1
            else:
                reward = -1
        
            print('LP', LP, 'reward', reward)
            
        if args.reward_function == 'target_task_score':
            reward = target_task_score
        if args.reward_function == '0_target_task_score':
            if success:
                reward = target_task_score
            else:
                reward = 0

        
        if args.reward_function == 'L2T':
            if success:
                reward = -math.log(student_sucesss_episode/args.student_episodes)
            else:
                reward = 0
            #print('reward', reward)
        if args.reward_function == 'Narvekar2018' or args.reward_function == 'Narvekar2017': 
            reward = -(source_task_training_score)

        if args.reward_function == "cost":
            if args.debug:
                print('using regular reward function')
            reward = target_task_reward # this will be 0 if the student solved the target task and -1 otherwise 

        elif args.reward_function == "LP":
            reward = args.alpha*LP 
            
        elif args.reward_function == 'simple_LP' or args.reward_function == 'simple_ALP':
            reward = -1 + args.alpha*LP 
        
        elif args.reward_function == 'LP_SF_log':
              
            if SF >= args.stagnation_threshold: #this says I have converged and I have still failed on the target task, so I should probably avoid this task
                reward = (-1)*math.log(10+SF) #target_task_reward acts as the -1 cost signal 
            else:
                reward = -1 + args.alpha*LP 
        elif args.reward_function == 'LP_log':

            reward = (-1)*math.log(10+SF) + args.alpha*LP 
        elif args.reward_function == 'LP_cost_relu':
            reward = target_task_reward - self.relu(SF) + args.alpha*LP 

        return reward

       
    def find_termination(self, target_task_score, current_episode, max_episode, args):
        #print(target_task_score, self.target_task_success_threshold)
        return_value = False
        print('max_episode', max_episode, 'current episode', current_episode)
        if current_episode == max_episode:
            return_value = True
        if target_task_score > self.target_task_success_threshold:
            print('success in find terminationn')
            self.student_success = True

        if args.reward_function == 'simple_LP' or args.reward_function == 'cost' or args.reward_function == 'L2T' or args.reward_function == 'LP':
            if target_task_score > self.target_task_success_threshold:
                print('success in find terminationn')
                return_value = True

        return return_value

    def get_max_value(self, task_index):
        max_values = [-13, -14, -11, -8, -5, -3, -10, -7, -4, -1]
        return max_values[task_index]

    def get_LP(self, new_G, task_index, args):
        past_G = self.student_returns_dict.get(task_index) #LP dict stores the most recent student return on a particular env
        # if args.percent_change:
        #     LP = (new_G-past_G)/abs((past_G)) #I'm noticing in cliff world,we shouldn't use ABS(LP)
        if args.normalize:
            print(f'task index = {task_index}')
            print('new G before normalizing', new_G)
            new_G = utils.normalize(-(args.max_time_step+1), self.get_max_value(task_index), new_G)
            print(f'new G after normalizing= {new_G} past G = {past_G}')
        LP = new_G-past_G #made a change here
        if args.normalize:
            print(f'LP = {LP}')
        assert LP <= 1 
        if args.reward_function == 'simple_ALP':
            LP = abs(new_G-past_G)
            assert LP <= 1 
        
        reward = LP
        if args.debug:
            print('returns dict', self.student_returns_dict)
            print('past return', past_G, 'return now', new_G, 'LP', LP)
        
        return reward 



    def get_policy_buffer_state_rep(self, q_values, action, args):
        #we could also think about embedding the sampled action taken in that state (might not be the argmax action)
        
        
        if args.student_type == 'PPO': #this is a strict one hot vector always
            one_hot_vector = utils.get_one_hot_action_vector(action, 1, 0, args.student_num_actions)
        else:
            if all(element == q_values[0] for element in q_values):
                one_hot_vector = [.25]*args.student_num_actions

            else:
                action = np.argmax(q_values)
                one_hot_vector = utils.get_one_hot_action_vector(action, 1, 0, args.student_num_actions)
        return one_hot_vector
    
    def get_embeb_q_value_state_rep(self, q_values, action, args):
        if args.student_type == 'q_learning' or args.student_type == 'sarsa':#This would embed the all q values of the state
            return q_values
        else: #this is really only used for ppo student
             #This would embed the q value of the argmax action
            return utils.get_one_hot_action_vector(action, q_values, 0, args.student_num_actions)
        
        #we could also think about embedding the q value of the sampled action taken in that state (might not be the argmax action)
    
    


    def add_obs_to_buffer(self,obss, q_values, actions, args):
    

        #print('states', obss)
        #print('actions', actions)
        #print(np.shape(obss), np.shape(actions))
        #print(type(obss), type(actions))
        shape = np.shape(obss)
        
        #print('shape', shape)
        #assert len(obss) == len(q_values) == len(actions)
        for i in range(0, shape[0]): #this goes through all the observations encountered during a training episode(s)
            state = obss[i]
            #print('state', state)
            #print('action', actions[i])
            #print('q va', q_values[i])
            #print('state', state, 'action', actions[i])
            #print(actions[i])
            if args.SR == 'buffer_policy' or args.SR == 'buffer_max_policy':
                one_hot_vector = self.get_policy_buffer_state_rep(q_values[i], actions[i], args)
                #print('vector', one_hot_vector)
            if args.SR == 'buffer_q_table' or args.SR == 'buffer_max_q_table':
                
                one_hot_vector = self.get_embeb_q_value_state_rep(q_values[i], actions[i],args)
                #print('vector', one_hot_vector)
            if args.SR == 'buffer_action' or args.SR == 'buffer_max_action':
                one_hot_vector = actions[i]
            #print(i)
            #print('state', one_hot_vector)
            state = utils.convert_array_to_tuple(state, args)
 
            #print(np.shape(state))
            self.policy_table[state] = one_hot_vector
            self.student_state_buffer.append(state)
        #print('shape of state', np.shape(state))
        #print('shape of action', np.shape(actions[0]))
        #print(f'student_state_buffer before making changes')
        #print(self.student_state_buffer)
        #print(type(self.student_state_buffer))
        self.student_state_buffer = list(set(self.student_state_buffer))
        
        # print(f'student_state_buffer')
        # print(self.student_state_buffer)


    def reset_env_dicts(self, args, seed, student_seed):

        self.env_dict = None
        if args.student_type == 'DDPG' and args.env != 'fetch_push':
            self.env_dict = self.RT_run.init_student(args) 
        if args.student_type == 'DDPG' and args.env == 'fetch_push':
            self.env_dict = self.build_env.main(args) 


    def reset_not_trained_teacher_run(self, args, seed, student_seed):
        self.eval_flag = False
        student_env, self.student_input_size = student_env_setup.make_env(args)
        self.set_teacher_action_list(args)
        
        utils.set_global_seeds(seed, args, student_seed)
        self.teacher_total_steps = 0 #need to fix this
        
        self.reset_env_dicts(args, seed, student_seed)
        return student_env

    def reset_teacher_run(self, args, seed,student_seed):
        self.eval_flag = False
        student_env, self.student_input_size = student_env_setup.make_env(args)
        self.set_teacher_action_list(args)
        self.initalize_teacher_state_space(args)
        
        utils.set_global_seeds(seed, args, student_seed)
        self.teacher_total_steps = 0 #need to fix this
        
        self.reset_env_dicts(args, seed, student_seed)
        return student_env

    def reset_not_trained_teacher_params(self, args, student_type):
        self.determine_target_task_success_threshold(args)
        if args.env == 'four_rooms' and args.student_type == 'PPO':
            utils.remove_dir(args, f'{args.rootdir}/storage/{args.model_folder_path}') #need to create a folder path
            print('removing directory')
        self.student_params = 0
        self.student_type = student_type #0 if student_type == 'q_learning' else 1
        self.student_model = None
        self.action_count=0 
    def reset_teacher_params(self, args, student_type):
        #print(f'student type', student_type)
        self.determine_target_task_success_threshold(args)
        if args.env == 'four_rooms' and args.student_type == 'PPO':
            utils.remove_dir(args, f'{args.rootdir}/storage/{args.model_folder_path}') #need to create a folder path
            print('removing directory')

        self.state_SF = [0]*self.teacher_action_size    
    
        self.initalize_all_dicts()
        self.student_state_buffer = []
        self.student_params = 0
        self.student_type = student_type #0 if student_type == 'q_learning' else 1
        self.student_model = None
        self.action_count=0 
        self.average_target_task_score = 0
        self.average_target_task_LP = 0
        if args.random_student_seed == False:
            
            self.env_dict = utils.set_student_seeds()
        print('at the end of reset')

    def get_gym_env(self, task_index, args):

        if args.env == 'maze' or args.env == 'cliff_world' or args.env == 'combination_lock' or args.env == 'expanded_fourrooms':
            task_name = self.teacher_action_list[task_index]
        elif args.env == 'four_rooms' and args.tabular == False:
            task_name = f'MiniGrid-Simple-4rooms-{task_index}-v0'
        if args.env == 'fetch_reach_2D':
            task_name = f'FetchReachSparse-v{task_index}' 
        if args.env == 'fetch_reach_2D_outer' :
            task_name = f'FetchReachOuterSparse-v{task_index}' 
        if args.env == 'fetch_reach_3D':
            task_name = f'FetchReach3DSparse-v{task_index}' 
        if args.env == 'fetch_reach_3D_outer' :
            task_name = f'FetchReachOuter3DSparse-v{task_index}' 

        if args.env == 'fetch_push':
            task_name = f'FetchPush-v{task_index}' 
        return task_name




