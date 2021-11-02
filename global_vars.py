import numpy as np
import config
from env import lavaworld, four_rooms, expanded_four_rooms, maze
from scipy import stats
import math

#This class creates the environment (four rooms, maze, etc) and has most of the 
#functions for getting the teaacher interaction (i.e getting reward, getting next trajectory)
class env_variables():
    def __init__(self):
        self.shortest_path_approach = False # not used
        self.ALP_dict = dict()
        self.returns_dict = dict()
        self.env = config.env
        self.rows = 0
        self.columns = 0
        self.SR = config.SR
        self.ALP_list = list() #not used
        self.cost_list = list() #not used

        #for student agent
        self.discount =.99
        self.oneDmaze = False
        self.student_num_actions = 4

        #teacher decay information
        self.eps_start=.5
        self.eps_end=0.01
        self.eps_decay=0.99

        self.live_env = self.make_env()
        print('self.live env', self.live_env)
        self.alpha = None
        self.one_hot_vector = False

        self.state_SF = [0]*self.teacher_action_size
        self.success_converge = 0
        self.stagnation_threshold = 3
        self.target_task_success_threshold = None
        self.policy_table= dict()
        self.reward_function = config.reward_function
        

    def configure_env_paramters(self):

        if self.env == 'four_rooms':
            self.rows = 7
            self.columns =7
            self.teacher_action_list = [np.array([0,0]), np.array([3,1]), np.array([5,3]), np.array([1,3]),np.array([3,5])]
        
        elif self.env == 'expanded_fourrooms':
            self.rows = 7
            self.columns =10
            self.teacher_action_list = [np.array([0,0]),np.array([3,1]), np.array([5,3]), np.array([1,3]),np.array([3,5]), np.array([0,9]), np.array([6,6]), np.array([3,7]), np.array([5,9])]
        
        elif self.env == 'maze':
        
            self.rows = 11
            self.columns = 16
            self.teacher_action_list = [np.array([10,4]),np.array([1,1]), np.array([5,1]), np.array([9,1]),np.array([7,5]), np.array([3,6]), np.array([5,10]), np.array([2,12]), np.array([10,8]),  np.array([10,14]),  np.array([7,13])]
        
        if config.debug:
            print(f'env = {self.env}')
        self.teacher_action_size = len(self.teacher_action_list)
        self.initalize_state_space()
        


    
    def initalize_state_space(self):
        if self.SR == 'SF_with_time_step':
            if self.one_hot_action_vector:
                self.state_size =  4 + self.teacher_action_size*2 # action, time_step, reward, updated_ALP, target_task_success + stagnaton factors
            else:
                self.state_size =  5 + self.teacher_action_size #[teacher_action, time_step, reward, updated_ALP, target_task_success]
        
        if self.SR == 'SF_no_time_step':
            if self.one_hot_action_vector:
                self.state_size =  3 + self.teacher_action_size*2 #action, reward, updated_ALP, target_task_success + stagnaton factors
            else:
                self.state_size =  4 + self.teacher_action_size #[teacher_action, reward, updated_ALP, target_task_success]
        
        if self.SR == 'action_return_reward_SF':
            if self.one_hot_action_vector:
                self.state_size =  2 + self.teacher_action_size*2 # action, return, reward + stagnaton factors
            else:
                self.state_size =  3 + self.teacher_action_size #[teacher_action, time_step, reward, updated_ALP, target_task_success]
        
        if self.SR == 'action_return_reward_time_step_SF':
            if self.one_hot_action_vector:
                self.state_size =  3 + self.teacher_action_size*2 #action, return, reward, time_step + stagnaton factors
            else:
                self.state_size =  4 + self.teacher_action_size 

        elif self.SR == 'action_return':
            if self.one_hot_action_vector:
                self.state_size =  1 + self.teacher_action_size #return + one hot encoding of the action
            else:
                self.state_size = 2 #action, return

        elif self.SR == 'action_return_SF':
            if self.one_hot_action_vector:
                self.state_size =  1 + self.teacher_action_size*2 #return + one hot encoding of the action
            else:
                self.state_size = 2 + self.teacher_action_size #action, return
        
        elif self.SR == 'action_return_time_step':
            if self.one_hot_action_vector:
                self.state_size =  2 + self.teacher_action_size
            else:
                self.state_size = 3 #action, return, time_step

        elif self.SR == 'action_return_time_step_SF':
            if self.one_hot_action_vector:
                self.state_size =  2 + self.teacher_action_size*2
            else:
                self.state_size = 3 + self.teacher_action_size #action, return, time_step

        elif self.SR == 'q_matrix':
            self.state_size = self.columns*self.rows*self.student_num_actions

        elif self.SR == 'q_matrix_SF':
            self.state_size = self.columns*self.rows*self.student_num_actions + self.teacher_action_size

        elif self.SR == 'q_matrix_action_return':
            self.state_size = self.columns*self.rows*self.student_num_actions + 2

        elif self.SR == 'q_matrix_action_return_SF':
            self.state_size = self.columns*self.rows*self.student_num_actions + self.teacher_action_size + 2
        
        elif self.SR == 'policy_table':
            self.state_size = self.columns*self.rows*self.student_num_actions

        elif self.SR == 'policy_table_SF':
            self.state_size = self.columns*self.rows*self.student_num_actions + self.teacher_action_size

        elif self.SR == 'policy_table_action_return':
            self.state_size = self.columns*self.rows*self.student_num_actions + 2

        elif self.SR == 'policy_table_action_return_SF':
            self.state_size = self.columns*self.rows*self.student_num_actions + self.teacher_action_size + 2 
    
        print(f'Using State Representation: {self.SR}')
    def make_env(self):
        self.configure_env_paramters()
        if self.env == 'four_rooms':
            env = four_rooms(self.rows, self.columns, oneDmaze=False)
        elif self.env == 'expanded_fourrooms':
            env = expanded_four_rooms(self.rows, self.columns, oneDmaze=False)
        elif self.env == 'maze':
            env = maze(self.rows, self.columns, oneDmaze=False)
        return env

    def get_teacher_action(self, teacher_agent, traj, eps):
        teacher_action_int = teacher_agent.act(traj, eps)
        #returns the index
        teacher_action_array = self.teacher_action_list[teacher_action_int]
        
        return teacher_action_array

    def get_traj_prime(self, teacher_action_int, source_task_score, student_agent, reward, target_task_success, time_step, updated_ALP):

        if self.SR == 'SF_with_time_step':
            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                statelist =  one_hot_vector + [time_step, reward, updated_ALP, target_task_success] +  self.state_SF #all_actions_average_ALP 
            else:
                statelist =  [teacher_action_int, time_step, reward, updated_ALP, target_task_success] +  self.state_SF #all_actions_average_ALP 
            traj_prime = np.array(statelist)
            traj_prime = np.reshape(traj_prime, (1,self.state_size))
        
        elif self.SR == 'SF_no_time_step':
            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                statelist =  one_hot_vector + [reward, updated_ALP, target_task_success] +  self.state_SF #all_actions_average_ALP 
            else:
                statelist =  [teacher_action_int,reward, updated_ALP, target_task_success] +  self.state_SF #all_actions_average_ALP 
            traj_prime = np.array(statelist)
            traj_prime = np.reshape(traj_prime, (1,self.state_size))
        
        elif self.SR == 'action_return':
            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                if config.debug:
                    print(f'one hot vector for action {teacher_action_int} = {one_hot_vector}')
                traj_prime = one_hot_vector + [source_task_score]
            else:
                traj_prime = [teacher_action_int, source_task_score] 

            traj_prime = np.array(traj_prime)
        
        elif self.SR == 'action_return_SF':

            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                if config.debug:
                    print(f'one hot vector for action {teacher_action_int} = {one_hot_vector}')
                traj_prime = one_hot_vector + [source_task_score] + self.state_SF
            else:
                traj_prime = [teacher_action_int, source_task_score] + self.state_SF

            traj_prime = np.array(traj_prime)


        elif self.SR == 'action_return_time_step':
            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                traj_prime = one_hot_vector + [source_task_score, time_step]
            else:
                traj_prime = [teacher_action_int, source_task_score, time_step] 
            traj_prime = np.array(traj_prime)
        elif self.SR == 'action_return_reward_SF':
            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                traj_prime = one_hot_vector + [source_task_score, reward] + self.state_SF
            else:
                traj_prime = [teacher_action_int, source_task_score, reward] + self.state_SF
            
            traj_prime = np.array(traj_prime)
        elif self.SR == 'action_return_time_step_SF':
            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                traj_prime = one_hot_vector + [source_task_score, time_step] + self.state_SF
            else:
                traj_prime = [teacher_action_int, source_task_score, time_step] + self.state_SF
            
            traj_prime = np.array(traj_prime)
        elif self.SR == 'action_return_reward_time_step_SF':
            if self.one_hot_action_vector:
                one_hot_vector = self.one_hot_action_vector(teacher_action_int)
                traj_prime = one_hot_vector + [source_task_score, reward, time_step] + self.state_SF
            else:
                traj_prime = [teacher_action_int, source_task_score, reward, time_step] + self.state_SF
            
            traj_prime = np.array(traj_prime)
        elif self.SR == 'q_matrix':
            q_values = list(student_agent.q_matrix.values())
            traj_prime = np.array(q_values)
            traj_prime = np.reshape(traj_prime, (1,self.state_size))
        elif self.SR == 'q_matrix_SF':
            q_values = list(student_agent.q_matrix.values())
            traj_prime = np.array(q_values)
            traj_prime = np.append(traj_prime,self.state_SF)
            traj_prime = np.reshape(traj_prime, (1,self.state_size))

        elif self.SR == 'q_matrix_action_return':
            q_values = list(student_agent.q_matrix.values()) 
            traj_prime = np.array(q_values) 
            action_return = [teacher_action_int, source_task_score] 
            traj_prime = np.append(traj_prime,action_return)
            traj_prime = np.reshape(traj_prime, (1,self.state_size)) 

        elif self.SR == 'q_matrix_action_return_SF':
            q_values = list(student_agent.q_matrix.values()) 
            traj_prime = np.array(q_values) 
            action_return_SF = self.state_SF + [teacher_action_int, source_task_score] 
            traj_prime = np.append(traj_prime,action_return_SF)
            traj_prime = np.reshape(traj_prime, (1,self.state_size)) 

        elif self.SR == 'policy_table':
            policy_table_values = self.get_policy_table(student_agent)
            traj_prime = np.array(policy_table_values) 
            traj_prime = np.reshape(traj_prime, (1,self.state_size)) 
        
        elif self.SR == 'policy_table_SF':
            policy_table_values = self.get_policy_table(student_agent)
            traj_prime = np.array(policy_table_values) 
            traj_prime = np.append(traj_prime,self.state_SF)

            traj_prime = np.reshape(traj_prime, (1,self.state_size)) 

        elif self.SR == 'policy_table_action_return':
            policy_table_values = self.get_policy_table(student_agent)
            traj_prime = np.array(policy_table_values) 

            action_return = [teacher_action_int, source_task_score]
            traj_prime = np.append(traj_prime,action_return)
            traj_prime = np.reshape(traj_prime, (1,self.state_size))

        elif self.SR == 'policy_table_action_return_SF':
            policy_table_values = self.get_policy_table(student_agent)
            traj_prime = np.array(policy_table_values) 

            action_return_SF = self.state_SF + [teacher_action_int, source_task_score]
            traj_prime = np.append(traj_prime,action_return_SF)
            traj_prime = np.reshape(traj_prime, (1,self.state_size))

        if config.debug:
            print(f'using state rep = {self.SR}')

        return traj_prime

    def get_policy_table(self, student_agent):
        for row_num in range(self.rows):
            for col_num in range(self.columns):
                self.policy_table.update({(row_num,col_num): np.zeros([1,self.student_num_actions])})
                max_action_index = np.argmax(student_agent.q_matrix[(row_num,col_num)])
                self.policy_table[(row_num,col_num)][0][max_action_index] = 1
                if config.debug:
                    print(student_agent.q_matrix[(row_num, col_num)])
                    print('max action_index', max_action_index)
                    print(f'policy table {row_num},{col_num} = {self.policy_table[(row_num,col_num)]}')
        return list(self.policy_table.values())
    
    def convert_teacher_action(self, teacher_action_array): 
       
        for idx, action, in enumerate(self.teacher_action_list):
            if np.array_equal(teacher_action_array, action):
                return idx

        
    def initalize_ALP_dict(self):
        print('initalizing ALP and student return data structs')
        for action in self.teacher_action_list:
            emptylist = list()
            self.ALP_dict.update({str(action):emptylist})
       
        self.returns_dict = dict()

    def update_ALP_dict(self, teacher_action_array, ALP_value):
    
        teacher_action_array = str(teacher_action_array)
        self.ALP_dict[teacher_action_array].append(ALP_value)

        



    def determine_target_task_success_threshold(self):

        if config.optimal_target_threshold: 

            if self.env == 'expanded_fourrooms' or self.env == 'four_rooms':
                self.target_task_success_threshold = (.99)**15
            else:
                self.target_task_success_threshold = (.99)**35
            if config.debug:
                print(f'self.target_task_success_threshold is {self.target_task_success_threshold}')
        else:
            self.target_task_success_threshold = 0 
    

            
    def check_for_success_on_target_task(self, target_task_score):
        self.determine_target_task_success_threshold()
        if target_task_score > self.target_task_success_threshold:
            target_task_reward = 0
            if config.debug:
                print('agent success on target task')
            success = True
        else:
            if config.debug:
                print('agent failure at target task')
            target_task_reward = -1
            success = False
        
        return target_task_reward, success

    def normalize(value, min, max):
        normalize_value = (value - min)/(max- min)
        return normalize_value
    def calculate_zscore(self, value_list):
        zscore = stats.zscore(value_list)[-1]
    
        return zscore
    
    def calculate_normalized_zscore(self, value_list):
        raw_zscore = self.calculate_zscore(value_list)
        
        max_value = max(value_list)
        min_value = min(value_list)
        normalized_value = (value - min_value)/(max_value- min_value)
        return normalized_value


    def get_SF(self, teacher_action_array):
        teacher_action_int = self.convert_teacher_action(teacher_action_array)
        teacher_action_array = str(teacher_action_array)
        length = len(self.ALP_dict[teacher_action_array])
        ALP_value_list = list(self.ALP_dict[teacher_action_array])
        if config.debug:
            print(f'ALP_value_list for action {teacher_action_array} = {ALP_value_list}')
        SF = 0 #stagnation factor
        
        for ALP in ALP_value_list:
            if ALP < .01 and ALP > -.01:
                SF+=1
            else:
                SF = 0
       
        self.state_SF[teacher_action_int] = SF
        if config.debug:
            print(f'SF for action {teacher_action_int} = {SF}') 
            print(f'all SF = {self.state_SF}')
        return SF

    def get_teacher_reward(self, teacher_action, ALP, target_task_score, source_task_cost):
        target_task_reward, target_task_success = self.check_for_success_on_target_task(target_task_score)

        SF = self.get_SF(teacher_action)

        if self.reward_function == 'LP_cost':  

            if SF >= self.stagnation_threshold and target_task_success == False: #this says I have converged and I have still failed on the target task, so I should probably avoid this task
                if config.reward_log:
                    reward = (-1)*math.log(10+SF) + target_task_reward #target_task_reward acts as the -1 cost signal 
                    if config.debug:
                        print(f'log signal: {(-1)*math.log(10+SF)}, target task signal: {target_task_reward}')
                else:
                    reward = -1 + target_task_reward #target_task_reward acts as the -1 cost signal
               
     
            else: #this says I haven't converged and I haven't have a success on the target task
                reward = target_task_reward + self.alpha*ALP 
        


        elif self.reward_function == "cost":
            if config.debug:
                print('using regular reward function')
            #if the student agent has learned the target task, we give the teacher a reward of 0
            if target_task_success:
                reward = 0
            #if the student agent hasn't learned the target task, we give the teacher a reward of -1
            else:
                reward = -1
            

        elif self.reward_function == "LP":
            if SF >= self.stagnation_threshold and target_task_success == False: #this says I have converged and I have still failed on the target task, so I should probably avoid this task
                if config.reward_log:
                    reward = (-1)*math.log(10+SF) 
                    if config.debug:
                        print(f'log signal: {(-1)*math.log(10+SF)}, target task signal: {target_task_reward}')
                else:
                    reward = -1 
              
            
            else: #this says I haven't converged and I haven't have a success on the target task
                reward =  self.alpha*ALP 
        
        
        return reward, SF, target_task_reward

       
    def find_termination(self, target_task_score, episode, max_episode):
        if target_task_score > self.target_task_success_threshold or episode == max_episode:
            if episode == max_episode:
                print('student timed out and didnt succeed on target task')
            else:
                print('student success on target task')
            return True, 1
        else:
            return False, 0


    def get_ALP(self, new_G, teacher_action):
        past_G = self.returns_dict.get(str(teacher_action)) #ALP dict stores the most recent student return on a particular env
        
        ALP = abs(new_G-past_G)
        reward = ALP
        if config.debug:
            print('returns dict', self.returns_dict)
            print('past return', past_G, 'return now', new_G, 'ALP', ALP)
        
        return reward 

    def one_hot_action_vector(self, teacher_action_int):
        one_hot_vector = [0]*self.teacher_action_size
        one_hot_vector[teacher_action_int] =1 
        return one_hot_vector

