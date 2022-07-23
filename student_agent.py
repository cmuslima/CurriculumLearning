import numpy as np

import utils
import random
class tabular_agent():    
    def __init__(self, rows, columns, action_size, LR, discount, eps):
        self.discount= discount#1
        print('epi', eps)
        self.LR= LR #0.5
        self.eps= eps
        self.rows = rows
        self.columns = columns
        self.q_matrix= dict()
        self.num_actions=action_size
        self.start_state= np.array([0,0])
        self.state_buffer = []
        self.action_buffer = []
        self.bob_alice_q_matrix = dict()
        #np.random.seed(0)
        
    def initalize_q_matrix(self):
        for row_num in range(self.rows):
            for col_num in range(self.columns):
                self.q_matrix.update({(row_num,col_num): np.zeros([1,self.num_actions])})

    def reset(self,start_state):
        self.start_state = start_state
        return self.start_state

    def argmax(self, q_values, env):
        """argmax with random tie-breaking
            Args:
            q_values (Numpy array): the array of action values
            Returns:
            action (int): an action with the highest value
            """
        top = float("-inf")
        ties = []
        q_values = q_values[0]
        for i in range(len(q_values)):
            
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            
            if q_values[i] == top:
                ties.append(i)
        
   
        movement, index_of_action = env.action_list[np.random.choice(ties)]
     
        return movement, index_of_action

    def e_greedy_action_selection(self,state, env, eps):
        #print('eps', eps)
        state_tuple= tuple(state)
        
      
        # Epsilon-greedy action selection
        if np.random.random() > eps:
            #print('being greedy')
            #print('q vector', self.q_matrix[state_tuple], state_tuple)
            movement, index_of_action = self.argmax(self.q_matrix[state_tuple], env)
            return movement, index_of_action
        else:
          
            action = np.random.choice(np.arange(self.num_actions))
            movement, index_of_action = env.action_list[action]
            return movement, index_of_action

    
    # def sarsa_update(self, state,index_of_action, reward,next_state, index_of_next_action,done):
    #     next_action_value= self.q_matrix[tuple(next_state)][0][index_of_next_action] 

    #     if done:
    #         target = reward
    #     else:
    #         target = reward + self.discount*next_action_value

    #     Q_A_value= self.q_matrix[tuple(state)][0][index_of_action] 
    #     self.q_matrix[tuple(state)][0][index_of_action] = Q_A_value + self.LR*(target- Q_A_value)
    def sarsa_update(self, state,index_of_action, reward,next_state, index_of_next_action,done):



        if done:
            target = reward
            #print("Only using the reward as the only")
        else:
            next_action_value= self.q_matrix[tuple(next_state)][0][index_of_next_action] 
            target = reward + self.discount*next_action_value
            #print('Bootstrapping')

        
        Q_A_value= self.q_matrix[tuple(state)][0][index_of_action] 
        #print('Q value before',Q_A_value )
        self.q_matrix[tuple(state)][0][index_of_action] = Q_A_value + self.LR*(target- Q_A_value)
    def q_learning_update(self, state,index_of_action, reward,next_state, done):  


        best_action_value= np.max(self.q_matrix[tuple(next_state)])

        Q_A_prime_value=best_action_value
        # mean_value = []
        # for i in range(0, self.num_actions):
        #     mean_value.append(self.q_matrix[tuple(next_state)][0][i])
        
        # Q_A_prime_value = np.mean(mean_value)
        
        #Q_A_prime_value = self.q_matrix[tuple(next_state)][0][index_next_action] 
        if done:
            target = reward
            #print('using reward only', reward)
        else:
            target = reward + self.discount*Q_A_prime_value
            #print('target =', target)
        #print('state', state)
        #print('q vector', self.q_matrix[tuple(state)][0])
        Q_A_value= self.q_matrix[tuple(state)][0][index_of_action] 
        #print('updating student model')
        #print('q value before',self.q_matrix[tuple(state)][0][index_of_action])

        self.q_matrix[tuple(state)][0][index_of_action] = Q_A_value + self.LR*(target- Q_A_value)
        #print('q value after', self.q_matrix[tuple(state)][0][index_of_action])
       

    def push(self, tuple, trajectory):
        trajectory.append(tuple)


    def act(self, state, env):
        #print('in the student act function', self.eps)
        movement, index_of_action =self.e_greedy_action_selection(state, env, self.eps)
        return movement, index_of_action

    def step(self, state, action_index, action_movement, env): #should return next_state, reward, done
        
        #print('in the step function')
        #print(env.termination_state)
        next_state= action_movement + state #this applies the action and moves to the next state 
        reward, done = env.check_reward(next_state)
        # print('reward', reward)
        # print('done', done)
        # print('next_state',next_state)

        next_state = env.check_state(next_state, state, action_index) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state

        self.add_to_buffer(tuple(state), action_index)
       
        return next_state, reward, done

    def clear_buffer(self):
        self.state_buffer = []
        self.action_buffer = []
    def add_to_buffer(self, state, action):
        self.state_buffer.append(state)
        self.action_buffer.append(action)


class self_play_agent():    
    def __init__(self, rows, columns, action_size, LR, discount, eps):
        self.discount= discount#1
        
        self.LR= LR #0.5
        self.eps= eps
        self.rows = rows
        self.columns = columns
        self.num_actions=action_size
        self.num_actions_Bob = action_size-1
        self.start_state= np.array([0,0])
        self.q_matrix = dict()
        #np.random.seed(0)
        


    def initalize_bob_alice_q_matrix(self, num_actions):
        
        for row_num in range(self.rows):
            for col_num in range(self.columns):
                state = (row_num, col_num)
                for i in range(self.rows):
                    for j in range(self.columns):
                        possible_start_state = (i, j)
                        self.q_matrix.update({(state, possible_start_state): np.zeros([1,num_actions])})

    def reset(self,start_state):
        self.start_state = start_state
        return self.start_state

    def argmax(self, q_values):
        """argmax with random tie-breaking
            Args:
            q_values (Numpy array): the array of action values
            Returns:
            action (int): an action with the highest value
            """
        top = float("-inf")
        ties = []
        q_values = q_values[0]
        for i in range(len(q_values)):
            
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            
            if q_values[i] == top:
                ties.append(i)
        
        
        max_index = np.random.choice(ties)
        index_of_action = max_index
     
        return  index_of_action


    # def e_greedy_action_selection(self,state, eps, player):
    #     #print('state in e_greedy_action_selection', state)
    #     #print('state and action value', self.q_matrix[state])
        
      
    #     # Epsilon-greedy action selection
    #     #print('state', state)
    #     if np.random.random() > eps:
        
    #         index_of_action = self.argmax(self.q_matrix[state])
    #         #print('being greedy', index_of_action)
    #         #print('self.q_matrix[state]', self.q_matrix[state], state)
    #         if state[0] == (0,0) and player == 'Alice':
                
    #             action = 1 #np.random.choice(np.arange(self.num_actions-1))
    #             index_of_action = action
    #             #print('im at this random place', action)
    #         if state[0] == (0,1) and player == 'Alice':
    #             if index_of_action == 0:
    #                 action = np.random.choice(np.arange(1,3))
    #                 index_of_action = action
    #         return index_of_action
    #     else:
    #         print('being random')
    #         if player == 'Alice':
    #             num_actions = 3
    #         else:
    #             num_actions = 2
    #         action = np.random.choice(np.arange(num_actions))
    #         index_of_action = action

    #         if state[0] == (0,1) and player == 'Alice':
                
    #             action = np.random.choice(np.arange(1,3))
    #             index_of_action = action
    #         if state[0] == (0,0) and player == 'Alice':
    #             action = 1 #np.random.choice(np.arange(self.num_actions-1))
    #             index_of_action = action

    #         return index_of_action

    
    def sarsa_update(self, state,index_of_action, reward,next_state, index_of_next_action,done):



        if done:
            target = reward
            #print("Only using the reward as the only")
        else:
            next_action_value= self.q_matrix[tuple(next_state)][0][index_of_next_action] 
            target = reward + self.discount*next_action_value
            #print('Bootstrapping')

        
        Q_A_value= self.q_matrix[tuple(state)][0][index_of_action] 
        #print('Q value before',Q_A_value )
        self.q_matrix[tuple(state)][0][index_of_action] = Q_A_value + self.LR*(target- Q_A_value)

