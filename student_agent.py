import numpy as np

import global_vars

class agent():    
    def __init__(self, rows, columns, action_size, LR, discount):
        self.discount= discount#1
        
        self.LR= LR #0.5
        self.eps=0.01
        self.rows = rows
        self.columns = columns
        self.q_matrix= dict()
        self.num_actions=action_size
        self.start_state= np.array([0,0])
        np.random.seed(0)
        
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

        state_tuple= tuple(state)
        
      
        # Epsilon-greedy action selection
        if np.random.random() > eps:
            
            movement, index_of_action = self.argmax(self.q_matrix[state_tuple], env)
            return movement, index_of_action
        else:
          
            action = np.random.choice(np.arange(self.num_actions))
            movement, index_of_action = env.action_list[action]
            return movement, index_of_action

    def learning_update(self, reward, next_state, state,index_of_action, done):  

        best_action_value= np.max(self.q_matrix[tuple(next_state)])

        Q_A_prime_value=best_action_value

        if done:
            target = reward
        else:
            target = reward + self.discount*Q_A_prime_value

        Q_A_value= self.q_matrix[tuple(state)][0][index_of_action] 
        self.q_matrix[tuple(state)][0][index_of_action] = Q_A_value + self.LR*(target- Q_A_value)

    def push(self, tuple, trajectory):
        trajectory.append(tuple)


    def act(self, state, env):
        movement, index_of_action =self.e_greedy_action_selection(state, env, self.eps)
        return movement, index_of_action

    def step(self, state, action_index, action_movement, env): #should return next_state, reward, done
        

        next_state= action_movement + state #this applies the action and moves to the next state 
        reward, done = env.check_reward(next_state)

        next_state = env.check_state(next_state, state) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state

        return next_state, reward, done

