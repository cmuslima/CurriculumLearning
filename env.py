import numpy as np
import random

class lavaworld():
    def __init__(self, rows, columns, oneDmaze):       
        self.start_state= np.array([0,0])  #target start state   
        self.termination_state= np.array([rows-1,columns-1])    

        self.blocked_states= [(0,3), (1,3), (2,3), (3,3), (4,3),(5,3), (6,3), (7,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11)    ]
        self.rows=rows
        self.columns=columns
        self.oneDmaze = oneDmaze
        #actions 
        if self.oneDmaze: 
            self.left=np.array([0, -1]) # 2
            self.right=np.array([0, 1])  #3
            self.action_list=[(self.left,0), (self.right, 1)]
        else:
            self.up=np.array([-1,0])  #0
            self.down=np.array([1, 0]) # 1
            self.left=np.array([0, -1]) # 2
            self.right=np.array([0, 1])  #3
            self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]


    def check_reward(self, state):
        if self.oneDmaze:
            freq_reward = False #reward function with -1 per time step

            terminal= False
            
            if np.array_equal(state,self.termination_state) == True:
                if freq_reward:
                    reward = 0
                else:
                    reward = 1
                terminal= True
                
            else:
                if freq_reward:
                    reward = -1
                else:
                    reward = 0

            return reward, terminal
        else:
            terminal= False

            if np.array_equal(state, self.termination_state) == True:
                reward = 0
                terminal= True
            elif tuple(state) in self.blocked_states:
                reward = -100
            else:
                reward = -1

            return reward, terminal

    def check_state(self, next_state, state):
        
        next_state_tuple= tuple(next_state)
        
        if self.oneDmaze:
            if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
                next_state = state   
            
        else:
            if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
                next_state = state

            if next_state_tuple in self.blocked_states: # bc if this happens it wouldnt be near a blocked state 
                next_state = self.start_state #return back to start state
                #print('here')
            
        return next_state 

class four_rooms():
    def __init__(self, rows, columns, oneDmaze):       
        self.start_state= np.array([0,0])  #target start state   
        self.termination_state= np.array([5,4])    

        self.blocked_states= [(3,0), (0,3), (6,3), (3,6), (2,3), (4,3), (3,2), (3,3), (3,4)]
        self.rows=rows
        self.columns=columns
        self.oneDmaze = oneDmaze
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]


    def check_reward(self, state):
        sparse = True
        if sparse == True:
            if np.array_equal(state, self.termination_state) == True:
                reward = 1
                terminal= True
            else:
                reward = 0
                terminal= False
        else:
            if np.array_equal(state, self.termination_state) == True:
                reward = -1
                terminal= True
            else:
                reward = -1
                terminal= False

        return reward, terminal

    def check_state(self, next_state, state):
        
        next_state_tuple= tuple(next_state)
        
        if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
            next_state = state

        if next_state_tuple in self.blocked_states: # bc if this happens it wouldnt be near a blocked state 
            next_state = state #return back to start state
            
        return next_state 


class expanded_four_rooms():
    def __init__(self, rows, columns, oneDmaze):       
        self.start_state= np.array([0,0])  #target start state   
        self.termination_state= np.array([5,4])    

        self.blocked_states= [(3,0), (0,3), (6,3), (3,6), (2,3), (4,3), (3,2), (3,3), (3,4)]
        self.rows=rows
        self.columns=columns
        self.oneDmaze = oneDmaze
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]


    def check_reward(self, state):
        sparse = True
        if sparse == True:
            if np.array_equal(state, self.termination_state) == True:
                reward = 1
                terminal= True
            else:
                reward = 0
                terminal= False
        else:
            if np.array_equal(state, self.termination_state) == True:
                reward = -1
                terminal= True
            else:
                reward = -1
                terminal= False

        return reward, terminal

    def check_state(self, next_state, state):
        
        next_state_tuple= tuple(next_state)
        
        if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
            next_state = state

        if next_state_tuple in self.blocked_states: # bc if this happens it wouldnt be near a blocked state 
            next_state = state #return back to start state
            
        return next_state 

class maze():
    def __init__(self, rows, columns, oneDmaze):       
        self.start_state= np.array([10,4])  #target start state   
        self.termination_state= np.array([0,15])    

        self.blocked_states= [(0,3), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (9,3), (10,3), (3,0), (3,1), (3,2), (7,0), (7,1), (7,2), (3,4), (3,5), (7,4), (7,6), (7,7), (3,7), (4,7), (5,7), (6,7), (8,7), (9,7), (10,7), (0,6), (1,6), (6,8), (6,9), (6,10), (0,10), (1,10), (2,10), (3,10),(8,10), (9,10),(10,10), (5,12), (5,13), (5,14), (5,15), (6,12), (7,12), (8,13), (8,14), (8,15)]

        self.rows=rows
        self.columns=columns
        self.oneDmaze = oneDmaze
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]


    def check_reward(self, state):
        sparse = True
        if sparse == True:
            if np.array_equal(state, self.termination_state) == True:
                reward = 1
                terminal= True
            else:
                reward = 0
                terminal= False
        else:
            if np.array_equal(state, self.termination_state) == True:
                reward = -1
                terminal= True
            else:
                reward = -1
                terminal= False

        return reward, terminal

    def check_state(self, next_state, state):
        
        next_state_tuple= tuple(next_state)
        
        if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
            next_state = state

        if next_state_tuple in self.blocked_states: # bc if this happens it wouldnt be near a blocked state 
            next_state = state #return back to start state
            
        return next_state 