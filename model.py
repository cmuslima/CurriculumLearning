#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:20:34 2021

@author: kerrick
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, args):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        #print('self.state_size',self.state_size,self.action_size )
        self.hidden_size = args.teacher_network_hidden_size
       
        print('self.hidden_size', self.hidden_size)
        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        #print('pass')
        self.fc2_q_learning = nn.Linear(self.hidden_size, self.hidden_size)
        #print('pass 2')
        self.fc3_q_learning = nn.Linear(self.hidden_size, action_size)
        
        #there is a mistake in fc3_sarsa, it should be hidden size to action size, but luckily I don't use it for now
        self.fc2_sarsa = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_sarsa = nn.Linear(self.hidden_size, action_size)
        self.multi_controller = args.multi_controller
        self.multi_controller_v2 = args.multi_controller_v2
        self.student_type = args.student_type

        print(args.multi_controller, args.student_type)
        print('self.student type in the init Q netowrk', self.student_type)
        
        
    #I just need to change the dqn so I can adapt the student type 
    def multi_controller_forward(self, state):
        """Build a network that maps state -> action values."""
       # print('in the multi controller NN')
        x = self.fc1(state)
        #print('after fc1', x)
        x = F.relu(x)
        #print(f'student type in the NN {self.student_type}')
        if self.student_type == 'q_learning':
            print("Making a q learning update in the nN")
            x = self.fc2_q_learning(x)
            x = F.relu(x)
            action_values = self.fc3_q_learning(x)
        else:
            print('making a sarsa update in the NN')
            x = self.fc2_sarsa(x)
            x = F.relu(x)
            action_values = self.fc3_sarsa(x)  
        
        return action_values
    def multi_controller_forward2(self, state):
        """Build a network that maps state -> action values."""
       # print('in the multi controller NN')
        x = self.fc1(state)
        #print('after fc1', x)
        x = F.relu(x)
        #print(f'student type in the NN {self.student_type}')

        x = self.fc2_q_learning(x)
        x = F.relu(x)
        if self.student_type == 'q_learning':
            print("Making a q learning update in the nN")
            action_values = self.fc3_q_learning(x)
        else:

            action_values = self.fc3_sarsa(x)  
        
        return action_values
    def vanilla_forward(self, state):
        #print('in v for')
        #print('in the vanilla forward function', self.student_type)
        #print('Shuold be in the forward function in args.multi_controller == False')
        """Build a network that maps state -> action values."""
        #print('state inputted into forward', state)
        x = self.fc1(state)
        #print('after fc1', x)
        x = F.relu(x)
        #print('after relu', x)
        x = self.fc2_q_learning(x)
        #print('after fc2', x)
        x = F.relu(x)

        action_values = self.fc3_q_learning(x)
        
        return action_values
    def forward(self, state):
        #print('in the inital forward function')
        """Build a network that maps state -> action values."""
        #print('haha', self.student_type)
        if self.multi_controller:

            if self.multi_controller_v2:
                action_values = self.multi_controller_forward2(state)
            else:
                action_values = self.multi_controller_forward(state)
        else:
            #print('here')
            #print(state)
            action_values = self.vanilla_forward(state)
        
        return action_values