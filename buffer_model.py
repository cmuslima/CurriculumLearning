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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Teacher_State_Encoder(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, batch_size, buffer_size, args):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Teacher_State_Encoder, self).__init__()
        self.student_state_encoder = Student_State_Encoder(seed).to(device)
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        hidden_size = 64
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        

    def get_state_rep(self, augmented_state): #handle 64 (N x A things)
        """Build a network that maps state -> action values."""

        if len(list(augmented_state.size())) <3:
            augmented_state = torch.reshape(augmented_state,(self.batch_size,self.buffer_size,6))
        x = self.student_state_encoder.forward(augmented_state) #This ouputs N x h
        #print(f'This is th output from the student state encoder function {x} size = {np.shape(x)}')
        x = torch.max(x, 1)[0] #This should be size h 
        #print(f'Now I took the max = {x} shape = {np.shape(x)}')
        return x
    def forward(self, augmented_state):
        state = self.get_state_rep(augmented_state)
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_values = self.fc3(x)        
        return action_values

#buffer_policy2 contains 2 layers total
#next try one with only 1 softmax layer
class Student_State_Encoder(nn.Module): #This ouputs N X h
    """Actor (Policy) Model."""

    def __init__(self, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Student_State_Encoder, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hs_size = 2
        self.he_size = 4
        self.hs_hidden_size = 32
        self.he_hidden_size = 32
        self.h_hidden_size = self.hs_hidden_size+self.he_hidden_size 
        
        self.fc1 = nn.Linear(self.hs_size, self.hs_hidden_size)
        self.fc2 = nn.Linear(self.he_size, self.he_hidden_size)
        self.fc3 = nn.Linear((self.hs_hidden_size+self.he_hidden_size), self.h_hidden_size)
        
        

    def forward(self, state): #This can handle N by A input
        #print(f'In the student state encoder function, inital state = {state}, size = {np.shape(state)}')
        #shape = list(state.size())
        #print(f'shape = {shape}')
        h_s = state[:,:,0:self.hs_size] #This gets the student states from the buffer input
        #print(f'H_s = {h_s} shape = {np.shape(h_s)}')
        h_e = state[:,:,self.hs_size:] #This gets e_i from the buffer input (Q(s) or one_hot_vector(Q(s)))
        #print(f'h_e = {h_e} shape = {np.shape(h_e)}')
        #h = torch.cat((h_s[0], h_e[0]), dim=1)
        #print(f'h = {h} shape = {np.shape(h)}')

        h_s = self.fc1(h_s)
        h_s = F.relu(h_s)
        h_e = self.fc2(h_e)
        h_e = F.relu(h_e)
        #print(f'h_s.unsqueez(0) = {h_s.squeeze(0)} size = {np.shape(h_s.squeeze(0))}')
        #print(f'h_s shape regular = {np.shape(h_s)}')
        h = torch.cat((h_s, h_e), dim=2)
        #h = torch.stack((h_s, h_e), dim=1)
        #print(f'h = {h} shape = {np.shape(h)}')
        
        h = self.fc3(h)
        h = F.relu(h)
        #h = torch.reshape(h, (shape[0],shape[1],shape[2]))
        #print('returning h')
        return h
