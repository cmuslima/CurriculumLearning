import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
#import json
from model import QNetwork
from replay_buffer import ReplayBuffer
import pickle


BUFFER_SIZE = int(1e8)  # replay buffer size
#BATCH_SIZE = 5  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
#LR = 5e-2 # learning rate
UPDATE_EVERY = 1  # UPDATE FREQUENCY: how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(1)
class DQNAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed, args):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state, state size is 6 in this case, 
            action_size (int): dimension of each action, action size is 5
            seed (int): random seed
        """
        self.batchsize = args.teacher_batchsize
       
        self.LR = args.teacher_lr
        self.state_size = state_size
        self.action_size = action_size
        self.update_num = 0
        self.two_buffer = args.two_buffer
        self.multi_students = args.multi_students
        self.multi_controller = args.multi_controller
        self.seed = random.seed(seed)
        self.student_type = args.student_type
        print('seed being used in the params teacher agent', seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, args).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, args).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Replay memory

        self.q_learning_memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batchsize, seed)
        self.sarsa_memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batchsize, seed)
        print('type of self.q_learning_memory.memory')
        print(type(self.q_learning_memory.memory))
        a = list(self.q_learning_memory.memory)
        print(a)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def upload_memory(self):
        dir = f'{args.rootdir}/teacher-checkpoints' 
        if 'buffer' in args.SR:
            model_name = f'{dir}/teacher_agent_replaybuffer_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        else:
            model_name = f'{dir}/teacher_agent_replaybuffer_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}'

        data = utils.get_data(model_name)
        for transition in data:
            state, action, reward, next_state, done = transition
            self.q_learning_memory.add(state, action, reward, next_state, done)

    def update_student_type(self):
        self.qnetwork_local.student_type = self.student_type
        self.qnetwork_target.student_type = self.student_type
    def add_to_memory(self, state, action, reward, next_state, done):
        if self.two_buffer:
            if self.student_type == 'q_learning':
                #print('adding q learning experience into the buffer')
                self.q_learning_memory.add(state, action, reward, next_state, done)
            else:
                #print('adding sarsa experience into the buffer')
                self.sarsa_memory.add(state, action, reward, next_state, done)

        else: #add everything to this buffer
            
            self.q_learning_memory.add(state, action, reward, next_state, done)
            print('finished adding to memory')
            


    def save_memory(self, args, seed):
        dir = f'{args.rootdir}/teacher-checkpoints' 
        if 'buffer' in args.SR:
            model_name = f'{dir}/teacher_agent_replaybuffer_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        else:
            model_name = f'{dir}/teacher_agent_replaybuffer_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}'
        
        # self.local_mem = []
        # for i in self.q_learning_memory.memory:
        #     self.local_mem.append(i)

        # print(f'self.local_mem {self.local_mem}')
        # print(self.q_learning_memory.m)
        # with open(model_name, 'wb') as output:
        #     pickle.dump(self.q_learning_memory.m, output)
        
        
        #utils.save_data(model_name, self.q_learning_memory.m)
        # with open(model_name+ '.json','w+') as outfile:
        #     json.dump(self.local_mem, outfile)
        print('NOT saving buffer')
        
    def get_teacher_action(self,args, action):

        if args.student_type == 'DDPG' and args.env != 'fetch_push':
            print(f'env = {args.env}')
            action+=2

        if args.student_type == 'DDPG' and args.env == 'fetch_push':
            print(f'env = {args.env}')
            action+=1
        return action
    def ready_to_learn(self):
        if self.two_buffer:
            if self.student_type == 'q_learning':
                if len(self.q_learning_memory) >= self.batchsize:
                    #print('updating q learning student')
                    experiences = self.q_learning_memory.sample()
                    self.learn(experiences, GAMMA)
                    self.update_num+=1
            else:
                if len(self.sarsa_memory) >= self.batchsize:
                    #print('updating sarsa student')
                    experiences = self.sarsa_memory.sample()
                    self.learn(experiences, GAMMA)
                    self.update_num+=1
        else:
            #print('should be here if self.two_buffer is false', self.two_buffer)
            #print(f'Student Type = {self.student_type}')
            if len(self.q_learning_memory) >= self.batchsize:
                experiences = self.q_learning_memory.sample()
                self.learn(experiences, GAMMA)
                self.update_num+=1
            
                    
    def step(self, state, action, reward, next_state, done,args):
        #print('in step')

        # Save experience in replay memory
        #print('in the step functin')
        self.add_to_memory(state, action, reward, next_state, done)
        #print('just finished adding to memory ')

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        #print(f't step = {eslf.t_step}')
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            #print('about to learn')
            self.ready_to_learn()
            #print('completed the learn update')
    


    def act(self, state, args, eps=0.):
        #print('in the act')
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #print('state', state)
        #print(np.shape(state))
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            #print('actoin values', action_values)
        
        #print('number of actions', self.action_size, self.action_list)
        #print('numbe of action values', np.shape(action_values))
    
        



        #print('action values', action_values)

        #print('max action', np.argmax(action_values))
        self.qnetwork_local.train()

        

        # Epsilon-greedy action selection
        if random.random() > eps:
            #print('taking a greedy teacher action')
            action = np.argmax(action_values.cpu().data.numpy())
            action = self.get_teacher_action(args, action)
            #print('action values', action_values)
            return action
        else:
            #print('taking a random teacher action')
            action = random.choice(np.arange(self.action_size))
            action = self.get_teacher_action(args, action)
            return action

    def evaluate_q(self, state, eps=0.):
        #print('in evaluate q')
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        return action_values.numpy()[0]
    


    def learn(self, experiences, gamma): 
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #print('start of the learn function')
        states, actions, rewards, next_states, dones = experiences



        # get targets
       
        self.qnetwork_target.eval()
  
        with torch.no_grad():
          
            if self.multi_students and self.multi_controller:
                #print('after mutli student and multi controller check')
                Q_targets_next = torch.max(self.qnetwork_target.multi_controller_forward(next_states), dim=1, keepdim=True)[0]
                #print('just received the q targets next')
            else:
                #print(f'At the start of the learn function self.multi_students = {self.multi_students} and self.multi_controller = {self.multi_controller}')
                #print('Shuold be in the forward function in args.multi_controller == False')
                Q_targets_next = torch.max(self.qnetwork_target.forward(next_states), dim=1, keepdim=True)[0]
            #print('Q_targets_next', Q_targets_next)
        #print('finished with torch no grad')
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
       
        # get outputs
        self.qnetwork_local.train()
        if self.multi_students and self.multi_controller:
            Q_expected = self.qnetwork_local.multi_controller_forward(states).gather(1, actions)
        else:
            #print(f'At the end of the learn function self.multi_students = {self.multi_students} and self.multi_controller = {self.multi_controller}')

            #print('Shuold be in the forward function in args.multi_controller == False')
            Q_expected = self.qnetwork_local.forward(states).gather(1, actions)
        
        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        #print('loss', loss)
        # clear gradients
        self.optimizer.zero_grad()

        # update weights local network
        loss.backward()

        # take one SGD step
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        #print('end of the learn function')

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)