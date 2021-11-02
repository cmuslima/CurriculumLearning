
#try to look at making the weights 0, or having some optimistic inital values.


import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


from model import QNetwork
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
#BATCH_SIZE = 5  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
#LR = 5e-2 # learning rate
UPDATE_EVERY = 1  # UPDATE FREQUENCY: how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed,  action_list, LR, BATCHSIZE):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state, state size is 6 in this case, 
            action_size (int): dimension of each action, action size is 5
            seed (int): random seed
        """
        self.batchsize = BATCHSIZE
       
        self.LR = LR
        self.state_size = state_size
        self.action_size = action_size
        self.action_list = action_list
        self.seed = random.seed(seed)


        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batchsize, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0



    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
    

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.batchsize:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #print('state', state)
    
        #state = state[0]
        #print('state', state, 'of size', np.shape(state))

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        #print('number of actions', self.action_size, self.action_list)
        #print('numbe of action values', np.shape(action_values))
    
        



        #print('action values', action_values)

        #print('max action', np.argmax(action_values))
        self.qnetwork_local.train()

        

        # Epsilon-greedy action selection
        if random.random() > eps:
            #print('taking a greedy teacher action')
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            #print('taking a random teacher action')
            action = random.choice(np.arange(self.action_size))
            return action

    def evaluate_q(self, state, eps=0.):
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
    
        #print('size of action', np.shape(actions))
        #print('actions in the learn function', actions)
        #print('size of states', np.shape(states))
        #print('state', states)
        #print('rewards', rewards)

        # get targets
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_targets_next = torch.max(self.qnetwork_target.forward(next_states), dim=1, keepdim=True)[0]
            #print('Q_targets_next', Q_targets_next)

        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        #print('Q_targets', Q_targets)
        #print('shape of q targets')
       # print(np.shape(Q_targets))
       # print('Qtargets', Q_targets)

        # get outputs
        self.qnetwork_local.train()
        Q_expected = self.qnetwork_local.forward(states).gather(1, actions)
        #print('Q_expected', Q_expected)
        #print('shape of q expected', np.shape(Q_expected),Q_expected)
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