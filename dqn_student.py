import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


from student_dqn_model import QNetwork, Student_State_Encoder
from replay_buffer import GoalReplayBuffer

BUFFER_SIZE = int(1e8)  # replay buffer size
#BATCH_SIZE = 5  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
#LR = 5e-2 # learning rate
UPDATE_EVERY = 1  # UPDATE FREQUENCY: how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, batchsize, lr,seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state, state size is 6 in this case, 
            action_size (int): dimension of each action, action size is 5
            seed (int): random seed
        """
        self.batchsize = batchsize
       
        self.LR = lr
        self.state_size = 7*7*3+2
        self.action_size = 4
      
        #self.seed = np.random.seed(0)
        self.seed = np.random.seed(seed)
        #print('seed being used in the teacher agent', seed)

        # Q-Network
        decode = True
        if decode:
            self.qnetwork_local = Student_State_Encoder(0).to(device)
            self.qnetwork_target = Student_State_Encoder(0).to(device) 
        
        else:
            self.qnetwork_local = QNetwork(self.state_size, self.action_size, 0).to(device)
            self.qnetwork_target = QNetwork(self.state_size, self.action_size, 0).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Replay memory

        self.q_learning_memory = GoalReplayBuffer(self.action_size, BUFFER_SIZE, self.batchsize, 0)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def add_to_memory(self, state, goal, action, reward, next_state, done):
        self.q_learning_memory.add(state, goal, action, reward, next_state, done)
        #print('finished adding to memory')
    def ready_to_learn(self):
        if len(self.q_learning_memory) >= self.batchsize:
            experiences = self.q_learning_memory.sample()
            self.learn(experiences, GAMMA)
          
            
                    
    def step(self, state, goal, action, reward, next_state, done):

        self.add_to_memory(state, goal, action, reward, next_state, done)
       
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            #print('about to learn')
            self.ready_to_learn()
            

    def concat_state_goal(self, state,goal):
        goal = np.reshape(goal, (1,2))
    
        return np.concatenate((state, goal), axis = 1)
    def act(self, state, goal, eps=0.):
        #print('in the act')
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        goal_conditioned_state = self.concat_state_goal(state, goal)
        goal_conditioned_state = torch.from_numpy(goal_conditioned_state).float().to(device)
        
        #print(np.shape(goal_conditioned_state))

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(goal_conditioned_state)
        
        #print('number of actions', self.action_size, self.action_list)
        #print('numbe of action values', np.shape(action_values))
    
        



        #print('action values', action_values)

        #print('max action', np.argmax(action_values))
        self.qnetwork_local.train()

        

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            #print('taking a greedy teacher action')
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            #print('taking a random teacher action')
            action = np.random.choice(np.arange(self.action_size))
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
        states, goals, actions, rewards, next_states, dones = experiences
        goal_conditioned_states = []
        goal_conditioned_next_states = []
        for s, g, s_prime in zip(states, goals, next_states):
            goal_conditioned_states.append(np.concatenate((s, g), axis = 0))
            goal_conditioned_next_states.append(np.concatenate((s_prime, g), axis = 0))

        goal_conditioned_states = np.array(goal_conditioned_states)
        goal_conditioned_states = torch.from_numpy(goal_conditioned_states).float().to(device)
        goal_conditioned_next_states = np.array(goal_conditioned_next_states)
        goal_conditioned_next_states = torch.from_numpy(goal_conditioned_next_states).float().to(device)
        #print(np.shape(goal_conditioned_next_states))
        # get targets
       
        self.qnetwork_target.eval()
  
        with torch.no_grad():
            Q_targets_next = torch.max(self.qnetwork_target.forward(goal_conditioned_next_states), dim=1, keepdim=True)[0]
           
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
       
        # get outputs
        self.qnetwork_local.train()
        Q_expected = self.qnetwork_local.forward(goal_conditioned_states).gather(1, actions)
        
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