class state_rep():
    def __init__(self, state_rep):
        self.action_return = False
        self.bit_map = False
        self.q_matrix = False
        self.state_rep = state_rep

    def determine_state_rep(self):
        if self.state_rep == 'action_return':
            self.action_return = True
        if self.state_rep == 'q_matrix':
            self.q_matrix = True 
        if self.state_rep == 'bit_map':
            self.bit_map = True 
