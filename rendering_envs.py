import random
import numpy as np
import gym

COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'white' : np.array([200, 200, 200]),
    'black' : np.array([0, 0, 0]),
    'orange': np.array([255,165,0])
}

class grid(gym.Env):
    def __init__(self):       
        #self.door = np.array([2,2])
        self.start_state= np.array([0,0])  #target start state   
        self.termination_state= np.array([5,4])  

        self.blocked_states= [(3,0), (0,3), (6,3), (3,6), (2,3), (4,3), (3,2), (3,3), (3,4)]

    
        self.ss_list = [(0,0),(3,1), (5,3),(1,3),(3,5), (0,9), (6,6),(3,7), (5,9)]

        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]
        self.lockeddoor = 1 #True
        self.key = (0,0)
        self.have_key = 0 #False
        self.agent_state = (self.start_state, self.have_key)
        self.rows=7
        self.columns=10
        #actions 
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]
        self.tile_size = 64



        #this is for the fixed agent
        

    
    def fixed_act(self,time_step):
        return self.action_sequence[self.episode_number][time_step]
    
    def check_reward(self, state):
        state = state[0]
        terminal= False
        r = -1

        if np.array_equal(state,self.termination_state) == True:
            terminal= True
            r = 0
        return terminal, r

    def check_for_key(self, state):
        state = tuple(state)
        if state == self.key:
            self.have_key = 1 #True
            self.unlockdoor()
            print('we have the key')
             
        



    def fill_square(self, row, col, color, img):
        img[row*self.tile_size:(row+1)*self.tile_size,col*self.tile_size:(col+1)*self.tile_size] = color
        return img
        
    def l2_dist(self, point1, point2):
        xdist = point1[1]-point2[1]
        ydist = point1[0]-point2[0]
        return np.sqrt(xdist**2 + ydist**2)

    def fill_circle(self, row, col, color, img):
        center = (row*self.tile_size+self.tile_size//2, col*self.tile_size+self.tile_size//2)
        rad = self.tile_size//3
        for r in range(row*self.tile_size, (row+1)*self.tile_size):
            for c in range(col*self.tile_size, (col+1)*self.tile_size):
                if self.l2_dist((r,c),center) < rad:
                    img[r,c] = color
        return img
    def fill_coords(self, img, fn, color):
        """
        Fill pixels of an image with coordinates matching a filter function
        """

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                yf = (y + 0.5) / img.shape[0]
                xf = (x + 0.5) / img.shape[1]
                if fn(xf, yf):
                    img[y, x] = color

        return img

    def point_in_rect(self, xmin, xmax, ymin, ymax):
        def fn(x, y):
            return x >= xmin and x <= xmax and y >= ymin and y <= ymax
        return fn

    def point_in_circle(self, cx, cy, r):
        def fn(x, y):
            return (x-cx)*(x-cx) + (y-cy)*(y-cy) <= r * r
        return fn   
        



    def render(self):
        width_px = self.columns * self.tile_size
        height_px = self.rows * self.tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        # Color background grey
        img[:,:,:] = COLORS['black']


        #if self.lockeddoor = render x, else render y
        # if self.havekey = false, render x, else render y 

        # Color lava red


        for ss in self.ss_list:
            print(ss)
            img = self.fill_square(ss[0], ss[1], COLORS['yellow'], img)
        # Color goal state green
        img = self.fill_square(self.termination_state[0],self.termination_state[1],COLORS['green'],img)
        # Color start state green
        img = self.fill_square(self.start_state[0],self.start_state[1],COLORS['red'],img)
        for blocked in self.blocked_states:
            img = self.fill_square(blocked[0], blocked[1], COLORS['grey'], img) 
        # Draw lines in grid
        row_ticks = list(range(1,self.rows))
        col_ticks = list(range(1,self.columns))
        for r in row_ticks:
            img[r*self.tile_size,:] = COLORS['grey']
        for c in col_ticks:
            img[:,c*self.tile_size] = COLORS['grey']

    

        return img

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
env_class = grid()

img = env_class.render()
fig = plt.figure()
plt.imshow(img)
plt.axis('off')
# fig.text(.5, 0, "Task at student episode 0", ha='center')
# plt.xlabel('Task at student episodes 30 to 100')
plt.show()