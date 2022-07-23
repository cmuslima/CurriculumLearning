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
    'white' : np.array([255, 255, 255]),
    'black' : np.array([0, 0, 0]),
    'orange': np.array([255,165,0])
}

class grid(gym.Env):
    def __init__(self):       
        #self.door = np.array([2,2])
        #maze
        self.start_state= np.array([10,4])  #target start state   
        self.termination_state= np.array([0,15]) 
        self.blocked_states= [(0,3), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (9,3), (10,3), (3,0), (3,1), (3,2), (7,0), (7,1), (7,2), (3,4), (3,5), (7,4), (7,6), (7,7), (3,7), (4,7), (5,7), (6,7), (8,7), (9,7), (10,7), (0,6), (1,6), (6,8), (6,9), (6,10), (0,10), (1,10), (2,10), (3,10),(8,10), (9,10),(10,10), (5,12), (5,13), (5,14), (5,15), (6,12), (7,12), (8,13), (8,14), (8,15)]
        self.ss_list = [(10,4),(1,1), (5,1), (9,1),(7,5), (3,6), ([5,10]), (2,12),(10,8),(10,14), (7,13)]
        self.rows=11
        self.columns=16
        #four rooms
        # self.start_state= np.array([0,0])  #target start state   
        # self.termination_state= np.array([5,4])    

        # self.blocked_states= [(3,0), (0,3), (6,3), (3,6), (2,3), (4,3), (3,2), (3,3), (3,4)]
        # self.ss_list = [(0,0),(0,1), (1,1), (3,1),(5,1), (5,3), (5,5), (3,5),(1,5),(1,3)]
        # self.rows=7
        # self.columns=7
        # self.blocked_states= [(5,0),(5,1), (5,2), (6,2), (7,2), (8,2), (9,2), (10,2),(10,3),\
        # (10,4),(10,5),(10,6),(10,7),(11,7),(12,7),(13,7),(13,6),(13,5),\
        # (13,4),(14,4),(15,4),(16,4),(17,4),(18,4),(18,5),(18,6),(18,7),\
        # (18,8),(18,9),(19,9),(0,6),(1,6),(2,6),(3,6),(4,6),(4,7),(4,8),\
        # (4,9),(5,9),(6,9),(7,9),(8,9),(9,9),(9,10),(9,11),(9,12),(9,13),(9,14),\
        # (10,14),(11,14),(12,14),(13,14),(14,14),(14,10),(14,11),(14,12),(14,13),\
        # (15,10),(15,11),(15,12),(15,13),(15,15),(15,14),(15,15),(15,17),(15,18),\
        # (16,18),(17,18),(18,18),(19,18)]   



        
       
    

        self.tile_size = 32





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




    def render(self, probs):
        width_px = self.columns * self.tile_size
        height_px = self.rows * self.tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        # Color background grey
        img[:,:,:] = COLORS['black']


        #if self.lockeddoor = render x, else render y
        # if self.havekey = false, render x, else render y 

        # Color lava red
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='hot')
        # Color start state blue
        img = self.fill_square(self.start_state[0],self.start_state[1],COLORS['blue'],img)
        # for idx, ss in enumerate(self.ss_list):
            

        #     img = self.fill_square(ss[0], ss[1], COLORS['yellow'], img)
        #     try:
        #         probability = probs[idx]
        #     except:
        #         probability = 0
        #     img = self.fill_square(ss[0], ss[1], np.array(mapper.to_rgba(probability)[:3])*255, img)
        # Color goal state green
        img = self.fill_square(self.termination_state[0],self.termination_state[1],COLORS['green'],img)
        #ßimg = self.fill_square(self.start_state[0],self.start_state[1],COLORS['blue'],img)
        for blocked in self.blocked_states:
            img = self.fill_square(blocked[0], blocked[1], COLORS['grey'], img) 
        # Draw lines in grid
        row_ticks = list(range(1,self.rows))
        col_ticks = list(range(1,self.columns))
        for r in row_ticks:
            img[r*self.tile_size,:] = COLORS['grey']
        for c in col_ticks:
            img[:,c*self.tile_size] = COLORS['grey']

        # img[2*self.tile_size:] = COLORS['blue']
        # img[:,2*self.tile_size] = COLORS['blue']
        # img[1*self.tile_size:] = COLORS['blue']
        # img[:,1*self.tile_size] = COLORS['blue']
        #img[5*self.tile_size,:] = COLORS['grey']

        


    

        return img, mapper

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import matplotlib.cm as cm

def render_env(probs):
    env_class = grid()

    img, mapper = env_class.render(probs)
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')
    # fig.text(.5, 0, "Task at student episode 0", ha='center')
    # plt.xlabel('Task at student episodes 30 to 100')
    plt.colorbar(mappable=mapper)
    #plt.title('Mid stages of curriculum')
    plt.show()

