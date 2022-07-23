from tilecodingAPI import IHT, tiles, tileswrap, hashcoords
import numpy as np
#from sklearn.preprocessing import normalize

class ACTileCoder:
    def __init__(self, iht_size=4096, num_tilings=176, num_tiles=4):
        """
        Initializes the Action Value Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        
    
    def get_tiles(self, action_value):
    
        """
        Takes in the q values of an RL#
        and returns a numpy array of active tiles.
        
        Arguments:
        action_value -- float
        returns:
        tiles - np.array, active tiles
        """
        # Use the ranges above and self.num_tiles to scale position and velocity to the range [0, 1]
        # then multiply that range with self.num_tiles so it scales from [0, num_tiles]
        minP=0
        maxP=1
        #minV=-.07
        #maxV=.07
        scaleP= maxP- minP
        #scaleV= maxV-minV

        #action_value = normalize(action_value)
        scaled_values = []
        #print(action_value)
        for value in action_value:
            #print('value', value)
            #print('value-minP',value-minP)
            #print('scale P', scaleP)
            value_scaled = ((value-minP)/(scaleP))*self.num_tiles
            #print('value_scaled', value_scaled)
            scaled_values.append(value_scaled)
        #velocity_scaled = ((velocity-minV)/(scaleV))*self.num_tiles
       
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        #print('scaled values', scaled_values)
        mytiles = tiles(self.iht, self.num_tilings, scaled_values)
        #print('mytiles', mytiles)
        return np.array(mytiles)