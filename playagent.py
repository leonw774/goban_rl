import numpy as np
from time import time
import go
from montecarlo import MonteCarlo

WHITE = go.WHITE
BLACK = go.BLACK

def masked_softmax(mask, x, temperature):
    # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
    # sometime the sum exceed 1.0 due to numerical rounding
    x = x.astype("float64")
    # do not consider i if mask[i] == True
    masked_x = x[mask]
    # stablize/normalize because if x too big will cause NAN when exp(x)
    normal_masked_x = masked_x - max(masked_x)
    masked_softmax_x = np.exp(normal_masked_x/temperature) / np.sum(np.exp(normal_masked_x/temperature))
    softmax = np.zeros(x.shape)
    softmax[mask] = masked_softmax_x
    return softmax

class Agent():
    def __init__ (self, size) :    
        self.actor = None
        self.critic = None
        self.size = size
        self.size_square = size**2
        self.action_size = size**2 + 1

        self.resign_value = -0.5 # resign if determined value is low than this value

        # batch_size too large will cause exploration too high
        self.monte_carlo = MonteCarlo(self, simulation_num=1, simulation_depth=1)

    def get_valid_mask(self, board):
        valid_mask = np.zeros((self.action_size), dtype=bool)
        # transpose so that [x, y] become [x+y*size]
        valid_mask[:self.size_square] = np.transpose(np.sum(board.grid, axis=2)==0).flatten()
        for p in board.suicide_illegal.union(board.same_state_illegal):
            valid_mask[p[0]+p[1]*self.size] = False
        if not np.all(valid_mask): # not empty
            valid_mask[self.action_size-1] = True # can pass
        return valid_mask

    def decide(self, board, playout, temperature=0.01):
        prev_point = board.log[-1][1] if len(board.log) else (-1, 0)
        prev_action = prev_point[0] + prev_point[1] * self.size
        x, y, pi = self.monte_carlo.search(board, prev_action, int(playout), temperature)
        return x, y, pi
