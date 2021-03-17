import numpy as np
import go
from time import time
import pickle
import pstats
from board_eval import board_eval
from sys import getsizeof

measure_times = []

# stats = pstats.Stats('mcts_search.profile')
# stats.strip_dirs().sort_stats('time').print_stats(30)

board = go.Board(19, 6.5)
for i in range(500):
    if (i + 1) % 160 == 0:
        board.clear()
    random_point = np.random.randint(19, size=(2))
    add_stione = go.Stone(board, (random_point[0], random_point[1]))
    
    t = time()
    state = board.to_state()
    eval_grid, b, w = board_eval(state)
    measure_times.append(time()-t)
print(np.sum(measure_times), np.mean(measure_times), np.std(measure_times))
"""
measure_times = []
board.clear()
for i in range(10000):
    if i % 80 == 0 and  i > 0:
        
        board.clear()
    random_point = np.random.randint(19, size=(2))
    add_stione = go.Stone(board, (random_point[0], random_point[1]))
    t = time()
    state = board.to_state()
    
    board_copy = go.board_from_state(state)
    board = board_copy
    measure_times.append(time()-t)
    assert board.grid_hash() == board_copy.grid_hash()
print(getsizeof(state))
print(np.sum(measure_times), np.mean(measure_times), np.std(measure_times)) """