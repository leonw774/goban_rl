import numpy as np
import go
from time import time
import pstats
import cProfile
from sys import getsizeof

measure_times = []

stats = pstats.Stats('mcts_search.profile', stream=open('stat.txt', 'w'))
stats.strip_dirs().sort_stats('time').print_stats()

# def test():
#     for _ in range(1000):
#         b, w = board.eval()

# board = go.Board(19, 6.5)
# for i in range(10000):
#     if (i + 1) % 160 == 0:
#         board.clear()
#     random_point = np.random.randint(19, size=(2))
#     add_stione = go.Stone(board, (random_point[0], random_point[1]))
    
#     t = time()
#     state = board.to_state()
#     b, w = board.eval()
#     # prof = cProfile.run("test()")
#     measure_times.append(time()-t)
#     # stats = pstats.Stats(prof)
#     # stats.strip_dirs().sort_stats('time').print_stats()
# print(np.sum(measure_times), np.mean(measure_times), np.std(measure_times))
