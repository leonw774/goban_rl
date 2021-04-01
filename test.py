import numpy as np
import go
from time import time
import pstats
import pickle
import cProfile
from sys import getsizeof

stats = pstats.Stats('mcts_search.profile', stream=open('stat.txt', 'w'))
stats.strip_dirs().sort_stats('time').print_stats()

board = go.Board(19, 6.5)
measure_times = []

# def test():
#     for _ in range(100):
#         v, p = board.eval()

# for i in range(200):
#     random_point = np.random.randint(19, size=(2))
#     add_stione = go.Stone(board, (random_point[0], random_point[1]))
# prof = cProfile.Profile()
# prof.enable()
# test()
# v, p = board.eval()
# prof.disable()
# # board.debugPrint()
# # outstr = ""
# # for i in range(19):
# #     for j in range(19):
# #         outstr += "%.2f " % (p[i+j*19])
# #     outstr += "\n"
# # print(outstr+str(p[-1]))
# print("value", v)
# stats = pstats.Stats(prof)
# stats.strip_dirs().sort_stats('time').print_stats()

""" 
for i in range(1000):
    if (i + 1) % 300 == 0:
        board.clear()
        # break
    t = time()
    random_point = np.random.randint(19, size=(2))
    add_stione = go.Stone(board, (random_point[0], random_point[1]))
    
    # state = board.state
    b, w = board.eval()
    measure_times.append(time()-t)
    # stats = pstats.Stats(prof)
    # stats.strip_dirs().sort_stats('time').print_stats()
b, w = board.eval()
print(b, w)
b, w, outstr = board.score(output=True)
print(outstr)
print(np.sum(measure_times), np.mean(measure_times), np.std(measure_times)) """
