import numpy as np
import go
from time import time
import pickle

measure_times = []

def masked_softmax(mask, x, temperture):
    if len(x.shape) != 1:
        print("softmax input must be 1-D numpy array")
        return
    # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
    # sometime the sum exceed 1.0 due to numerical rounding
    x = x.astype("float64")
    # stablize/normalize because if x too big will cause NAN when exp(x)
    normal_x = x - max(x)
    #x = (x - x.mean()) / (np.std(x) + 1e-9)
    mask_indice = np.argwhere(~mask)
    masked_x = normal_x[mask_indice]
    masked_softmax = np.exp(masked_x/temperture)/np.sum(np.exp(masked_x/temperture))
    softmax = np.zeros(x.shape)
    softmax[mask_indice] = masked_softmax
    if np.isnan(softmax).any():
        print("masked_instincts cantains NaN")
        print(x)
        print(normal_x)
        print(mask)
        print(temperture)
        print(softmax)
        exit()
    return softmax

def fast_rand_int_sample(size, p):
    """
    This function has problem: it samples 0 weighted 
        a: int. The sample would be np.arange(a)
        size: sample size
        p: 1-D array of weight
    """
    cumsum = np.cumsum(p)
    rand_uniform = np.random.random(size)
    return np.searchsorted(cumsum, rand_uniform)

""" board = go.Board(19, 6.5)
for i in range(10000):
    t = time()
    if i % 80 == 0 and  i > 0:
        board.score()
        board.clear()
    random_point = np.random.randint(19, size=(2))
    add_stione = go.Stone(board, (random_point[0], random_point[1]))
    board_state = pickle.dumps(board)
    measure_times.append(time()-t) """

for _ in range(10000):
    t = time()
    samples = fast_rand_int_sample(size=10, p=np.array([1/82]*82))
    measure_times.append(time()-t)
print(np.sum(measure_times), np.mean(measure_times), np.std(measure_times))
for _ in range(10000):
    t = time()
    samples = np.random.choice(a=82, size=10, p=np.array([1/82]*82))
    measure_times.append(time()-t)
print(np.sum(measure_times), np.mean(measure_times), np.std(measure_times))

# mask = np.array([False, True, True, False, True, True, False, True, False])
# x = np.array([0, 600, 0, 120, 70, 5, 180, 0, 0])
# print(masked_softmax(mask, x, 1.0))