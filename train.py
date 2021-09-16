import numpy as np
import argparse
from time import time, strftime
import go
import playmodel
import cProfile

parser = argparse.ArgumentParser()
parser.add_argument("--episode", "-e", default=10000, type=int)
parser.add_argument("--output-intv", "-o", dest="output_intv", default=1000, type=int)
parser.add_argument("--size", "-s", dest="size", default=19, type=int)
parser.add_argument("--playouts", "-p", dest="playouts", default=256, type=int)

args = parser.parse_args()

# training parameters
EPOCHS = args.episode
PRINT_INTV = args.output_intv

WHITE = go.WHITE
BLACK = go.BLACK
BOARD_SIZE = args.size

if BOARD_SIZE <= 9:
    KOMI = 5.5
elif BOARD_SIZE <= 13:
    KOMI = 6.5
else:
    KOMI = 7.5

def train():
    open("log.txt", "w")
    record_times = []
    b_win_count = 0
    playouts = args.playouts

    for episode in range(EPOCHS):
        #temperature = 1.0
        steps = 0
        pass_count = 0
        winner = None
        reward = 0 # reward is viewed from BLACK

        while True:
            t = time()
            temperature = max(0.1, (1 + 1 / BOARD_SIZE) ** (-steps / 2)) # more step, less temperature
            playouts = max(args.playouts // 2, int(playouts * 0.9)) # more step, less playouts
            prev_grid = board.grid.copy()
            if playouts > 0:
                x, y = model.decide_monte_carlo(board, playouts)
            else:
                x, y = model.decide(board, temperature)
            if y == BOARD_SIZE: # pass is indexed #size_square --> y = pass//BOARD_SIZE = BOARD_SIZE
                pass_count += 1
                board.pass_move()
            else:
                added_stone = go.Stone(board, (x, y))
                if added_stone.islegal:
                    pass_count = 0
                else:
                    continue
            
            if pass_count >= 2:
                winner, score_diff = board.score()
                reward = model.WIN_REWARD if winner == BLACK else model.LOSE_REWARD # reward is viewd from BLACK
                board.log_endgame(winner, "by " + str(score_diff))
                if winner == BLACK:
                    b_win_count += 1

            model.push_step(prev_grid, x + y * BOARD_SIZE, board.grid.copy())
            if winner is not None:
                break
            steps += 1
            record_times.append(time()-t)
        # end while game
        #temperature = max(min_temperature, initial_temperature / (1 + temperature_decay * episode))
        model.enqueue_new_record(reward)
        model.learn(verbose = ((episode + 1) % PRINT_INTV == 0))

        if (episode + 1) % PRINT_INTV == 0:
            model.save(str(BOARD_SIZE)+"_tmp.h5")
            print("episode: %d\t B win rate: %.3f"%(episode, b_win_count/(episode+1)))
            board.write_log_file(open("log.txt", "a"))
            print("decide + update time: total %.4f, mean %.4f" % (np.sum(record_times), np.mean(record_times)))
        board.clear()
    # end for episodes
    print(strftime(str(BOARD_SIZE)+"_%Y%m%d%H%M"))
    model.save(strftime(str(BOARD_SIZE)+"_%Y%m%d%H%M")+".h5")

if __name__ == '__main__':
    model = playmodel.ActorCritic(BOARD_SIZE)
    board = go.Board(size=BOARD_SIZE, komi=KOMI)
    # cProfile.run("train()", "train.profile")
    train()