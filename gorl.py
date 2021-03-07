import numpy as np
import argparse
from time import time, strftime
import pygame
import go
import playmodel
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", default=10000, type=int)
parser.add_argument("--out-intv", "-o", dest="out_intv", default=1000, type=int)
parser.add_argument("--size", "-s", dest="size", default=19, type=int)
parser.add_argument("--playout", "-p", dest="playouts", default=1024, type=int)
parser.add_argument("--use-model", "-m", dest="use_model", type=str, default="", action="store")
parser.add_argument("--self-play", dest="self_play", action="store_true")
parser.add_argument("--test", type=str, default="", action="store")
args = parser.parse_args()

EPOCHS = args.epochs
LAERN_THRESHOLD = 10
TRAIN_RECORD_SIZE = 4
TEST_ONLY = (args.test == "b" or args.test == "w")
PRINT_INTV = args.out_intv

WHITE = go.WHITE
BLACK = go.BLACK
BOARD_SIZE = args.size

BACKGROUND = 'images/ramin.jpg'
COLOR = ((0, 0, 0), (255, 255, 255))
GRID_SIZE = 20
DRAW_BOARD_SIZE = (GRID_SIZE * BOARD_SIZE + 20, GRID_SIZE * BOARD_SIZE + 20)

MAX_STEP = 2 * BOARD_SIZE**2
#ILLEGAL_PUNISHMENT = -4

if BOARD_SIZE <= 9:
    KOMI = 8.5
elif BOARD_SIZE <= 13:
    KOMI = 7.5
else:
    KOMI = 6.5

class Stone(go.Stone):
    def __init__(self, board, point):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point)
        if board.is_drawn and self.islegal:
            board.draw_stones()

class Board(go.Board):
    def __init__(self, size, komi, is_drawn=True):
        """Create, initialize and draw an empty board.
        """
        self.is_drawn = is_drawn
        super(Board, self).__init__(size, komi)
        if is_drawn:
            self.outline = pygame.Rect(GRID_SIZE+5, GRID_SIZE+5, DRAW_BOARD_SIZE[0]-GRID_SIZE*2, DRAW_BOARD_SIZE[1]-GRID_SIZE*2)
            self.draw_board()
    
    def draw_board(self):
        """Draw the board to the background and blit it to the screen.

        The board is drawn by first drawing the outline, then the 19x19
        grid and finally by adding hoshi to the board. All these
        operations are done with pygame's draw functions.

        This method should only be called once, when initializing the
        board.
        """
        pygame.draw.rect(background, BLACK, self.outline, 3)
        # Outline is inflated here for future use as a collidebox for the mouse
        self.outline.inflate_ip(20, 20)
        for i in range(self.size-1):
            for j in range(self.size-1):
                rect = pygame.Rect(5+GRID_SIZE+(GRID_SIZE*i), 5+GRID_SIZE+(GRID_SIZE*j), GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(background, COLOR[BLACK], rect, 1)
        if self.size >= 13:
            for i in range(3):
                for j in range(3):
                    coords = (5+4*GRID_SIZE+(GRID_SIZE*6*i), 5+4*GRID_SIZE+(GRID_SIZE*6*j))
                    pygame.draw.circle(background, COLOR[BLACK], coords, 5, 0)
        screen.blit(background, (0, 0))
        pygame.display.update()
    
    def draw_stones(self):
        """
        This method is called every time the board is updated by add_stone() and it is succesful
        """
        screen.blit(background, (0, 0))
        for g in self.groups:
            for p in g.stones:
                coords = (5+GRID_SIZE+p[0]*GRID_SIZE, 5+GRID_SIZE+p[1]*GRID_SIZE)
                pygame.draw.circle(screen, COLOR[g.color], coords, GRID_SIZE//2, 0)
        pygame.display.update()

    def write_game_log(self, logfile):
        COLOR_CHAR = ("B", "W")
        logstr = ""
        for entry in self.log:
            logstr += COLOR_CHAR[entry[0]] + " " + str(entry[1]) + "\n"
        logfile.write(logstr)
        win, w, b, outstr = self.score(output=True)
        logfile.write(outstr+"\n")

def train():
    open("log.txt", "w")
    record_times = []
    b_win_count = 0
    record_as = BLACK
    playout = args.playouts
    playout_diff = 1
    playout_limit = args.playouts / 2

    for epoch in range(EPOCHS):
        playout = max(playout-playout_diff, playout_limit)
        #temperature = 1.0
        steps = 0
        pass_count = 0
        winner = None
        b_score = 0
        w_score = 0
        reward = 0 # black: positive; white: negtive
        # because same board's reward of BLACK == -reward of WHITE
        # have to record seperatly 
        #record_as = WHITE if record_as == BLACK else BLACK
        #model.set_record_player(record_as)
        while (steps <= MAX_STEP):
            temperature = (1 + 1/BOARD_SIZE) ** -steps
            t = time()
            play_as = board.next
            pre_grid = board.grid.copy()
            x, y, value = model.decide_monte_carlo(board, playout, temperature)
            #x, y, value = model.decide_minimax(board, depth=1, kth=4)
            #x, y, instinct = model.decide_instinct(board, temperature)
            if x >= BOARD_SIZE or y >= BOARD_SIZE:
                pass_count += 1
                board.pass_move()
            else:
                added_stone = Stone(board, (x, y))
                if added_stone.islegal:
                    pass_count = 0
                else:
                    continue
            record_times.append(time()-t)
            if pass_count >= 2 or steps == MAX_STEP:
                winner, b_score, w_score = board.score()
                reward = 1 if play_as == winner else -1 # if use tanh
                #reward = 1 if play_as == winner else 0 # if use sigmoid
                #reward = b_score - w_score
                board.log_endgame(winner, "by " + str(b_score - w_score))
                if winner == BLACK:
                    b_win_count += 1
                
            if winner is not None:
                if winner != record_as: model.pop_step()
                model.push_step((x, y), pre_grid, board.grid, reward)
                #print("reward:", reward, "steps", steps)
                break
            elif play_as == record_as:
            #else:
                model.push_step((x, y), pre_grid, board.grid, reward)
            steps += 1
        # end while game
        #temperature = max(min_temperature, initial_temperature / (1 + temperature_decay * epoch))
        model.enqueue_new_record()
        model.monte_carlo.clear_visit()
        if epoch > LAERN_THRESHOLD:
            model.learn(learn_record_size=TRAIN_RECORD_SIZE, verbose=((epoch+1)%PRINT_INTV==0))
        
        if (epoch+1) % PRINT_INTV == 0:
            model.save("model-tmp.h5")
            print("epoch: %d\t B win rate: %.3f"%(epoch, b_win_count/(epoch+1)))
            board.write_game_log(open("log.txt", "a"))
            print("decide + update time", np.sum(record_times), np.mean(record_times), np.std(record_times))
        board.clear()
    # end for epochs
    print(strftime("%Y%m%d%H%M"))
    model.save("model_"+strftime("%Y%m%d%H%M")+".h5")

def test(ai_play_as):
    print("begin test")
    playout = args.playouts
    playout_decrease = 1
    playout_limit = args.playouts / 2
    pass_count = 0
    while True:
        pygame.time.wait(250)
        if ai_play_as == board.next:
            #print("value:", model.get_value(board.grid))
            #print("instinct:", model.get_intuitions(board.grid))
            x, y, value = model.decide_minimax(board, depth=4, kth=4)
            if x >= BOARD_SIZE or y >= BOARD_SIZE:
                print("model passes\tvalue:%.3f"%(value))
                pass_count += 1
                board.pass_move()
            else:
                print("model choose (%d, %d)\tvalue:%.3f"%(x, y, value))
                pass_count = 0
                added_stone = Stone(board, (x, y))
                if not added_stone.islegal:
                    print("model tried illegal move")
                    break
            playout = max(playout-playout_decrease, playout_limit)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and board.outline.collidepoint(event.pos):
                        x = int(round(((event.pos[0]-GRID_SIZE-5) / GRID_SIZE), 0))
                        y = int(round(((event.pos[1]-GRID_SIZE-5) / GRID_SIZE), 0))
                        added_stone = Stone(board, (x, y))
                        pass_count = 0
                        value = model.get_value(board.grid)
                        print("player choose (%d, %d)\tvalue:%.3f"%(x, y, value))
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        pass_count += 1
                        break
        if pass_count >= 2:
            break
    print("game over")
    winner, b, w, out_str = board.score(output=True)
    print(out_str)
    # end while True

def self_play():
    print("begin self play")
    playout = args.playouts
    playout_decrease = 1
    playout_limit = args.playouts / 2
    pass_count = 0
    measure_times = []
    game_over = False
    while not game_over and pass_count < 2:
        pygame.time.wait(250)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    game_over = True
                    continue
                t=time()
                x, y, value = model.decide_monte_carlo(board, playout)
                print(time()-t)
                measure_times.append(time()-t)
                #x, y, value = model.decide_instinct(board, 0.1)
                if x >= BOARD_SIZE or y >= BOARD_SIZE:
                    pass_count += 1
                    if pass_count >= 2: break
                    board.pass_move()
                    play_as = "B" if board.next == WHITE else "W"
                    print(play_as, "pass\tvalue:%.3f"%(value))
                else:
                    added_stone = Stone(board, (x, y))
                    if not added_stone.islegal:
                        print(play_as, "tried illegal move (%d, %d)\t value:%.3f"%(x, y, value))
                        # game_over = True
                        # break
                    else:
                        pass_count = 0
                        play_as = "B" if board.next == WHITE else "W"
                        print(play_as, "choose (%d, %d)\t value:%.3f"%(x, y, value))
                playout = max(playout-playout_decrease, playout_limit)
                
    print("game over")
    winner, b, w, out_str = board.score(output=True)
    print(out_str)
    print("decision time", np.sum(measure_times), np.mean(measure_times), np.std(measure_times))

if __name__ == '__main__':
    model = playmodel.ActorCritic(BOARD_SIZE, LAERN_THRESHOLD, args.use_model)
    print("use model:", args.use_model)
    if TEST_ONLY:
        pygame.init()
        pygame.display.set_caption('Go-Ai')
        screen = pygame.display.set_mode(DRAW_BOARD_SIZE, 0, 32)
        background = pygame.image.load(BACKGROUND).convert()
        board = Board(size=BOARD_SIZE, komi=KOMI)
        test(ai_play_as=(BLACK if args.test=="b" or args.test=="black" else WHITE))
    elif args.self_play:
        pygame.init()
        pygame.display.set_caption('Go-Ai')
        screen = pygame.display.set_mode(DRAW_BOARD_SIZE, 0, 32)
        background = pygame.image.load(BACKGROUND).convert()
        board = Board(size=BOARD_SIZE, komi=KOMI)
        self_play()
    else:
        board = Board(size=BOARD_SIZE, komi=KOMI, is_drawn=False)
        train()

