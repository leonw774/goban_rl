import numpy as np
import argparse
from time import time, strftime
import pygame
import go
import playagent
import cProfile

parser = argparse.ArgumentParser()
parser.add_argument("--size", "-s", dest="size", default=19, type=int)
parser.add_argument("--playouts", "-p", dest="playouts", default=1600, type=int)
parser.add_argument("--selfplay", "-S", dest="self_play", action="store_true")
parser.add_argument("--playas", type=str, dest="playas", default="random", action="store")
args = parser.parse_args()

WHITE = go.WHITE
BLACK = go.BLACK
BOARD_SIZE = args.size
if args.playas == "b":
    PLAYAS = BLACK
elif args.playas == "w":
    PLAYAS = WHITE
else:
    PLAYAS = np.random.randint(2)

BACKGROUND = 'images/ramin.jpg'
COLOR = ((0, 0, 0), (255, 255, 255))
GRID_SIZE = 20
DRAW_BOARD_SIZE = (GRID_SIZE * BOARD_SIZE + 20, GRID_SIZE * BOARD_SIZE + 20)

if BOARD_SIZE <= 9:
    KOMI = 8.5
elif BOARD_SIZE <= 13:
    KOMI = 7.5
else:
    KOMI = 6.5

print("Board size:", BOARD_SIZE)
print("Komi:", KOMI)

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
        win, score_diff, outstr = self.score(output=True)
        logfile.write(outstr+"\n")

def main(ai_play_as):
    print("begin game")
    pass_count = 0
    steps = 0
    while True:
        pygame.time.wait(250)
        if ai_play_as == board.next:
            x, y, policy = model.decide(board, args.playouts)
            value = policy[x+BOARD_SIZE*y]
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
            steps += 1
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
                        steps += 1
                        print("player choose (%d, %d)"%(x, y))
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        pass_count += 1
                        break
        if pass_count >= 2:
            break
    print("game over")
    winner, score_diff, out_str = board.score(output=True)
    print(out_str)
    # end while True

def self_play():
    print("begin self play")
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
                cProfile.run("x, y, policy = model.decide(board, args.playouts)", "mcts_search.profile")
                #x, y, policy = model.decide_monte_carlo(board, args.playouts)
                print(time()-t)
                measure_times.append(time()-t)
                if y == BOARD_SIZE:
                    board.pass_move()
                    play_as = "B" if board.next == WHITE else "W"
                    intuition = policy[x+BOARD_SIZE*y]
                    print(play_as, "pass\t intuition:%.3f"%(intuition))
                    pass_count += 1
                    if pass_count >= 2: break
                elif y < 0:
                    board.pass_move()
                    play_as = "B" if board.next == WHITE else "W"
                    print(play_as, "resigned")
                    game_over = True
                else:
                    added_stone = Stone(board, (x, y))
                    if not added_stone.islegal:
                        value = policy[x+BOARD_SIZE*y]
                        print("B" if play_as == "W" else "W", "tried illegal move (%d, %d)\t value:%.3f"%(x, y, value))
                        # game_over = True
                        # break
                    else:
                        pass_count = 0
                        play_as = "B" if board.next == WHITE else "W"
                        intuition = policy[x+BOARD_SIZE*y]
                        print(play_as, "choose (%d, %d)\t intuition:%.3f"%(x, y, intuition))
                
    print("game over")
    winner, score_diff, out_str = board.score(output=True)
    print(out_str)
    print("decision time", np.sum(measure_times), np.mean(measure_times), np.std(measure_times))

if __name__ == '__main__':
    model = playagent.Agent(BOARD_SIZE)
    if args.self_play:
        pygame.init()
        pygame.display.set_caption('Go-Ai')
        screen = pygame.display.set_mode(DRAW_BOARD_SIZE, 0, 32)
        background = pygame.image.load(BACKGROUND).convert()
        board = Board(size=BOARD_SIZE, komi=KOMI)
        self_play()
    else:
        pygame.init()
        pygame.display.set_caption('Go-Ai')
        screen = pygame.display.set_mode(DRAW_BOARD_SIZE, 0, 32)
        background = pygame.image.load(BACKGROUND).convert()
        board = Board(size=BOARD_SIZE, komi=KOMI)
        main(ai_play_as=PLAYAS)
        
