#!/usr/bin/env python
# coding: utf-8

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import numpy as np
import argparse
from time import sleep
import pygame
import go
import playmodel
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", default=1000, type=int)
parser.add_argument("--size", "-s", dest="size", default=19, type=int)
parser.add_argument("--use-model", dest="use_model", type=str, default="", action="store")
parser.add_argument("--test", type=str, default="", action="store")
args = parser.parse_args()
EPOCHS = args.epochs
TEST_ONLY = (args.test == "b" or args.test == "w")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOARD_SIZE = args.size

BACKGROUND = 'images/ramin.jpg'
GRID_SIZE = 20
DRAW_BOARD_SIZE = (GRID_SIZE * BOARD_SIZE + 20, GRID_SIZE * BOARD_SIZE + 20)

MAX_STEP = 2*BOARD_SIZE**2
B_WIN_REWARD = 1.0
UNKNOWN_REWARD = 0.0
W_WIN_REWARD = -1.0
if BOARD_SIZE <= 9:
    KOMI = 0.75
elif BOARD_SIZE <= 13:
    KOMI = 3.75
else:
    KOMI = 6.5

class Stone(go.Stone):
    def __init__(self, board, point, color, is_drawn = True):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point, color)
        self.board.update_map(point, color)
        self.is_drawn = is_drawn and board.is_drawn
        if self.is_drawn:
            self.coords = (25 + self.point[0] * GRID_SIZE, 25 + self.point[1] * GRID_SIZE)
            self.draw()
    
    def draw(self):
        """Draw the stone as a circle."""
        pygame.draw.circle(screen, self.color, self.coords, 10, 0)
        pygame.display.update()
    
    def remove(self):
        """Remove the stone from board.
        Also remove this stone from board map"""
        if self.is_drawn:
            blit_coords = (self.coords[0] - 10, self.coords[1] - 10)
            area_rect = pygame.Rect(blit_coords, (20, 20))
            screen.blit(background, blit_coords, area_rect)
            pygame.display.update()
        self.board.map[self.point] = [0.0, 0.0]
        super(Stone, self).remove()

        
class Board(go.Board):
    def __init__(self, size, is_drawn=True):
        """Create, initialize and map an empty board.
        map is a numpy array representation of the board
        empty = (0, 0)
        black = (1, 0)
        white = (0, 1)
        """
        self.is_drawn = is_drawn
        super(Board, self).__init__(size)
        if is_drawn:
            self.outline = pygame.Rect(25, 25, DRAW_BOARD_SIZE[0]-40, DRAW_BOARD_SIZE[1]-40)
            self.draw()
    
    def is_gameover(self, pass_count=0):
        """ Return winner if game is over, Return None if not"""
        empty_count = 0
        b_count = 0
        w_count = KOMI
        for i in range(self.size):
            for j in range(self.size):
                if np.max(board.map[i, j]) == 0:
                    empty_count += 1
                else:
                    b_count += board.map[i, j][0]
                    w_count += board.map[i, j][1]

        if empty_count-np.argwhere(self.illegal).shape[0]<=3 or pass_count==2:
            return BLACK if (b_count+self.w_catched*0.5 >= w_count+self.b_catched*0.5) else WHITE
        elif (b_count+self.w_catched*0.5 - w_count+self.b_catched*0.5) > 60:
            return BLACK
        elif (b_count+self.w_catched*0.5 - w_count+self.b_catched*0.5) < -60:
            return WHITE
        else:
            return None
       
    def draw(self):
        """Draw the board to the background and blit it to the screen.

        The board is drawn by first drawing the outline, then the grid
        and finally by adding hoshi to the board. All these
        operations are done with pygame's draw functions.

        This method should only be called once, when initializing the
        board.

        """
        pygame.draw.rect(background, BLACK, self.outline, 3)
        # Outline is inflated here for future use as a collidebox for the mouse
        self.outline.inflate_ip(GRID_SIZE, GRID_SIZE)
        for i in range(self.size-1):
            for j in range(self.size-1):
                rect = pygame.Rect(25 + (GRID_SIZE * i), 25 + (GRID_SIZE * j), GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(background, BLACK, rect, 1)
        if self.size == 19:
            for i in range(3):
                for j in range(3):
                    coords = (85 + (120 * i), 85 + (120 * j))
                    pygame.draw.circle(background, BLACK, coords, 5, 0)
        screen.blit(background, (0, 0))
        pygame.display.update()

def train():
    MAX_TEMPERATURE = 10.0
    MIN_TEMPERATURE = 0.1
    TEMPERATURE = MAX_TEMPERATURE
    TEMPERATURE_DECAY = (MIN_TEMPERATURE/TEMPERATURE) ** (EPOCHS/2)
    black_win_count = 0
    for epoch in range(EPOCHS*2):
        steps = 0
        pass_count = 0
        # only record as one side because white has "less steps" adventage
        trainas = (BLACK, WHITE)[epoch%2]
        while (steps < MAX_STEP):
            try_steps = 0
            while (try_steps < MAX_STEP - steps):
                pre_map = board.map
                x, y, instinct = model.decide_random(board, TEMPERATURE)
                if x==-1 and y==-1:
                    pass_count += 1
                    board.turn()
                    break
                else:
                    pass_count = 0
                try_steps += 1
                added_stone = Stone(board, (x, y), board.turn())
                # if is illegal
                if board.update_liberties(added_stone) != "illegal":
                    break
            # end while try
            winner = board.is_gameover(pass_count)
            if winner:
                if winner==BLACK:
                    reward = B_WIN_REWARD
                    black_win_count += 1
                else:
                    reward = W_WIN_REWARD
                model.record((x, y), pre_map, board.map, reward, True)
                if epoch%20==0:
                    print("epoch", epoch, "Black win rate:", black_win_count/(epoch+1))
                break
            elif board.next != trainas:
                model.record((x, y), pre_map, board.map, UNKNOWN_REWARD, False)
            steps += 1
        # end while game
        board.clear()
        model.learn(verbose=(epoch%20==0))
        model.add_record()
        if epoch>1 and epoch%20==0:
            model.save("model.h5")
            TEMPERATURE = max(MIN_TEMPERATURE, TEMPERATURE*TEMPERATURE_DECAY)

def test(ai_play_as):
    print("begin test")
    print("use model:", args.use_model)
    print("value:", model.get_value(board.map))
    while True:
        pygame.time.wait(250)
        if ai_play_as == board.next:
            old_board = board.map
            x, y, instinct, win_rate = model.decide_tree_search(board, temperature=0.01, depth=2, kth=4)
            if x==-1 and y==-1:
                print("model passes")
                board.turn()
                continue
            else:
                print("model choose (%d, %d)\ninstinct:%.4e win rate:%.3f"%(x, y, instinct, win_rate))
            if board.search(point=(x, y)) != []:
                continue
            added_stone = Stone(board, (x, y), board.turn())
            # if is suicide
            if board.update_liberties(added_stone) == "illegal":
                continue
            print("value:", model.get_value(old_board)[x+BOARD_SIZE*y])
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and board.outline.collidepoint(event.pos):
                        old_board = board.map
                        x = int(round(((event.pos[0] - 25) / GRID_SIZE), 0))
                        y = int(round(((event.pos[1] - 25) / GRID_SIZE), 0))
                        #print(x, y)
                        if board.search(point=(x, y)) != []:
                            continue
                        added_stone = Stone(board, (x, y), board.turn())
                        board.update_liberties(added_stone)
                    print("player choose (%d, %d)"%(x, y))
                    print("win rate:%0.3f"%win_rate, "value:", model.get_value(old_board)[x+BOARD_SIZE*y]) 

if __name__ == '__main__':
    model = playmodel.ActorCritic(BOARD_SIZE, args.use_model)
    if not TEST_ONLY:
        board = Board(size=BOARD_SIZE, is_drawn=False)
        train()
    else:
        pygame.init()
        pygame.display.set_caption('Go-Ai')
        screen = pygame.display.set_mode(DRAW_BOARD_SIZE, 0, 32)
        background = pygame.image.load(BACKGROUND).convert()
        board = Board(size=BOARD_SIZE)
        test(ai_play_as=(BLACK if args.test=="b" or args.test=="black" else WHITE))

