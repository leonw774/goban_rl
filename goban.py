#!/usr/bin/env python
# coding: utf-8

"""Goban made with Python, pygame and go.py.

This is a front-end for my go library 'go.py', handling drawing and
pygame-related activities. Together they form a fully working goban.

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import pygame
import pickle
import numpy as np
from copy import deepcopy 
import go
from board_eval import board_eval
from time import time
import random
from sys import exit, getsizeof

BACKGROUND = 'images/ramin.jpg'
BOARD_SIZE = 19
KOMI = 6.5
GRID_SIZE = 25
DRAW_BOARD_SIZE = (GRID_SIZE * BOARD_SIZE + GRID_SIZE, GRID_SIZE * BOARD_SIZE + GRID_SIZE)

WHITE = go.WHITE
BLACK = go.BLACK
COLOR = ((0, 0, 0), (255, 255, 255))

class Stone(go.Stone):
    def __init__(self, board, point):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point)
        board.draw_stones()

class Board(go.Board):
    def __init__(self, is_debug):
        """Create, initialize and draw an empty board."""
        super(Board, self).__init__(BOARD_SIZE, 6.5, is_debug=is_debug)
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
        screen.blit(background, (0, 0))
        for g in self.groups:
            for p in g.stones:
                coords = (5+GRID_SIZE+p[0]*GRID_SIZE, 5+GRID_SIZE+p[1]*GRID_SIZE)
                pygame.draw.circle(screen, COLOR[g.color], coords, GRID_SIZE//2, 0)
        pygame.display.update()

def main():
    #measure_times = []
    while True:
        pygame.time.wait(250)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and board.outline.collidepoint(event.pos):
                    x = int(round(((event.pos[0]-GRID_SIZE-5) / GRID_SIZE), 0))
                    y = int(round(((event.pos[1]-GRID_SIZE-5) / GRID_SIZE), 0))
                    print(x, y)
                    added_stone = Stone(board, (x, y))
                    w, b = board.eval()
                    outstring = ""
                    for i in range(BOARD_SIZE):
                        for j in range(BOARD_SIZE):
                            if eval_grid[j, i] > 0:
                                outstring += "X "
                            elif eval_grid[j, i] < 0:
                                outstring += "O "
                            else:
                                outstring += "  "
                        outstring += "\n"
                    print(outstring)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    winner, score_diff, out_str = board.score(output=True)
                    print(out_str)
                    #print(np.sum(measure_times), np.mean(measure_times), np.std(measure_times))
                    return
    
if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Goban')
    screen = pygame.display.set_mode(DRAW_BOARD_SIZE, 0, 32)
    background = pygame.image.load(BACKGROUND).convert()
    board = Board(is_debug = False)
    main()
