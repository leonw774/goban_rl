#!/usr/bin/env python
# coding: utf-8

"""Goban made with Python, pygame and go.py.

This is a front-end for my go library 'go.py', handling drawing and
pygame-related activities. Together they form a fully working goban.

"""

"""
Edit by leow774 for keras ai training
"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import numpy as np
import argparse
from time import sleep
import pygame
import go
import playmodel
from sys import exit

BACKGROUND = 'images/ramin.jpg'
BOARD_SIZE = (410, 410)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--usemodel", type=str, default="", action="store")
parser.add_argument("--onlytest", action="store_true")
args = parser.parse_args()
EPOCHS = args.epochs
TEST_ONLY = (args.onlytest)

MAX_STEP = 540

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WIN_REWARD = 1
UNKNOWN_REWARD = 0.5
LOSE_REWARD = 0
KOMI = 6.5

class Stone(go.Stone):
    def __init__(self, board, point, color):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point, color)
        self.board.update_map(point, color)
        self.coords = (25 + self.point[0] * 20, 25 + self.point[1] * 20)
        self.draw()

    def draw(self):
        """Draw the stone as a circle."""
        pygame.draw.circle(screen, self.color, self.coords, 10, 0)
        pygame.display.update()

    def remove(self):
        """Remove the stone from board.
        Also remove this stone from board map"""
        blit_coords = (self.coords[0] - 10, self.coords[1] - 10)
        area_rect = pygame.Rect(blit_coords, (20, 20))
        screen.blit(background, blit_coords, area_rect)
        pygame.display.update()
        self.board.map[self.point] = [0.0, 0.0]
        super(Stone, self).remove()
        
    def find_group(self):
        """Find or create a group for the stone."""
        groups = []
        stones = self.board.search(points=self.neighbors)
        for stone in stones:
            if stone.color == self.color and stone.group not in groups:
                groups.append(stone.group)
        if not groups:
            group = Group(self.board, self)
            return group
        else:
            if len(groups) > 1:
                for group in groups[1:]:
                    groups[0].merge(group)
            groups[0].stones.append(self)
            return groups[0]

class Group(go.Group):
    def __init__(self, board, stone):
        """Create and initialize a new group.

        Arguments:
        board -- the board which this group resides in
        stone -- the initial stone in the group

        """
        super(Group, self).__init__(board, stone)
        self.color = stone.color
    
    def update_liberties(self):
        """Update the group's liberties.
        Return liberties, stone count
        As this method will remove the entire group if no liberties can
        be found, it should only be called once per turn.

        """
        liberties = []
        for stone in self.stones:
            for liberty in stone.liberties:
                liberties.append(liberty)
        self.liberties = set(liberties)  
        return len(self.liberties), len(self.stones), self.color
        
        
class Board(go.Board):
    def __init__(self):
        """Create, initialize and map an empty board.
        map is a numpy array representation of the board
        empty = (0, 0)
        black = (1, 0)
        white = (0, 1)
        """
        self.black_catch = 0
        self.white_catch = 0
        self.map = np.zeros((19, 19, 2))
        self.outline = pygame.Rect(25, 25, 360, 360)
        self.draw()
        super(Board, self).__init__()
    
    def is_gameover(self):
        """ Return winner if game is over, Return None if not"""
        empty_count = 0
        black_count = 0
        white_count = KOMI
        for i in range(19):
            for j in range(19):
                if np.max(board.map[i, j]) == 0:
                    empty_count += 1
                else:
                    black_count += board.map[i, j][0]
                    white_count += board.map[i, j][1]
        if empty_count <= 25:
            return BLACK if (black_count+self.white_catch*0.5 >= white_count+self.black_catch*0.5) else WHITE
        else:
            return None
    
    def update_map(self, point, color):
        if color == BLACK:
            self.map[point] = [1.0, 0.0]
        if color == WHITE:
            self.map[point] = [0.0, 1.0]
    
    def update_liberties(self, added_stone=None):
        """Updates the liberties of the entire board, group by group.
        Return True if it is a legal move, False if not

        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.

        """
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            liberties, stone_count, stone_color = group.update_liberties()
            if liberties == 0:
                group.remove()
                if stone_color == BLACK:
                    self.black_catch += stone_count
                else:
                    self.white_catch += stone_count
        if added_stone:
            liberties, stone_count, stone_color = added_stone.group.update_liberties()
            # if is suicide
            if liberties == 0:
                added_stone.remove()
                self.turn()
                return "Suicide"
        return None
    
    def clear(self):
        while self.groups:
            self.groups[0].remove()
        self.groups = []
        self.next = BLACK
        self.black_catch = 0
        self.white_catch = 0
       
    def draw(self):
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
        for i in range(18):
            for j in range(18):
                rect = pygame.Rect(25 + (20 * i), 25 + (20 * j), 20, 20)
                pygame.draw.rect(background, BLACK, rect, 1)
        for i in range(3):
            for j in range(3):
                coords = (85 + (120 * i), 85 + (120 * j))
                pygame.draw.circle(background, BLACK, coords, 5, 0)
        screen.blit(background, (0, 0))
        pygame.display.update()

def train():
    TEMPERATURE = 2.0
    MIN_TEMPERATURE = 0.1
    TEMPERATURE_DECAY = (MIN_TEMPERATURE/TEMPERATURE) ** (2/(3*EPOCHS))
    for epoch in range(EPOCHS*2):
        # when epoch is even, train as black
        # when epoch is odd, train as white
        train_as = (BLACK, WHITE)[epoch%2]
        print("epoch", epoch)
        steps = 0
        while (steps < MAX_STEP):
            old_map = board.map
            x, y, prob = model.decide(old_map, TEMPERATURE)
            # if is already a stone
            if board.search(point=(x, y)) != []:
                continue
            added_stone = Stone(board, (x, y), board.turn())
            # if is suicide
            if board.update_liberties(added_stone) == "Suicide":
                continue
            winner = board.is_gameover()
            if winner:
                reward = WIN_REWARD if winner == train_as else LOSE_REWARD
                model.record(point=(x, y),
                             old_map = old_map,
                             new_map = board.map,
                             reward = reward,
                             is_terminal = True)
                print("winner:", winner)
                break
            elif board.next != train_as:
                model.record(point=(x, y),
                             old_map = old_map,
                             new_map = board.map,
                             reward = UNKNOWN_REWARD,
                             is_terminal = False)
            steps += 1
            pygame.time.wait(int((2.0 - TEMPERATURE - MIN_TEMPERATURE)**3 * 50))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
        # end while
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        board.clear()
        model.learn()
        model.add_record()
        if epoch%2 == 1:
            model.save("model.h5")
            TEMPERATURE = max(MIN_TEMPERATURE, TEMPERATURE*TEMPERATURE_DECAY)

def test(ai_play_as):
    while True:
        pygame.time.wait(250)
        if ai_play_as == board.next:
            x, y, prob = model.decide(board.map, 0.1)
            print("model choose (%d, %d) for prob: %.4e"%(x, y, prob))
            if board.search(point=(x, y)) != []:
                continue
            added_stone = Stone(board, (x, y), board.turn())
            # if is suicide
            if board.update_liberties(added_stone) == "Suicide":
                continue
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and board.outline.collidepoint(event.pos):
                        x = int(round(((event.pos[0] - 25) / 20.0), 0))
                        y = int(round(((event.pos[1] - 25) / 20.0), 0))
                        #print(x, y)
                        if board.search(point=(x, y)) != []:
                            continue
                        added_stone = Stone(board, (x, y), board.turn())
                        board.update_liberties(added_stone)    

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Goban')
    screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
    background = pygame.image.load(BACKGROUND).convert()
    model = playmodel.ActorCritic(args.usemodel)
    board = Board()
    if not TEST_ONLY:
        train()
    test(ai_play_as=WHITE)
