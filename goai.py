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
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--use-model", dest="use_model", type=str, default="", action="store")
parser.add_argument("--test", type=str, default="", action="store")
args = parser.parse_args()
EPOCHS = args.epochs
TEST_ONLY = (args.test == "b" or args.test == "w")

MAX_STEP = 720

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

B_WIN_REWARD = 1.0
UNKNOWN_REWARD = 0.0
W_WIN_REWARD = -1.0
KOMI = 4.5

class Stone(go.Stone):
    def __init__(self, board, point, color, is_draw = True):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point, color)
        self.board.update_map(point, color)
        self.is_drawn = is_draw
        if self.is_drawn:
            self.coords = (25 + self.point[0] * 20, 25 + self.point[1] * 20)
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
    
    @property
    def liberties(self):
        """Find and return the liberties of the stone."""
        neighbors = self.neighbors
        liberties = []
        for neighbor in neighbors:
            if np.max(self.board.map[neighbor[0], neighbor[1]]) == 0:
                liberties.append(neighbor)
        return liberties
    
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
        Return liberties count
        As this method will remove the entire group if no liberties can
        be found, it should only be called once per turn.

        """
        liberties = []
        for stone in self.stones:
            for lib in stone.liberties:
                liberties.append(lib)
        liberties = set(liberties)
        if len(liberties) == 0:
            return 0
        return len(liberties)
        
class Board(go.Board):
    def __init__(self):
        """Create, initialize and map an empty board.
        map is a numpy array representation of the board
        empty = (0, 0)
        black = (1, 0)
        white = (0, 1)
        """
        self.black_catched = 0
        self.white_catched = 0
        self.map = np.zeros((19, 19, 2))
        self.illegal = np.full((19, 19, 2), False)
        self.outline = pygame.Rect(25, 25, 360, 360)
        self.draw()
        super(Board, self).__init__()
    
    def is_gameover(self, pass_count=0):
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
        if empty_count-np.argwhere(self.illegal).shape[0]<=3 or pass_count==2:
            return BLACK if (black_count+self.white_catched*0.5 >= white_count+self.black_catched*0.5) else WHITE
        elif (black_count+self.white_catched*0.5 - white_count+self.black_catched*0.5) > 60:
            return BLACK
        elif (black_count+self.white_catched*0.5 - white_count+self.black_catched*0.5) < -60:
            return WHITE
        else:
            return None
    
    def has_stone(self, point):
        return np.max(self.map[point]) == 1
    
    def update_illegal(self):
        empty_points = np.argwhere(np.max(self.map, axis=2)==0)
        next_color = 0 if self.next == BLACK else 1
        self.illegal[:, :, next_color] = False
        #print(empty_points)
        for e in empty_points:
            neighbors = [(e[0] - 1, e[1]),
                         (e[0] + 1, e[1]),
                         (e[0], e[1] - 1),
                         (e[0], e[1] + 1)]
            neighbors = [n for n in neighbors if ((0<=n[0]<19) and (0<=n[1]<19))]
            
            if all([self.has_stone(x) for x in neighbors]):
                neighbor_stones = self.search(points=neighbors)
                is_suicide = False
                # suicide: made itself killed or made neighboring same color stone killed
                is_suicide = all([neighbor_stone.color != self.next for neighbor_stone in neighbor_stones])
                if not is_suicide:
                    for neighbor_stone in neighbor_stones:
                        if neighbor_stone.color == self.next:
                            if neighbor_stone.group.update_liberties() == 1:
                                is_suicide = True
                                break
                if is_suicide:
                    is_suicide_kill = False
                    for neighbor_stone in neighbor_stones:
                        #print("lib test - looking at:", neighbor_stone.group)
                        if neighbor_stone.color != self.next:
                            if neighbor_stone.group.update_liberties() == 1:
                                is_suicide_kill = True
                                break
                    #print("next:", self.next, "exam:", e, is_suicide_kill)
                    if not is_suicide_kill:
                        self.illegal[e[0], e[1], next_color] = True
        #print(np.argwhere(self.illegal))
    
    def update_map(self, point, color):
        if color == BLACK:
            self.map[point] = [1.0, 0.0]
        if color == WHITE:
            self.map[point] = [0.0, 1.0]
    
    def update_liberties(self, added_stone=None):
        """Updates the liberties of the entire board, group by group.
        Return None if it is a legal move, Return string "illegal" if not

        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.

        """
        if added_stone:
            if self.illegal[added_stone.point[0], added_stone.point[1], 0 if added_stone.color==BLACK else 1]:
                added_stone.remove()
                self.turn()
                return "illegal"
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            liberties = group.update_liberties()
            group_color = group.color
            group_stone_count = len(group.stones)
            if liberties == 0:
                group.remove()
                if group_color == BLACK:
                    self.black_catched += group_stone_count
                else:
                    self.white_catched += group_stone_count
        self.update_illegal()
    
    def clear(self):
        while self.groups:
            self.groups[0].remove()
        self.groups = []
        self.illegal = np.full((19, 19, 2), False)
        self.next = BLACK
        self.black_catched = 0
        self.white_catched = 0
       
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
    MAX_TEMPERATURE = 10.0
    TEMPERATURE = MAX_TEMPERATURE
    MIN_TEMPERATURE = 0.1
    TEMPERATURE_DECAY = (MIN_TEMPERATURE/TEMPERATURE) ** (EPOCHS/2)
    for epoch in range(EPOCHS*2):
        print("epoch", epoch)
        steps = 0
        pass_count = 0
        # only record as one side because white has "less steps" adventage
        trainas = (BLACK, WHITE)[epoch%2]
        while (steps < MAX_STEP):
            try_steps = 0
            while (try_steps < MAX_STEP - steps):
                old_map = board.map
                x, y, cor = model.decide(board, TEMPERATURE)
                if x==-1 and y==-1:
                    pass_count += 1
                    break
                else:
                    pass_count = 0
                try_steps += 1
                added_stone = Stone(board, (x, y), board.turn())
                # if is suicide
                if board.update_liberties(added_stone) != "illegal":
                    break
            winner = board.is_gameover(pass_count)
            if winner:
                print("winner:", winner)
                model.record((x, y), old_map, board.map, (B_WIN_REWARD if winner == BLACK else W_WIN_REWARD), True)
                break
            elif board.next != trainas:
                model.record((x, y), old_map, board.map, UNKNOWN_REWARD, False)
            steps += 1
            pygame.time.wait(int(((MAX_TEMPERATURE - TEMPERATURE)/MAX_TEMPERATURE)))
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
        if epoch>5 and epoch%2==1:
            model.save("model.h5")
            TEMPERATURE = max(MIN_TEMPERATURE, TEMPERATURE*TEMPERATURE_DECAY)

def test(ai_play_as):
    print("begin test")
    print("value:", model.get_value(board.map))
    while True:
        pygame.time.wait(250)
        if ai_play_as == board.next:
            old_board = board.map
            x, y, cor = model.decide(board, 0.01)
            print("model choose (%d, %d) for: %.4e"%(x, y, cor))
            if board.search(point=(x, y)) != []:
                continue
            added_stone = Stone(board, (x, y), board.turn())
            # if is suicide
            if board.update_liberties(added_stone) == "illegal":
                continue
            print("value:", model.get_value(old_board)[x+19*y])
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and board.outline.collidepoint(event.pos):
                        old_board = board.map
                        x = int(round(((event.pos[0] - 25) / 20.0), 0))
                        y = int(round(((event.pos[1] - 25) / 20.0), 0))
                        #print(x, y)
                        if board.search(point=(x, y)) != []:
                            continue
                        added_stone = Stone(board, (x, y), board.turn())
                        board.update_liberties(added_stone)
                    print("value:", model.get_value(old_board)[x+19*y]) 

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Go-Ai')
    screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
    background = pygame.image.load(BACKGROUND).convert()
    model = playmodel.ActorCritic(args.use_model)
    board = Board()
    if not TEST_ONLY:
        train()
    test(ai_play_as=(BLACK if args.test=="b" or args.test=="black" else WHITE))

