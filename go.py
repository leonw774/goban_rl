#!/usr/bin/env python
# coding: utf-8

import numpy as np

"""Go library
Edited by leow774 for keras ai training

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Stone(object):
    def __init__(self, board, point, color):
        """Create and initialize a stone.

        Arguments:
        board -- the board which the stone resides on
        point -- location of the stone as a tuple, e.g. (3, 3)
                 represents the upper left hoshi
        color -- color of the stone

        """
        self.board = board
        self.point = point
        self.color = color
        self.group = self.find_group()
    
    def remove(self):
        """Remove the stone from board."""
        self.board.map[self.point] = [0.0, 0.0]
        self.group.stones.remove(self)
        del self

    @property
    def neighbors(self):
        """Return a list of neighboring points."""
        neighborings = [(self.point[0] - 1, self.point[1]),
                       (self.point[0] + 1, self.point[1]),
                       (self.point[0], self.point[1] - 1),
                       (self.point[0], self.point[1] + 1)]
        neighborings = [n for n in neighborings if ((0<=n[0]<self.board.size) and (0<=n[1]<self.board.size))]
        return neighborings

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

    def __str__(self):
        """Return the location of the stone, e.g. 'D17'."""
        return 'ABCDEFGHJKLMNOPQRST'[self.point[0]] + str(self.board.size-(self.point[1]))


class Group(object):
    def __init__(self, board, stone):
        """Create and initialize a new group.

        Arguments:
        board -- the board which this group resides in
        stone -- the initial stone in the group

        """
        self.board = board
        self.board.groups.append(self)
        self.stones = [stone]
        self.color = stone.color
        self.liberties = None
        self.update_liberties()

    def merge(self, group):
        """Merge two groups.

        This method merges the argument group with this one by adding
        all its stones into this one. After that it removes the group
        from the board.

        Arguments:
        group -- the group to be merged with this one

        """
        for stone in group.stones:
            stone.group = self
            self.stones.append(stone)
        self.board.groups.remove(group)
        del group

    def remove(self):
        """Remove the entire group."""
        while self.stones:
            self.stones[0].remove()
        if self in self.board.groups:
            self.board.groups.remove(self)
        del self
    
    def update_liberties(self):
        """Update the group's liberties.
        As this method will NOT remove the entire group if no liberties can
        be found. The removal is now handled in Board.update_liberties

        """
        liberties = []
        for stone in self.stones:
            for liberty in stone.liberties:
                liberties.append(liberty)
        self.liberties = set(liberties)

    def __str__(self):
        """Return a list of the group's stones as a string."""
        return str([str(stone) for stone in self.stones])


class Board(object):
    def __init__(self, size):
        """Create and initialize an empty board."""
        self.groups = []
        self.size = size
        # map is a numpy array representation of the board
        # empty = (0, 0) black = (1, 0) white = (0, 1)
        self.map = np.zeros((self.size, self.size, 2)) 
        self.illegal = np.full((self.size, self.size, 2), False)
        self.b_catched = 0
        self.w_catched = 0
        self.next = BLACK
    
    def has_stone(self, point):
        return np.max(self.map[point]) == 1
    
    def update_illegal(self):
        empty_points = np.argwhere(np.max(self.map, axis=2)==0)
        next_color = 0 if self.next == BLACK else 1
        self.illegal[:, :, next_color] = False
        for e in empty_points:
            neighbors = [(e[0] - 1, e[1]),
                         (e[0] + 1, e[1]),
                         (e[0], e[1] - 1),
                         (e[0], e[1] + 1)]
            neighbors = [n for n in neighbors if ((0<=n[0]<self.size) and (0<=n[1]<self.size))]
            
            if all([self.has_stone(x) for x in neighbors]):
                neighbor_stones = self.search(points=neighbors)
                # suicide: made itself killed or made neighboring same color stone killed
                is_suicide = all([neighbor_stone.color != self.next for neighbor_stone in neighbor_stones])
                if not is_suicide:
                    for neighbor_stone in neighbor_stones:
                        if neighbor_stone.color == self.next:
                            if len(neighbor_stone.group.liberties) == 1:
                                is_suicide = True
                                break
                if is_suicide:
                    is_suicide_kill = False
                    for neighbor_stone in neighbor_stones:
                        if neighbor_stone.color != self.next:
                            if len(neighbor_stone.group.liberties) == 1:
                                is_suicide_kill = True
                                break
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
            group.update_liberties()
            if len(group.liberties) == 0:
                if group.color == BLACK:
                    self.b_catched += len(group.stones)
                else:
                    self.w_catched += len(group.stones)
                group.remove()
        self.update_illegal()
    
    def search(self, point=None, points=[]):
        """Search the board for a stone.

        The board is searched in a linear fashion, looking for either a
        stone in a single point (which the method will immediately
        return if found) or all stones within a group of points.

        Arguments:
        point -- a single point (tuple) to look for
        points -- a list of points to be searched

        """
        stones = []
        for group in self.groups:
            for stone in group.stones:
                if stone.point == point and not points:
                    return stone
                if stone.point in points:
                    stones.append(stone)
        return stones

    def turn(self):
        """Keep track of the turn by flipping between BLACK and WHITE."""
        if self.next == BLACK:
            self.next = WHITE
            return BLACK
        else:
            self.next = BLACK
            return WHITE
            
    def clear(self):
        while self.groups:
            self.groups[0].remove()
        self.groups = []
        self.illegal.fill(False)
        self.next = BLACK
        self.b_catched = 0
        self.w_catched = 0
