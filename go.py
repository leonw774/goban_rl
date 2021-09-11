#!/usr/bin/env python
# coding: utf-8
import numpy as np
from ctypes import c_double, c_int, POINTER, CDLL
from sys import getsizeof

"""Go library

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

Edited by leow774
"""

c_board_eval = CDLL("./board_evals.dll").boardEval

BLACK = 0
WHITE = 1
COLOR = ((0, 0, 0), (255, 255, 255))

class Stone(object):
    __slots__ = ("point", "color", "islegal")
    def __init__(self, board, point):
        """Create and initialize a stone."""
        self.point = point
        self.color = board.next
        self.islegal = board.add_stone(point)
        
class Group:
    __slots__ = ("stones", "color", "neighbors")
    def __init__(self, stones_frozenset, neighbors_frozenset, color):
        #self.id = np.random.randint(2147483647)
        # frozenset of tuple(x, y)
        self.stones = stones_frozenset
        self.neighbors = neighbors_frozenset
        self.color = color
    
    def __str__(self):
        return ("Group { color:" + str(self.color) +
                ", member:" + str([str(p) for p in self.stones]) +
                ", neighbor:" + str([str(p) for p in self.neighbors]) +
                " }")

def board_from_state(state):
    return Board(state = state)

class Board(object):
    __slots__ = ("size", "komi", "grid", "groups", "point_to_group", "next", 
                "neighbors", "all_points", "log", "same_position_illegal", "suicide_illegal", "outputDebug")
        
    def __init__(self, size=19, komi=6.5, state=None, is_debug=False):
        """Create and initialize an empty board."""
        # grid is a numpy array representation of the board
        # empty = (0.0, 0.0) black = (1.0, 0) white = (0.0, 1.0)
        if state is not None:
            self.from_state(state)
        else:
            self.size = size
            self.komi = komi
            self.neighbors = {
                (x, y) : 
                frozenset(filter(self.bound, [(x-1,y), (x+1,y), (x,y-1), (x,y+1)])) 
                for x in range(self.size) for y in range(self.size)
            }
            self.all_points = frozenset([
                (i, j)
                for i in range(self.size)
                for j in range(self.size)
            ])
            self.point_to_group = {p : None for p in self.all_points}
            self.clear()

        self.outputDebug = is_debug

    def debugPrint(self):
        print("Board Position:")
        for g in self.groups:
            print(g)
        
        valid_grid = self.get_valid_move_mask()
        grid_str = ""
        valid_str = ""
        for i in range(self.size):
            for j in range(self.size):
                p = (j, i)
                if self.grid[p][WHITE]:
                    grid_str += "O "
                elif self.grid[p][BLACK]:
                    grid_str += "X "
                else:
                    grid_str += "  "
                
                if valid_grid[j+i*self.size]:
                    valid_str += "V "
                else:
                    valid_str += "I "
            grid_str += "\n"
            valid_str += "\n"
        print("next", self.next)
        print("same board illegal:", self.same_position_illegal)
        print("suicide illegal:", self.suicide_illegal)
        print("latset stone added", self.log[-1])
        print(grid_str)
        print(valid_str)

    def clear(self):
        self.groups = set()
        self.grid = np.zeros((self.size, self.size, 2), dtype = int)
        self.point_to_group = {p : None for p in self.all_points}
        self.next = BLACK
        self.log = []
        # self.b_captured = 0
        # self.w_captured = 0
        self.same_position_illegal = set()
        self.suicide_illegal = set()

    @property
    def state(self):
        return [self.size, self.komi,
            # these has to copy
            self.grid.tobytes(), self.groups.copy(), self.point_to_group.copy(), 
            # next is int so no copy, neighbors and all_points is constant, log has to copy
            self.next, self.neighbors, self.all_points, self.log.copy()]

    def from_state(self, state):
        self.size = state[0]
        self.komi = state[1]
        self.grid = np.fromstring(state[2])
        self.grid.resize((self.size, self.size, 2))
        self.groups = state[3]
        self.point_to_group = state[4]
        self.next = state[5]
        self.neighbors = state[6]
        self.all_points = state[7]
        self.log = state[8]

    def bound(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size 

    def grid_hash(self):
        trinary_grid = np.zeros((self.size, self.size), dtype=int)
        trinary_grid[self.grid[:,:,BLACK]] = 1
        trinary_grid[self.grid[:,:,WHITE]] = 2
        trinary_grid_flat = trinary_grid.ravel()
        ghash = int(''.join(["1" if i == 1 else "2" if i else "0" for i in trinary_grid_flat]), 3)
        return ghash
        
    def update_grid(self, target, color=None):
        # asterick mean tuple unpack
        if type(target) == tuple:
            x, y = target
            if color == BLACK or color == WHITE:
                self.grid[x, y, color] = 1
            else:
                self.grid[x, y] = [False, False]
        elif hasattr(target, '__iter__'): # if it is  iterable
            for p in target:
                x, y = p
                if color == BLACK or color == WHITE:
                    self.grid[x, y, color] = 1
                else:
                    self.grid[x, y] = [False, False]

    def update_point_to_group(self, points, group):
        # to link points to group, group can be None
        for p in points:
            self.point_to_group[p] = group
    
    def get_valid_move_mask(self):
        # index 0 ~ self.size ** 2 - 1 are points
        # index self.size ** 2 is pass
        valid_mask = np.zeros((self.size ** 2 + 1), dtype=bool)
        # transpose so that [x, y] become [x + y * size]
        # used '~' operator to invert
        valid_mask[:self.size ** 2] = ~self.grid.any(axis=2).T.flatten()
        for p in self.suicide_illegal.union(self.same_position_illegal):
            valid_mask[p[0] + p[1] * self.size] = False
        if not np.all(valid_mask): # not empty
            valid_mask[self.size ** 2] = True # can pass
        return valid_mask

    def handle_same_position_illegal(self, adding_point):
        """
        points in "same_position_illegal" should be removed after a successful add_stone() try  
        if adding_point is None: remove points  
        else: add point
        """
        if adding_point:
            self.same_position_illegal.add(adding_point)
        else:
            self.same_position_illegal = set()

    def handle_suicide_illegal(self, adding_point):
        """
        points in "suicide_illegal" should be removed after a successful add_stone() try
        """
        if adding_point:
            self.suicide_illegal.add(adding_point)
        else:
            self.suicide_illegal = set()

    def check_legal(self, new_point, new_group, nb_enemy_groups):
        """
        return (new_group is killed IMPLY kill other) AND (NOT same board state)
        """

        if new_point in self.same_position_illegal or new_point in self.suicide_illegal:
            return False

        # check suicide move
        if not all([any(self.grid[p]) for p in new_group.neighbors]): # no suicide, no problem
            self.handle_same_position_illegal(None)
            self.handle_suicide_illegal(None)
            return True

        killed_nb_enemy_group = set()
        # only want to know if enemy ki == 0
        for nbeg in nb_enemy_groups:
            if all([any(self.grid[p]) for p in nbeg.neighbors]):
                killed_nb_enemy_group.add(nbeg)
        
        if len(killed_nb_enemy_group) == 0:
            self.handle_suicide_illegal(new_point)
            # if self.outputDebug:
            #     print("illegal: suicidal move")
            return False
        
        # if more than 1 stone killed then it won't be same position
        if len(killed_nb_enemy_group) > 1:
            return True
        else:
            (the_dead_nbeg,) = killed_nb_enemy_group # use tuple unpacking to extract the only element
            if len(the_dead_nbeg.stones) > 1:
                return True

        # temporary take off "dead" stones from grid to see if that would make same position
        for dead_nbeg in killed_nb_enemy_group:
            self.update_grid(dead_nbeg.stones, None)
        # check latest previous 6 board state (preventing n-ko rotation, n <= 3)
        same_position_rewind_limit = 6
        same_position = False
        ghash = self.grid_hash()
        for entry in self.log[-same_position_rewind_limit:]:
            if entry[2] == ghash and entry[2] != -1: # -1 means it was passed: no same position check require
                same_position = True
                break
        # recover grid
        for dead_nbeg in killed_nb_enemy_group:
            self.update_grid(dead_nbeg.stones, dead_nbeg.color)
        
        if same_position:
            self.handle_same_position_illegal(new_point)
            # if self.outputDebug:
            #     print("illegal: same board state")
            return False
        # suicide & kill enemy & no same position
        self.handle_same_position_illegal(None)
        self.handle_suicide_illegal(None)
        return True
    # end def check_legal
    
    def update_capture(self, nb_enemy_groups):
        # remove dead groups
        # only look around new point
        dead_groups = set()
        for g in nb_enemy_groups:
            if all([any(self.grid[p]) for p in g.neighbors]):
                dead_groups.add(g)
        
        for dg in dead_groups:
            # if dg.color == WHITE:
            #     self.w_captured += len(dg.stones)
            # else:
            #     self.b_captured += len(dg.stones)
            self.update_point_to_group(dg.stones, None)
            self.update_grid(dg.stones, None)
        self.groups = self.groups - dead_groups
    
    def add_stone(self, point):
        """
        2 possibility:
          - link up with one or more group
          - standalone
        return true if successfully add a stone in a legal place  
               false if not success  
        """
        if any(self.grid[point]): # has stone
            return False
        self.update_grid(point, self.next)

        # record neighbor groups
        nb_friend_groups = set()
        nb_enemy_groups = set()
        for nbp in self.neighbors[point]:
            if any(self.grid[point]): # has stone
                nbg = self.point_to_group[nbp]
                if nbg:
                    if nbg.color == self.next:
                        nb_friend_groups.add(nbg)
                    else:
                        nb_enemy_groups.add(nbg)

        # create a temporary stone group from the adding stone
        # because we didn't check legality yet
        new_group_stones = set([point])
        new_group_neighbors = set(self.neighbors[point])
        # merge neighbor groups into self temp group
        for nbfg in nb_friend_groups:
            new_group_stones.update(nbfg.stones)
            new_group_neighbors.update(nbfg.neighbors)
        new_group_neighbors.discard(point)

        new_group = Group(frozenset(new_group_stones),
                        frozenset(new_group_neighbors),
                        self.next)
        self.groups.add(new_group)
        
        if (not self.check_legal(point, new_group, nb_enemy_groups)):
            # delete temp adding group
            self.groups.remove(new_group)
            self.update_grid(point, None)
            return False

        self.update_point_to_group(new_group_stones, new_group)
        self.groups = self.groups - nb_friend_groups # set difference
        self.update_capture(nb_enemy_groups)
        self.log.append((self.next, point, self.grid_hash()))
        self.turn()
        # if self.outputDebug: self.debugPrint()
        return True

    def pass_move(self):
        self.log.append((self.next, "pass", -1))
        self.turn()

    def turn(self):
        """Keep track of the turn by flipping between BLACK and WHITE."""
        if self.next == BLACK:
            self.next = WHITE
            return BLACK
        else:
            self.next = BLACK
            return WHITE

    def log_endgame(self, winner, reason):
        self.log.append((winner, reason, -1))
    
    def write_log_file(self, file):
        for line in self.log:
            file.write(("B" if line[0] == BLACK else "W") + " " + str(line[1]) + "\n")
        grid_string = ""
        for i in range(self.size):
            for j in range(self.size):
                p = (j, i)
                if self.grid[p][WHITE]:
                    grid_string += "O "
                elif self.grid[p][BLACK]:
                    grid_string += "X "
                else:
                    grid_string += "  "
            grid_string += "\n"
        grid_string += "\n"
        file.write(grid_string)

    def score(self, output=False):
        """
        1. check life & dead of every group
        2. count territory controlled by every living group
        3. Tromp-Tayler rule's scoring:
           score = stones on the board + territory + komi(6.5) if white
        - Territory:
            - if a empty point P can reach a stone S, it means that
              there exist a path from P to S consisted of other empty points or
              P and S is adjacent
            - if all the stones that P can reach are stones of color C, then P is
              a territory of C
        """
        # for all open points determine whose territory it belongs to:
        b_territory = set()
        w_territory = set()
        neutral_points = set()
        all_points = set(self.all_points) # make a copy
        while len(all_points) > 0:
            p = all_points.pop()
            # if p in b_territory or p in w_territory or p in neutral_points:
            #     continue
            #print("examing",p)
            NEUTRAL = -1
            reached_color = None
            # init search 
            searched = set()
            expander = set()
            expander.add(p)
            # do BFS 
            while len(expander) > 0 and reached_color != NEUTRAL:
                p = expander.pop()
                #print("searched", p)
                if self.grid[p][BLACK]: # is black stone
                    if reached_color == None:
                        reached_color = BLACK
                    elif reached_color == WHITE:
                        reached_color = NEUTRAL
                elif self.grid[p][WHITE]: # is white stone
                    if reached_color == None:
                        reached_color = WHITE
                    elif reached_color == BLACK:
                        reached_color = NEUTRAL
                else: # is empty
                    searched.add(p)
                    if p in w_territory:
                        reached_color = WHITE
                        break
                    elif p in b_territory:
                        reached_color = BLACK
                        break
                    elif p in neutral_points:
                        reached_color = NEUTRAL
                        break
                    else:
                        for nbp in self.neighbors[p]:
                            if nbp not in searched:
                                expander.add(nbp)
            searched.update(expander)
            # end BFS
            if reached_color == BLACK:
                b_territory.update(searched)
            elif reached_color == WHITE:
                w_territory.update(searched)
            else:
                neutral_points.update(searched)
            #print(p, "is", reached_color)
            all_points = all_points - searched
        # end double for: every empty points
        
        # Japanese rule
        # score = living friendly stones + territory + captured + komi if white
        # w_score = len(w_territory) + len(b_dead_stones) + self.b_captured + self.komi
        # b_score = len(b_territory) + len(w_dead_stones) + self.w_captured

        # Chinese rule (not really)
        # score = stones on board + territory + komi if white
        w_score = np.sum(self.grid[:,:,WHITE]) + len(w_territory) + self.komi    
        b_score = np.sum(self.grid[:,:,BLACK]) + len(b_territory)
        score_diff = b_score - w_score
        if b_score > w_score:
            winner = BLACK
        else:
            winner = WHITE

        if output:
            output_str = ""
            # draw territory result on as string
            for i in range(self.size):
                for j in range(self.size):
                    p = (j, i)
                    if p in w_territory:
                        output_str += "c "
                    elif p in b_territory:
                        output_str += "v "
                    elif self.grid[p][WHITE]:
                        output_str += "O "
                    elif self.grid[p][BLACK]:
                        output_str += "X "
                    else:
                        output_str += "  "
                output_str += "\n"
            output_str += "B:" + str(b_score) + " W:" + str(w_score) + (" B" if b_score > w_score else " W") + " win"
            return winner, score_diff, output_str
        else:
            return winner, score_diff
    # end def score

    """
    double* c_board_eval(int size, int* init_grid, int d)
        PARAMETERS:
        init_grid is original board grid with black = 1, empty = 0, white = -1
        d is the parameter for territory estimation

        the final territory count is based on
        (estimated territory) OR (territory based on only unconditionally alive groups)

        RETURN:
        float: evaluation value = black score - white score
    """
    """
        This fuinction do mid-game score evaluation to help reduce bad moves in monte-carlo
    """
    def eval(self):
        d = 5 # dilation parameter
        size_square = int(self.size * self.size)
        c_board_eval.restype = c_double

        init_grid = np.zeros((self.size, self.size), dtype=int)
        init_grid[self.grid[:,:,BLACK]] = 1
        init_grid[self.grid[:,:,WHITE]] = -1
        grid_list = init_grid.ravel().tolist()
        init_arr = (c_int * size_square) (*grid_list)
        result_eval = c_board_eval(c_int(self.size), init_arr, c_int(d))
        return result_eval
    # end def eval
# end class Board
