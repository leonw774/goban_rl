#!/usr/bin/env python
# coding: utf-8

import numpy as np

"""Go library
Edited by leow774 for keras ai training

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

BLACK = 0
WHITE = 1
COLOR = ((0, 0, 0), (255, 255, 255))

class Stone(object):
    def __init__(self, board, point):
        """Create and initialize a stone."""
        self.point = point
        self.color = board.next
        self.islegal = board.add_stone(point)
        
class Group:
    def __init__(self, stones_frozenset, neighbors_frozenset, color):
        self.id = np.random.randint(2147483647)
        # frozenset of tuple(x, y)
        self.stones = stones_frozenset
        self.neighbors = neighbors_frozenset
        self.color = color

    # def __hash__(self):
    #     return hash((self.id, self.color))

    # def __eq__(self, other):
    #     return self.id == other.id and self.stones == other.stones and self.color == other.color
    
    def __str__(self):
        return ("Group { id:" + str(self.id) +
                ", color:" + str(self.color) +
                ", stones:" + str([str(p) for p in self.stones]) +
                ", neibors:" + str([str(p) for p in self.neighbors]) +
                " }")

class Board(object):
    def __init__(self, size=19, komi=6.5, is_debug=False):
        """Create and initialize an empty board."""
        # grid is a numpy array representation of the board
        # empty = (0, 0) black = (True, 0) white = (0, 1)
        self.size = size
        self.komi = komi
        self.grid = np.zeros((self.size, self.size, 2))
        self.groups = set()

        self.neighors = {
            (x, y) : 
            frozenset(filter(self.bound, [(x-1,y), (x+1,y), (x,y-1), (x,y+1)])) 
            for x in range(size) for y in range(size)
        }
        self.all_points = frozenset([
            (i,j)
            for i in range(self.size)
            for j in range(self.size)
            if sum(self.grid[i,j]) == 0
        ])
        
        self.b_captured = 0
        self.w_captured = 0
        self.next = BLACK
        self.log = [] # (color, point, board_hash)

        self.same_state_illegal = set()
        self.suicide_illegal = set()

        self.zobrist_init = np.random.randint(9223372036854775807, dtype=np.int64) # 2^63 - 1
        # for grid positions
        self.zobrist_grid = np.random.randint(9223372036854775807, size=(self.size, self.size, 2), dtype=np.int64)
        # for b_captured, w_captured, next
        self.zobrist_next = np.random.randint(9223372036854775807, dtype=np.int64)
        self.outputDebug = is_debug

    def bound(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size 

    # Zobrist Hash
    def grid_hash(self):
        zhash = self.zobrist_init
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j, 0] == 1:
                    zhash ^= self.zobrist_grid[i, j, 0]
                else:
                    zhash ^= self.zobrist_grid[i, j, 1]
        if self.next == WHITE: zhash ^= self.zobrist_next
        return zhash

    def has_stone(self, point):
        return sum(self.grid[point]) == 1
        
    def update_grid(self, point, color=None):
        # asterick mean tuple unpack
        x, y = point
        if color == BLACK:
            self.grid[x, y, WHITE] = 0
            self.grid[x, y, BLACK] = 1
        elif color == WHITE:
            self.grid[x, y, WHITE] = 1
            self.grid[x, y, BLACK] = 0
        else:
            self.grid[x, y] = [0, 0]
    
    def handle_same_state_illegal(self, adding_point):
        """
        points in "same_state_illegal" should be removed after a successful add_stone() try  
        if adding_point is None: remove points  
        else: add point
        """
        if adding_point:
            self.same_state_illegal.add(adding_point)
        else:
            self.same_state_illegal = set()

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

        if new_point in self.same_state_illegal or new_point in self.suicide_illegal:
            return False

        # check suicide move
        if not all([self.has_stone(p) for p in new_group.neighbors]): # no suicide, no problem
            self.handle_same_state_illegal(None)
            self.handle_suicide_illegal(None)
            return True

        killed_nb_enemy_group = set()
        # only want to know if enemy ki == 0
        for nbeg in nb_enemy_groups:
            if all([self.has_stone(p) for p in nbeg.neighbors]):
                killed_nb_enemy_group.add(nbeg)
        
        if len(killed_nb_enemy_group) == 0:
            self.handle_suicide_illegal(new_point)
            if self.outputDebug:
                print("illegal: suicidal move")
            return False
        
        # temporary take off "dead" stones from grid to see if that would make same state
        for dead_nbeg in killed_nb_enemy_group:
            for s in dead_nbeg.stones:
                self.update_grid(s, None)
        # check latest previous 4 board state (preventing n-ko rotation, n <= 2)
        same_state_rewind_limit = 4
        same_state = False
        zhash = self.grid_hash()
        for entry in self.log[-same_state_rewind_limit:]:
            if entry[2] == zhash and entry[2] != -1: # -1 means it was passed: no same state check require
                same_state = True
                break
        # recover grid
        for dead_nbeg in killed_nb_enemy_group:
            for s in dead_nbeg.stones:
                self.update_grid(s, dead_nbeg.color)
        
        if same_state:
            self.handle_same_state_illegal(new_point)
            if self.outputDebug:
                print("illegal: same board state")
            return False
        # suicide & kill enemy & no same state
        self.handle_same_state_illegal(None)
        self.handle_suicide_illegal(None)
        return True
    # end def check_legal
    
    def update_capture(self, nb_enemy_groups):
        # remove dead groups
        # only look around new point
        dead_groups = set()
        for g in nb_enemy_groups:
            if all([self.has_stone(p) for p in g.neighbors]):
                dead_groups.add(g)
        
        for dg in dead_groups:
            if dg.color == WHITE:
                self.w_captured += len(dg.stones)
            else:
                self.b_captured += len(dg.stones)
            for s in dg.stones:
                self.update_grid(s, None)
        self.groups = self.groups - dead_groups
    
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
    
    def add_stone(self, point):
        """
        2 possibility:
          - link up with one or more group
          - standalone
        return true if successfully add a stone in a legal place  
               false if not success  
        """
        if self.has_stone(point):
            return False
        self.update_grid(point, self.next)

        # record neighbor groups
        nb_friend_groups = set()
        nb_enemy_groups = set()
        for nbp in self.neighors[point]:
            if not self.has_stone(nbp): continue
            for g in self.groups:
                if nbp in g.stones:
                    if g.color == self.next:
                        nb_friend_groups.add(g)
                    else:
                        nb_enemy_groups.add(g)
                    break

        # create a temporary stone group from the adding stone
        # because we didn't check legality yet
        new_group_stones = set([point])
        new_group_neighbors = set(self.neighors[point])
        # merge neighbor groups into self temp group
        for nbfg in nb_friend_groups:
            new_group_stones.update(nbfg.stones)
            new_group_neighbors.update(nbfg.neighbors)
        new_group_neighbors.discard(point)

        new_group = Group(frozenset(new_group_stones),
                        frozenset(new_group_neighbors),
                        self.next)
        self.groups.add(new_group)
        
        islegal = self.check_legal(point, new_group, nb_enemy_groups)
        if (not islegal):
            # delete temp adding group
            self.groups.remove(new_group)
            self.update_grid(point, None)
            return False

        self.groups = self.groups - nb_friend_groups # set difference
        self.update_capture(nb_enemy_groups)
        self.log.append((self.next, point, self.grid_hash()))
        self.turn()
        if self.outputDebug: self.debugPrint()
        return True

    def log_endgame(self, winner, reason):
        self.log.append((winner, reason, -1))

    def clear(self):
        self.groups = set()
        self.grid = np.zeros((self.size, self.size, 2))
        self.next = BLACK
        self.log = []
        self.b_captured = 0
        self.w_captured = 0
        self.same_state_illegal = set()
        self.suicide_illegal = set()
    
    def debugPrint(self):
        print("Board State:")
        for g in self.groups:
            print(g)
        output_str = ""
        for i in range(self.size):
            for j in range(self.size):
                p = (j, i)
                if self.grid[p][WHITE]:
                    output_str += "O "
                elif self.grid[p][BLACK]:
                    output_str += "X "
                else:
                    output_str += "  "
            output_str += "\n"
        print("recorded illegal points:", )
        print("same board illegal:", self.same_state_illegal)
        print("suicide illegal:", self.suicide_illegal)
        print("latset stone added", self.log[-1])

    def score(self, output=False):
        """
        1. check life & dead of every group
        2. count territory controlled by every living group
        3. Tromp-Tayler rule's scoring:
           score = stones on the board + territory + captured + komi(6.5) if white
        - Territory:
            - if a empty point P can reach a stone S it means that
              there exist a path from P to S consisted of other empty points or
              P and S is adjacent
            - if all the stones that P can reach are stones of color C, then P is
              a territory of C
        """
        # for all open points determine whose territory it belongs to:
        b_territory = set()
        w_territory = set()
        neutral_points = set()
        #territory_influence_distance = 2
        all_points = set(self.all_points)
        while len(all_points) > 0:
            #print("len(all_points)",len(all_points))
            p = all_points.pop()
            # if p in b_territory or p in w_territory or p in neutral_points:
            #     continue
            #print("examing",p)
            NEUTRAL = -1
            reach_color = None
            # init search 
            searched = set()
            expander = set()
            expander.add(p)
            # do BFS 
            while len(expander) > 0 and reach_color != NEUTRAL:
                search_p = expander.pop()
                #print("searched", search_p)
                if self.grid[search_p][BLACK]: # is black stone
                    if reach_color == None:
                        reach_color = BLACK
                    elif reach_color == WHITE:
                        reach_color = NEUTRAL
                elif self.grid[search_p][WHITE]: # is white stone
                    if reach_color == None:
                        reach_color = WHITE
                    elif reach_color == BLACK:
                        reach_color = NEUTRAL
                else: # is empty
                    searched.add(search_p)
                    if search_p in w_territory:
                        reach_color = WHITE
                        break
                    elif search_p in b_territory:
                        reach_color = BLACK
                        break
                    elif search_p in neutral_points:
                        reach_color = NEUTRAL
                        break
                    else:
                        for nbp in self.neighors[search_p]:
                            if nbp not in searched:
                                expander.add(nbp)
            searched.update(expander)
            # end BFS
            if reach_color == BLACK:
                b_territory.update(searched)
            elif reach_color == WHITE:
                w_territory.update(searched)
            else:
                neutral_points.update(searched)
            #print(p, "is", reach_color)
            all_points = all_points - searched
        # end double for: every empty points
        
        # score = living friendly stones + territory + captured + dead enemy stones + komi(6.5) if white
        # w_score = len(w_living_stones) + len(w_territory) + len(b_dead_stones) + self.b_captured + self.komi
        # b_score = len(b_living_stones) + len(b_territory) + len(w_dead_stones) + self.w_captured

        # score = stones on board + territory + captured + komi(6.5) if white
        w_score = np.sum(self.grid[:,:,WHITE]==1) + len(w_territory) + self.b_captured + self.komi    
        b_score = np.sum(self.grid[:,:,BLACK]==1) + len(b_territory) + self.w_captured
        
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
            return winner, b_score, w_score, output_str
        else:
            return winner, b_score, w_score
    # end def cal_score
# end class Board