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
    def __init__(self, stones, color):
        self.id = np.random.randint(2147483647)
        self.stones = set(stones) # stones = tuple(x, y)
        self.color = color

    def is_neighbor(self, request_point):
        for p in self.stones:
            if (abs(p[0] - request_point[0]) + abs(p[1] - request_point[1]) == 1):
                return True
        return False

    def __hash__(self):
        return hash((self.id, self.color))

    # def __eq__(self, other):
    #     return self.id == other.id and self.stones == other.stones and self.color == other.color
    
    def __str__(self):
        return "id " + str(self.id) + ", color " + str(self.color) + ", coord " + str([str(p) for p in self.stones])

def nbpoints(point, limit):
    nblist = []
    if point[1] - 1 >= 0:
        nblist.append((point[0], point[1] - 1))
    if point[0] - 1 >= 0:
        nblist.append((point[0] - 1, point[1]))
    if point[1] + 1 < limit:
        nblist.append((point[0], point[1] + 1))
    if point[0] + 1 < limit:
        nblist.append((point[0] + 1, point[1]))
    return nblist
    
# end def nbpoints

class Board(object):
    def __init__(self, size, komi, is_debug=False):
        """Create and initialize an empty board."""
        # grid is a numpy array representation of the board
        # empty = (0, 0) black = (True, 0) white = (0, 1)
        self.size = size
        self.komi = komi
        self.grid = np.zeros((self.size, self.size, 2))
        self.groups = set()
        
        self.b_captured = 0
        self.w_captured = 0
        self.next = BLACK
        self.log = [] # (color, point, board_hash)

        self.same_state_illegal = set()
        self.suicide_illegal = set()

        self.zobrist_init = np.random.randint(9223372036854775807, dtype=np.int64) # 2^63 - 1
        self.zobrist_table = np.random.randint(9223372036854775807, size=(self.size, self.size, 2), dtype=np.int64)
        self.outputDebug = is_debug

    def zobrist_hash(self):
        zhash = self.zobrist_init
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j, 0] == 1:
                    zhash ^= self.zobrist_table[i, j, 0]
                elif self.grid[i, j, 1] == 1:
                    zhash ^= self.zobrist_table[i, j, 1]
        return zhash

    def has_stone(self, point):
        return max(self.grid[point]) == 1
        
    def update_grid(self, point, color=None):
        if color == BLACK:
            self.grid[point[0], point[1], BLACK] = 1
        elif color == WHITE:
            self.grid[point[0], point[1], WHITE] = 1
        else:
            self.grid[point] = [0, 0]
    
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
        points in "suicide_illegal" should be removed if a stone is on that point
        """
        self.suicide_illegal = set([p for p in self.suicide_illegal if max(self.grid[p]) == 0])
        if adding_point:
            self.suicide_illegal.add(adding_point)

    def check_legal(self, new_point, new_group, nb_enemy_groups):
        """
        return (new_group is killed IMPLY kill other) AND (NOT same board state)
        """
        # check suiside move
        killed_nb_enemy_group = set()
        # only want to know if enemy ki == 0
        for nbeg in nb_enemy_groups:
            is_alive = False
            for s in nbeg.stones:
                for nbp in nbpoints(s, self.size):
                    if max(self.grid[nbp]) == 0: # has no stone
                        is_alive = True
                        break
                if is_alive: break
            if not is_alive:
                killed_nb_enemy_group.add(nbeg)
        
        is_suicide = True
        for p in new_group.stones:
            for nbp in nbpoints(p, self.size):
                if not self.has_stone(nbp):
                    is_suicide = False
                    break
            if not is_suicide: break

        if is_suicide and len(killed_nb_enemy_group) == 0:
            self.handle_suicide_illegal(new_point)
            if self.outputDebug:
                print("illegal: suisidal move")
            return False
        
        # temporary take off "dead" stones from grid to see if that would make same state
        for dead_nbeg in killed_nb_enemy_group:
            for s in dead_nbeg.stones:
                self.update_grid(s, None)
        # check latest previous 6 board state (preventing n-ko rotation, n <= 3)
        same_state_rewind_limit = 6
        same_state_rewind_num = 0
        same_state = False
        zhash = self.zobrist_hash()
        for entry in reversed(self.log):
            if entry[2] == zhash and entry[2] != -1: # -1 means it was passed: no same state check require
                same_state = True
                break
            same_state_rewind_num += 1
            if same_state_rewind_num >= same_state_rewind_limit: break
        # recover grid
        for dead_nbeg in killed_nb_enemy_group:
            for s in dead_nbeg.stones:
                self.update_grid(s, dead_nbeg.color)
        
        if same_state:
            self.handle_same_state_illegal(new_point)
            if self.outputDebug:
                print("illegal: same board state")
            return False
        
        self.handle_same_state_illegal(None)
        self.handle_suicide_illegal(None)
        return True
    # end def check_legal
    
    def update_capture(self, new_group):
        # remove dead groups
        dead_groups = set()
        for g in self.groups:
            is_alive = False
            for p in g.stones:
                for nbp in nbpoints(p, self.size):
                    if max(self.grid[nbp]) == 0: # has no stone
                        is_alive = True
                        break
                if is_alive:
                    break
            if (not is_alive) and (g != new_group):
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

        # record nbGroupsId
        nb_friend_groups = set()
        nb_enemy_groups = set()
        for g in self.groups:
            if g.is_neighbor(point):
                if g.color == self.next:
                    nb_friend_groups.add(g)
                else:
                    nb_enemy_groups.add(g)

        # create a temporary stone group from the adding stone
        # because we didn't check legality yet
        new_group_stones = set([point])
        # merge neighbor groups into self temp group
        for nbfg in nb_friend_groups:
            new_group_stones = new_group_stones.union(nbfg.stones)
        new_group = Group(new_group_stones, self.next)
        self.groups.add(new_group)
        
        legal = self.check_legal(point, new_group, nb_enemy_groups)
        if (not legal):
            # delete temp adding group
            self.groups.remove(new_group)
            self.update_grid(point, None)
            return False

        self.groups = self.groups - nb_friend_groups # set difference
        self.update_capture(new_group)
        self.log.append((self.next, point, self.zobrist_hash()))
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
    
    def debugPrint(self):
        print("all groups:")
        for g in self.groups:
            print(g)
        print("recorded illegal points:", )
        print("same board illegal:", self.same_state_illegal)
        print("suicide illegal:", self.suicide_illegal)
        print("latset stone added", self.log[-1])

    def score(self, output=False):
        """
        1. check life & dead of every group
        2. count territory controlled by every living group
        3. Chinese scoring rule: score = living friendly stones + territory + captured + dead enemy stones + komi(6.5) if white
        - Life & Dead:
            - inner liberty: an inner liberty of a group is a liberty enclosed by:
                - same color group's stone
                - edges of the board
            - outer liberty: otherwise, a liberty of a group is an outer liberty
            - if a group has more than two inner liberty, this group is alive
        - Territory:
            - a color's territorys is enclosed by at most 4 line far away of:
                - all same color's living stones
                - edges of the board
        """
        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        b_living_stones = set()
        w_living_stones = set()
        b_dead_stones = set()
        w_dead_stones = set()
        public_outlibs = set()
        for i, g in enumerate(self.groups):
            inlib = set()
            for s in g.stones:
                for nbp in nbpoints(s, self.size):
                    if nbp in public_outlibs: continue
                    if (not self.has_stone(nbp)): # is liberty
                        is_inner = True
                        for n in range(4): # 4-direction
                            encsearch = (nbp[0] + directions[n][0], nbp[1] + directions[n][1])
                            while (0 <= encsearch[0] < self.size) and (0 <= encsearch[1] < self.size):
                                if (self.grid[encsearch][(g.color+1)%2] == 1): # find enemy stone
                                    #print(encsearch, "is enemy color,", nbp, "is outer lib")
                                    is_inner = False
                                    break
                                if (self.grid[encsearch][g.color] == 1): # find friendly stone
                                    break
                                encsearch = (encsearch[0] + directions[n][0], encsearch[1] + directions[n][1])
                            if not is_inner:
                                public_outlibs.add(nbp)
                                break
                        # end for: 4-direction enclosure search
                        if is_inner:
                            inlib.add(nbp)
                            #print(nbp, "is inner lib")
                    # end if is liberty
                # end for nbpoints
            # end for group points
            if len(inlib) >= 2:
                if g.color == BLACK:
                    b_living_stones = b_living_stones.union(g.stones)
                else:
                    w_living_stones = w_living_stones.union(g.stones)
            else:
                if g.color == BLACK:
                    b_dead_stones = b_dead_stones.union(g.stones)
                else:
                    w_dead_stones = w_dead_stones.union(g.stones)
        # end for: find living & dead groups 
        
        # for all open points determine whose territory it belongs to:
        b_territory = set()
        w_territory = set()
        for i in range(self.size):
            for j in range(self.size):
                if (max(self.grid[i, j]) == 0): # has no stone
                    p = (i, j)
                    belonging = None
                    NEUTRAL = 2
                    for n in range(4): # 4-direction
                        encsearch = (p[0] + directions[n][0], p[1] + directions[n][1])
                        distance = 0 # limit is 4
                        while (0 <= encsearch[0] < self.size) and (0 <= encsearch[1] < self.size):
                            # if encsearch in b_territory:
                            #     belonging = BLACK
                            #     break
                            if encsearch in b_living_stones:
                                if belonging == BLACK or belonging == None:
                                    belonging = BLACK
                                else:
                                    belonging = NEUTRAL
                                break
                            # if encsearch in w_territory:
                            #     belonging = WHITE
                            #     break
                            if encsearch in w_living_stones:
                                if belonging == WHITE or belonging == None:
                                    belonging = WHITE
                                else:
                                    belonging = NEUTRAL
                                break
                            encsearch = (encsearch[0] + directions[n][0], encsearch[1] + directions[n][1])
                            distance = distance + 1
                            if distance >= 4:
                                break
                        # end while: enclosure search
                        if belonging == NEUTRAL:
                            break
                    # end for: 4-direction enclosure search
                    if belonging == BLACK:
                        b_territory.add(p)
                    elif belonging == WHITE:
                        w_territory.add(p)
        # end double for: every open points
        
        # score = living friendly stones + territory + captured + dead enemy stones + komi(6.5) if white
        # w_score = len(w_living_stones) + len(w_territory) + len(b_dead_stones) + self.b_captured + self.komi
        # b_score = len(b_living_stones) + len(b_territory) + len(w_dead_stones) + self.w_captured

        # score = stones on board + territory + captured + komi(6.5) if white
        w_score = np.sum(self.grid[:,:,WHITE]==1) + len(w_territory) + + self.b_captured + self.komi    
        b_score = np.sum(self.grid[:,:,BLACK]==1) + len(b_territory) + + self.w_captured

        output_str = ""
        # draw territory result on as string
        for i in range(self.size):
            for j in range(self.size):
                p = (j, i)
                if p in w_territory:
                    output_str += "c "
                elif p in b_territory:
                    output_str += "v "
                elif p in w_living_stones:
                    output_str += "O "
                elif p in w_dead_stones:
                    output_str += "O " 
                elif p in b_living_stones:
                    output_str += "X "
                elif p in b_dead_stones:
                    output_str += "X "
                else:
                    output_str += "  "
            output_str += "\n"
        output_str += "B:" + str(b_score) + " W:" + str(w_score) + (" B" if b_score > w_score else " W") + " win"

        if output:
            print(output_str)
        
        if b_score > w_score:
            winner = BLACK
        else:
            winner = WHITE
        return winner, b_score, w_score, output_str
    # end def cal_score
# end class Board