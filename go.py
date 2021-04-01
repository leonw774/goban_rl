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

# int64_high = 2**63 - 1
# int64_low = -(2**63)
# ZOBRIST_INIT = np.random.randint(int64_low, int64_high, dtype=np.int64)
# # for grid positions
# ZOBRIST_GRID = np.random.randint(int64_low, int64_high, size=(19, 19, 3), dtype=np.int64)
# # for b_captured, w_captured, next
# ZOBRIST_NEXT = np.random.randint(int64_low, int64_high, dtype=np.int64)

def board_from_state(state):
    return Board(state = state)

class Board(object):
    __slots__ = ("size", "komi", "grid", "groups", "point_to_group", "next", 
                "neighbors", "all_points", "log", "same_state_illegal", "suicide_illegal", "outputDebug")
        
    def __init__(self, size=19, komi=6.5, state=None, is_debug=False):
        """Create and initialize an empty board."""
        # grid is a numpy array representation of the board
        # empty = (0.0, 0.0) black = (1.0, 0) white = (0.0, 1.0)
        if state is not None:
            self.from_state(state)
        else:
            self.size = size
            self.komi = komi
            self.grid = np.zeros((self.size, self.size, 2), dtype=bool)
            self.groups = set()
            self.next = BLACK
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
            self.log = []
        
        # self.b_captured = 0
        # self.w_captured = 0
        self.same_state_illegal = set()
        self.suicide_illegal = set()
        self.outputDebug = is_debug

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
        self.grid = np.fromstring(state[2], dtype=bool)
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
            if color == BLACK:
                self.grid[x, y, color] = True
            elif color == WHITE:
                self.grid[x, y, color] = True
            else:
                self.grid[x, y] = [False, False]
        else: # assume it is a iteratable
            for p in target:
                x, y = p
                if color == BLACK:
                    self.grid[x, y, color] = True
                elif color == WHITE:
                    self.grid[x, y, color] = True
                else:
                    self.grid[x, y] = [False, False]

    def update_point_to_group(self, points, group):
        # to link points to group, group can be None
        for p in points:
            self.point_to_group[p] = group
    
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
        if not all([any(self.grid[p]) for p in new_group.neighbors]): # no suicide, no problem
            self.handle_same_state_illegal(None)
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
        
        # if more than 1 stone killed then it won't be same state
        if len(killed_nb_enemy_group) > 1:
            return True
        else:
            (the_dead_nbeg,) = killed_nb_enemy_group # use tuple unpacking to extract the only element
            if len(the_dead_nbeg.stones) > 1:
                return True

        # temporary take off "dead" stones from grid to see if that would make same state
        for dead_nbeg in killed_nb_enemy_group:
            self.update_grid(dead_nbeg.stones, None)
        # check latest previous 6 board state (preventing n-ko rotation, n <= 3)
        same_state_rewind_limit = 6
        same_state = False
        ghash = self.grid_hash()
        for entry in self.log[-same_state_rewind_limit:]:
            if entry[2] == ghash and entry[2] != -1: # -1 means it was passed: no same state check require
                same_state = True
                break
        # recover grid
        for dead_nbeg in killed_nb_enemy_group:
            self.update_grid(dead_nbeg.stones, dead_nbeg.color)
        
        if same_state:
            self.handle_same_state_illegal(new_point)
            # if self.outputDebug:
            #     print("illegal: same board state")
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

    def clear(self):
        self.groups = set()
        self.grid = np.zeros((self.size, self.size, 2), dtype=bool)
        self.point_to_group = {p : None for p in self.all_points}
        self.next = BLACK
        self.log = []
        # self.b_captured = 0
        # self.w_captured = 0
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
        print(output_str)

    def score(self, output=False):
        """
        1. check life & dead of every group
        2. count territory controlled by every living group
        3. Tromp-Tayler rule's scoring:
           score = stones on the board + territory + komi(6.5) if white
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
                p = expander.pop()
                #print("searched", p)
                if self.grid[p][BLACK]: # is black stone
                    if reach_color == None:
                        reach_color = BLACK
                    elif reach_color == WHITE:
                        reach_color = NEUTRAL
                elif self.grid[p][WHITE]: # is white stone
                    if reach_color == None:
                        reach_color = WHITE
                    elif reach_color == BLACK:
                        reach_color = NEUTRAL
                else: # is empty
                    searched.add(p)
                    if p in w_territory:
                        reach_color = WHITE
                        break
                    elif p in b_territory:
                        reach_color = BLACK
                        break
                    elif p in neutral_points:
                        reach_color = NEUTRAL
                        break
                    else:
                        for nbp in self.neighbors[p]:
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
    def territory_eval(self):
        if np.sum(self.grid) > self.size * self.size / 3:
            d = 4
        else:
            d = 5
        size_square = int(self.size * self.size)
        c_territory_eval.restype = POINTER(c_int * size_square)

        init_grid = np.zeros((self.size, self.size), dtype=int)
        init_grid[self.grid[:,:,BLACK]] = 64
        init_grid[self.grid[:,:,WHITE]] = -64
        grid_list = init_grid.ravel().tolist()
        init_arr = (c_int * size_square) (*grid_list)

        result_ptr = c_territory_eval(c_int(self.size), c_int(d), init_arr)

        result_arr = np.frombuffer(result_ptr.contents, dtype=c_int)
        b_terr = np.sum(result_arr > 0)
        w_terr = np.sum(result_arr > 0) + self.komi
        return b_terr, w_terr
    
    def get_uncond_life_groups(self):
        R = [set(), set()] # region [BLACK, WHITE]
        Y = [set(), set()] # block/group [BLACK, WHITE]
        NB = {}
        for x in range(2):
            # find *small* regions and thier nb blocks(groups)
            all_points = set(self.all_points)
            while len(all_points) > 0:
                p = all_points.pop()
                if self.grid[p][x]: continue
                searched = set()
                expander = set()
                expander.add(p)
                is_small = True
                tmp_nbg = set()
                while len(expander) > 0:
                    p = expander.pop()
                    has_nb_stone = False
                    for nbp in self.neighbors[p]:
                        if self.grid[p][x]: # has at last one nb stone
                            tmp_nbg.add(self.point_to_group[nbp])
                            has_nb_stone = True
                        elif nbp not in searched:
                            expander.add(nbp)
                    searched.add(p)
                    if not has_nb_stone:
                        is_small = False
                if is_small:
                    searched = frozenset(searched)
                    tmp_nbg = frozenset(tmp_nbg)
                    R[x].add(searched)
                    NB[searched] = tmp_nbg
                    Y[x].add(tmp_nbg)

            # find healthy relationship of blocks to regions 
            H = {} # group : region
            for r in R[x]:
                empty_points_in_r = {p for p in r if not any(self.grid[p])}
                for nbg in NB[r]:
                    if empty_points_in_r.issubset(nbg.neighbors):
                        if nbg in H:
                            H[nbg] = set([r]) # inital a set with a frozenset
                        else:
                            H[nbg].add(r)
            
            # do the algorithm iteration
            while True:
                # Remove from Y all x-block with less than 2 *healthy* regions in R
                Y[x] = { y for y in Y[x] if len(H[y]) >= 2
                }
                # Remove from R all x-region if any of its neighboring block is not in Y
                newR = {
                    r for r in R[x] if all([
                        nbg in Y[x]
                        for nbg in NB[r]
                    ])
                }
                if len(newR) == len(R[x]) or len(newR) == 0:
                    break
                else:
                    R[x] = newR
                    H = { k : v for k, v in H.items() if v in R[x] }
            # end while
        return Y
    """

    """
    double* c_board_eval(int size, int* init_grid, int d, double c1, double c2, double c3)
        PARAMETERS:
        d is the parameter for territory estimation
        init_grid is original board grid with black = 1, empty = 0, white = -1
        c1: estimated territory
        c2:territory based on only unconditionally alive groups
        c3: liberty per stones
        //c4: liberty race situation

        RETURN:
        array of double with length (size * size + 2)
        index 0 ~ size*size is policy grid
            policy:
                see that comment at calcPolicy()
        index size*size+1 is evaluation value
            evaluation value = black score - white score 
    """
    def eval(self):
        c1 = 0.4 # territory parameter
        c2 = 0.3 # live groups
        c3 = 0.1 # liberty parameter
        d = 5 # dilation parameter
        e = 0.12 # adjusting parameter  such that sigmoid(result_eval * e) ~ winrate

        size_square = int(self.size * self.size)
        # c_board_eval.restype = POINTER(c_double * (size_square + 2))
        c_board_eval.restype = c_double

        init_grid = np.zeros((self.size, self.size), dtype=int)
        init_grid[self.grid[:,:,BLACK]] = 1
        init_grid[self.grid[:,:,WHITE]] = -1
        grid_list = init_grid.ravel().tolist()
        init_arr = (c_int * size_square) (*grid_list)

        # result_ptr = c_board_eval(c_int(self.size), init_arr, c_int(d), c_double(c1), c_double(c2), c_double(c3))
        # result_array = np.frombuffer(result_ptr.contents, dtype=float)
        # result_eval = result_array[-1]
        # result_policy = np.resize(result_array, size_square + 1)
        result_eval = c_board_eval(c_int(self.size), init_arr, c_int(d), c_double(c1), c_double(c2), c_double(c3))
        return result_eval * e
    # end def eval
# end class Board
