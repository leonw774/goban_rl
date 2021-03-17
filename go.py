#!/usr/bin/env python
# coding: utf-8
import numpy as np

"""Go library

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

Edited by leow774
"""

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
        #self.id = np.random.randint(2147483647)
        # frozenset of tuple(x, y)
        self.stones = stones_frozenset
        self.neighbors = neighbors_frozenset
        self.color = color
    
    # def __str__(self):
    #     return ("Group { color:" + str(self.color) +
    #             ", stones:" + str([str(p) for p in self.stones]) +
    #             ", neibors:" + str([str(p) for p in self.neighbors]) +
    #             " }")

int64_high = 2**63 - 1
int64_low = -(2**63)
ZOBRIST_INIT = np.random.randint(int64_low, int64_high, dtype=np.int64)
# for grid positions
ZOBRIST_GRID = np.random.randint(int64_low, int64_high, size=(19, 19, 3), dtype=np.int64)
# for b_captured, w_captured, next
ZOBRIST_NEXT = np.random.randint(int64_low, int64_high, dtype=np.int64)

def board_from_state(state):
    return Board(state = state)

class Board(object):
    def __init__(self, size=19, komi=6.5, state=None, is_debug=False):
        """Create and initialize an empty board."""
        # grid is a numpy array representation of the board
        # empty = (0.0, 0.0) black = (1.0, 0) white = (0.0, 1.0)

        if state is not None:
            self.from_state(state)
            self.b_captured = 0
            self.w_captured = 0
        else:
            self.size = size
            self.komi = komi
            self.grid = np.zeros((self.size, self.size, 2))
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
            self.log = []
            self.b_captured = 0
            self.w_captured = 0
            
        self.same_state_illegal = set()
        self.suicide_illegal = set()
        self.outputDebug = is_debug

    def to_state(self):
        return [self.size, self.komi, self.grid.tobytes(), self.groups.copy(), self.next, self.neighbors, self.all_points, self.log] # keep last 8 moves

    def from_state(self, state):
        self.size = state[0]
        self.komi = state[1]
        self.grid = np.fromstring(state[2])
        self.grid.resize((self.size, self.size, 2))
        self.groups = state[3].copy()
        self.next = state[4]
        self.neighbors = state[5]
        self.all_points = state[6]
        self.log = state[7]

    def bound(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size 

    # Zobrist Hash
    def grid_hash(self):
        bool_grid_flat = self.grid.astype(bool).ravel()
        bool_grid_flat = np.append(bool_grid_flat, bool(self.next))
        bool_grid_bytes = np.argwhere(bool_grid_flat).ravel().tobytes()
        return bool_grid_bytes

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
            # if self.outputDebug:
            #     print("illegal: suicidal move")
            return False
        
        # temporary take off "dead" stones from grid to see if that would make same state
        for dead_nbeg in killed_nb_enemy_group:
            for s in dead_nbeg.stones:
                self.update_grid(s, None)
        # check latest previous 6 board state (preventing n-ko rotation, n <= 3)
        same_state_rewind_limit = 6
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
        if sum(self.grid[point]) == 1: # has stone
            return False
        self.update_grid(point, self.next)

        # record neighbor groups
        nb_friend_groups = set()
        nb_enemy_groups = set()
        for nbp in self.neighbors[point]:
            if sum(self.grid[point]) == 0: continue # has no stone
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
        # if self.outputDebug: self.debugPrint()
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
                        for nbp in self.neighbors[search_p]:
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
        w_score = np.sum(self.grid[:,:,WHITE]==1) + len(w_territory) + self.komi    
        b_score = np.sum(self.grid[:,:,BLACK]==1) + len(b_territory)
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
    # end def cal_score

    '''
    Territory Evaluation Algorithm from
    Bouzy, B. (2003). Mathematical Morphology Applied to Computer Go. Int. J. Pattern Recognit. Artif. Intell., 17, 257-268.

    # Defination

    A = set of elements
    Dilation: D(A) = A + neigbors of A
    Erosion: E(A) = A - neighbors of complements of A
    External Boundary: ExtBound(A) = D(A) - A
    Internal Boundary: IntBound(A) = A - E(A)
    Closing: Close(A) = E(D(A))
    Cloasing is safe territory
    Terriorty Potential Evaluation Operator X(e, d) = E^e . D^d
    X(e, d) is the operation that do dilation d times and then erosion e times

    # Zobrist's Model To Recognize "Influence"
    assign +64/-64 to black/white points, and 0 elsewhere
    for p in every points:
        for n in neighbors of p:
            if n < 0
                p -= 1
            elif n > 0
                p += 1

    The Zobrist model above has similar effect as dilation.
    So defined that operator as Dz
    then define Ez in an analogous way:
    for p in every points:
        for n in neighbors of p:
            if color[n] != color[p]:
                if p > 0:
                    p -= 1
                elif p < 0:
                    p += 1

    It is figured that if there ia only one stone on board
    the operater X(e, d) must give same result as identity operator
    Thus e = d * (d - 1) + 1

    It found out that X gives better result when d = 4 or 5
    The bigger "d" is , the larger the scale of recognization territories is
    '''    

    def eval(self):
        eval_grid = np.zeros((self.size, self.size))
        eval_grid[self.grid[:,:,BLACK] == 1] = 64
        eval_grid[self.grid[:,:,WHITE] == 1] = -64

        if np.sum(self.grid) > self.size * self.size / 3:
            d = 4
            e = 13
        else:
            d = 5
            e = 21
        
        # Dilate
        for _ in range(d):
            new_eval_grid = eval_grid.copy()
            for p in self.all_points:
                for n in self.neighbors[p]:
                    if eval_grid[n] > 0:
                        new_eval_grid[p] += 1
                    elif eval_grid[n] < 0:
                        new_eval_grid[p] -= 1
            eval_grid = new_eval_grid
        
        # Erase
        for _ in range(e):
            new_eval_grid = eval_grid.copy()
            for p in self.all_points:
                if eval_grid[p] == 0: continue
                i = 1 if eval_grid[p] > 0 else -1
                for n in self.neighbors[p]:
                    if eval_grid[n] * eval_grid[p] <= 0: # if different color
                        new_eval_grid[p] -= i
                        if new_eval_grid[p] == 0: break
            eval_grid = new_eval_grid
        
        b_potentials = np.sum(eval_grid > 0)
        w_potentials = np.sum(eval_grid < 0) + self.komi

        return b_potentials, w_potentials
# end class Board