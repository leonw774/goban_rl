import gc
from go import Stone, board_from_state, WHITE, BLACK
from time import time
import numpy as np

def masked_softmax(valid_mask, x):
    # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
    # sometime the sum exceed 1.0 due to numerical rounding
    x = x.astype("float64")
    # do not consider i if mask[i] == True
    masked_x = x[valid_mask]
    # stablize/normalize because if x too big will cause NAN when exp(x)
    normal_masked_x = masked_x - np.max(masked_x, axis=-1)
    masked_softmax_x = np.exp(normal_masked_x) / np.sum(np.exp(normal_masked_x), axis=-1)
    softmax = np.zeros(x.shape)
    softmax[valid_mask] = masked_softmax_x
    return softmax

def get_valid_mask(board, size):
    action_size = size ** 2 + 1
    valid_mask = np.zeros((action_size), dtype=bool)
    # transpose so that [x, y] become [x+y*size]
    valid_mask[:action_size-1] = np.transpose(board.grid==0).flatten()
    for p in board.suicide_illegal.union(board.same_state_illegal):
        valid_mask[p[0]+p[1]*size] = False
    if not np.all(valid_mask): # not empty
        valid_mask[action_size-1] = True # can pass
    return valid_mask

class MonteCarloNode():
    def __init__ (self, parent, parent_action, action_size, state, children_actions):
        self.parent = parent
        self.parent_action = parent_action
        self.state = state
        self.children = {k : None for k in children_actions}  # action_id : node

        # Q, N, UCT are updated in back propergate phrase
        """
            UCT is upper confience bound of v
            UCT[a] = Q[a] + P[a] where P is exploitation score
            P[a] = c_put * sqrt(ln(sum(N)) / N[a])
            c_put is coefficent, it controls the exploration rate
            1.4 is recommend in [0, 1] valued environment
            1.5~3.0 are used in [-1, 1] valued environment
        """
        self.Q = np.full(action_size, -1e10, dtype=float) # expected value of an action from this node's perspective
        self.Q[children_actions] = 0
        self.N = np.zeros(action_size, dtype="int16") # number of times that this node take path to an action

class MonteCarlo():
    def __init__ (self, model, simulation_num=1, simulation_depth=20):
        self.model = model
        self.size = model.size
        self.size_square = self.size**2
        self.action_size = self.size_square + 1
        self.root = None
        self.playout_limit = 0 # reset every time search method
        self.c_put = 1.5
        self.simulation_num = simulation_num
        self.simulation_depth = simulation_depth
        print("Monte Carlo Simulation Times / Depth:", simulation_num, simulation_depth)

    def clear_visit(self):
        self.root = None
        self.visited = {}
        # explicitly release memory
        gc.collect()

    def re_root(self, new_root_action):
        if self.root.children.get(new_root_action, None) is None:
            # removing ref of root deletes the whole tree
            self.root = None
            return
        # moving ref in of root to new_root deletes all other children
        self.root = self.root.children[new_root_action]

    def search(self, root_board, prev_action, playout, temperature):
        self.record_time = []
        self.playout_limit = playout
        self.playout_count = 0
        if self.root is None:
            # clean out visited & make root from root baord
            self.root = self.add_node(None, None, board_from_state(root_board.to_state()))
        else:
            # check root_board.hash == root node hash
            if root_board.grid_hash() != board_from_state(self.root.state).grid_hash():
                self.root = self.add_node(None, None, board_from_state(root_board.to_state()))
        self.prev_action = prev_action
        self.playout_loop()
        # print("playout time", np.sum(self.record_time), np.mean(self.record_time), np.std(self.record_time))

        valid_N = self.root.N[list(self.root.children.keys())]
        children_values = valid_N / np.sum(valid_N)

        # if no search is done
        if children_values.sum() == 0:
            print(list(self.root.children.keys()))
            print(self.root.N)
            # No search
            raise "no search done"
            
        # exp(100) is 1e43, so keep temperature > 0.01, otherwise it would overflow
        if temperature < 0.01: temperature = 0.01
        value_softmax = np.exp(children_values/temperature) / np.sum(np.exp(children_values/temperature))
        mcts_policy = np.zeros((self.action_size))
        mcts_policy[list(self.root.children.keys())] = value_softmax
        if temperature == 0.01:
            action = np.argmax(mcts_policy)
        else:
            action = np.random.choice(self.action_size, p=mcts_policy)
        # print(self.root.children[action].Q, self.root.children[action].N)

        # choose resign if value too low OR value is action value is much lower than root
        # print(Q", self.root.Q, "\nQa", self.root.Q[action])
        if self.root.Q[action] < self.model.resign_value:
            return 0, -1, mcts_policy
        self.re_root(action)
        # print(self.root.Q, self.root.N)
        return action%self.size, action//self.size, mcts_policy

    def playout_loop(self):
        for _ in range(self.playout_limit):
            leaf_node, leaf_action, is_terminal = self.select()
            new_node = None
            if is_terminal: 
                value = self.handle_terminal(leaf_node)
            else:
                new_node, value = self.expand(leaf_node, leaf_action)
            if new_node:
                self.backpropagate(new_node, value)
            if len(self.root.children) <= 1:
                break

    def select(self, batching = False):
        # print("selecting nodes")
        curnode = self.root
        prev_action = self.prev_action
        is_terminal = False
        while True:
            best_a = -1
            with np.errstate(divide="ignore", invalid="ignore"):
                U = curnode.Q + np.nan_to_num(self.c_put * np.sqrt(np.log(np.sum(curnode.N)) / curnode.N), posinf=2.0)
            max_a = np.argwhere(U == np.max(U)).ravel()
            best_a = np.random.choice(max_a)
            # check two consecutive pass
            is_terminal = (best_a == prev_action and best_a == self.size_square)
            if is_terminal:
                break
            # check if not 
            try:
                if curnode.children[best_a] is None: 
                    break
                else:
                    curnode = curnode.children[best_a]
                    prev_action = best_a
            except:
                print(curnode.Q)
                print(best_a)
                print(list(curnode.children.keys()))
                exit()
        # traverse to an unexpanded node
        # print("selected path:", action_path)
        return curnode, best_a, is_terminal

    def handle_terminal(self, terminal_node):
        board = board_from_state(terminal_node.state)
        winner, score_diff = board.score()
        # node is valued as board.next player
        value = 1.0 if board.next == winner else -1.0
        # print("terminal action value", value)
        return value
        
    def expand(self, leaf_node, leaf_action):
        board = board_from_state(leaf_node.state)

        # update game board
        if leaf_action >= self.size_square: # is pass
            board.pass_move()
            islegal = True
        else:
            x = leaf_action % self.size
            y = leaf_action // self.size
            add_stone = Stone(board, (x, y))
            islegal = add_stone.islegal
        
        # add node
        if islegal:
            # new_node = self.add_node(leaf_node, leaf_action, board)
            valid_mask = get_valid_mask(board, self.size)
            children_actions = valid_mask.nonzero()[0]
            new_node = MonteCarloNode(parent = leaf_node,
                            parent_action = leaf_action,
                            state=board.to_state(),
                            action_size = self.action_size,
                            children_actions=children_actions)  
            value = self.simulation(board)
            # parent node add children
            leaf_node.children[leaf_action] = new_node
            return new_node, -value
        else:
            # delete this child's info
            del leaf_node.children[leaf_action]
            leaf_node.Q[leaf_action] = -1e10
            leaf_node.N[leaf_action] = 0
            # re-dump board state for its illegal record,
            # it can reduce some illegal children in endgame point expand
            leaf_node.state = board.to_state()
            #print(leaf_action, "is illegal")
        return None, None

    def add_node(self, parent, leaf_action, board):
        valid_mask = get_valid_mask(board, self.size)
        children_actions = valid_mask.nonzero()[0]
        return MonteCarloNode(parent = parent,
                            parent_action = leaf_action,
                            state=board.to_state(),
                            action_size = self.action_size,
                            children_actions=children_actions)

    def simulation(self, board):
        # random moves, can pass
        sim_values = []
        for _ in range(self.simulation_num):
            sim_board = board_from_state(board.to_state())
            pass_count = 0
            simulation_depth_count = 0
            while simulation_depth_count < self.simulation_depth:
                a = np.random.randint(self.action_size)
                if a >= self.size_square:
                    sim_board.pass_move()
                    pass_count += 1
                    if pass_count == 2:
                        break
                else:
                    x = a % self.size
                    y = a // self.size
                    if np.any(board.grid[x,y]):
                        continue
                    added_stone = Stone(sim_board, (x, y))
                    if added_stone.islegal:
                        pass_count = 0
                simulation_depth_count += 1
            
            if pass_count == 2:
                winner, score_diff = board.score()
                # node is valued as board.next player
                v = 1.0 if board.next == winner else -1.0
            else:
                b, w = board.eval()
                v = (b - w) / self.size_square
                if board.next == WHITE: v = -v
            sim_values.append(v)
        return np.mean(sim_values)

    def backpropagate(self, leaf_node, value):
        # print("bp with value:", value)
        a = leaf_node.parent_action
        curnode = leaf_node.parent
        while curnode is not None:
            curnode.Q[a] = (value + curnode.N[a] * curnode.Q[a]) / (curnode.N[a] + 1)
            curnode.N[a] += 1
            a = curnode.parent_action
            curnode = curnode.parent
            value = -value # switch side