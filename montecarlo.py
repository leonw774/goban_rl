import gc
from go import Stone, board_from_state, WHITE, BLACK
# from time import time
import numpy as np
from ctypes import c_double, c_uint, c_int, POINTER, CDLL

np.seterr(all='raise')
_c_uct = CDLL("uct.dll").uct
invalidQ = -2.0

def masked_softmax(mask, x, temperature):
    # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
    # sometime the sum exceed 1.0 due to numerical rounding
    x = x.astype("float64")
    # do not consider i if mask[i] == True
    masked_x = x[mask]
    # stablize/normalize because if x too big will cause NAN when exp(x)
    # masked_x = masked_x - max(masked_x)
    exp_masked_x = np.exp(masked_x / temperature)
    masked_softmax_x = exp_masked_x / np.sum(exp_masked_x)
    softmax = np.zeros(x.shape)
    softmax[mask] = masked_softmax_x
    return softmax

class MonteCarloNode():
    __slots__ = ("state", "value", "policy", "eval", "valid_actions", "children", "Q", "N")
    def __init__(self, state, value, policy, eval, valid_actions, action_size):
        self.state = state
        self.value = value
        self.policy = policy
        self.eval = eval
        self.children = {}  # action_id : node
        self.valid_actions = valid_actions # mask np array
        # Q, N are updated in back propergate phrase
        """
            UCT is upper confience bound of v
            UCT[a] = Q[a] + P[a] where P is exploitation score
            P[a] = cput * sqrt(ln(sum(N)) / N[a])
            cput is coefficent, it controls the exploration rate
            1.4 is recommend in [0, 1] valued environment
            1.5~3.0 are used in [-1, 1] valued environment
        """
        self.Q = np.full(action_size, invalidQ, dtype=np.float32) # expected value of an action from this node's perspective
        self.Q[self.valid_actions] = 0
        self.N = np.zeros(action_size, dtype=np.uint8) # number of times that this node take path to an action
    
    def __str__(self):
        return str({"value":self.value, "eval":self.eval, "valid_actions":self.valid_actions, "Q":self.Q, "N":self.N })

class MonteCarlo():
    def __init__(self, model, simulation_num=1, simulation_depth=8):
        self.model = model

        self.board_size = model.board_size
        self.board_size_square = self.board_size**2
        self.action_size = self.board_size_square + 1

        self.root = None
        self.playout_limit = 0 # reset every time search method
        self.cput = 1.4
        _c_uct.restype = POINTER(c_double * self.action_size)
        self.simulation_num = simulation_num
        self.simulation_depth = simulation_depth
        #print("Monte Carlo Simulation Times / Depth:", simulation_num, simulation_depth)

    def clear_visit(self):
        self.root = None
        # explicitly release memory
        gc.collect()

    def re_root(self, new_root_action):
        if self.root.children.get(new_root_action, None) is None:
            # removing ref of root deletes the whole tree
            self.root = None
            return
        # moving ref in of root to new_root deletes all other children
        self.root = self.root.children[new_root_action]
        gc.collect()

    def search(self, root_board, prev_action, playout, output = False):
        self.record_time = []
        self.playout_limit = playout
        self.playout_count = 0
        if self.root is None:
            # clean out visited & make root from root board
            self.root = self.add_node(None, None, board_from_state(root_board.state), root_board.eval())
        else:
            # check root_board.hash == root node hash
            if root_board.grid_hash() != board_from_state(self.root.state).grid_hash():
                self.root = self.add_node(None, None, board_from_state(root_board.state), root_board.eval())
        self.prev_action = prev_action # prev_action == -1 if there is none
        self.playout_loop()
        # print("playout time", np.sum(self.record_time), np.mean(self.record_time), np.std(self.record_time))

        # if no search is done
        if len(self.root.valid_actions) == 0:
            print(self.root.valid_actions)
            print(self.root.N)
            # No search
            raise BaseException("no search done")
            
        # exp(100) is 1e43, so keep temperature > 0.01, otherwise it would overflow
        invalid_mask = root_board.get_valid_move_mask()
        masked_valu_softmax = masked_softmax(invalid_mask, self.root.N, 1.0)
        action = np.random.choice(self.action_size, 1, p = masked_valu_softmax)[0]

        if (output):
            outstr = ""
            for i in range(self.board_size):
                for j in range(self.board_size):
                    outstr += "%.1f(%d) " % (self.root.Q[j+i*self.board_size], self.root.N[j+i*self.board_size])
                outstr += "\n"
            print("Q\n", outstr, "value", self.root.value, ", Qa", self.root.Q[action], ", Na", self.root.N[action])
            try:
                print("a-value", self.root.children[action].value)
            except:
                print("root has no children")

            # print("child Q N:")
            # print(self.root.children[action].Q, self.root.children[action].N)

        # choose resign if value too low OR value is action value is much lower than root
        if prev_action != -1 and self.root.Q[action] < self.model.RESIGN_THRESHOLD:
            return 0, -1
        
        self.re_root(action)
        # print(self.root.Q, self.root.N)
        return action % self.board_size, action // self.board_size

    def playout_loop(self):
        for i in range(self.playout_limit):
            node_path, action_path, is_terminal = self.select()
            if is_terminal: 
                value = self.handle_terminal(node_path[-1])
            else:
                value = self.expand(node_path[-1], action_path[-1])
            if value:
                self.backpropagate(node_path, action_path, value)

    def select(self, batching = False):
        # print("selecting nodes")
        curnode = self.root
        prev_action = self.prev_action
        is_terminal = False
        action_path = []
        node_path = []
        while True:
            best_a = -1
            # print("Q", curnode.Q, "\nN", curnode.N)
            Q_c_arr = (c_double * self.action_size) (* curnode.Q.tolist())
            N_c_arr = (c_int * self.action_size) (* curnode.N.tolist())
            result_ptr = _c_uct(Q_c_arr, N_c_arr, c_double(self.cput), c_uint(self.action_size))
            U = np.frombuffer(result_ptr.contents)
            U *= curnode.policy
            # print("Q", curnode.Q, "\nN", curnode.N, "\nU", U)
            max_a = np.argwhere(U == U.max()).flatten()
            best_a = np.random.choice(max_a)
            action_path.append(best_a)
            node_path.append(curnode)
            # check two consecutive pass
            is_terminal = (best_a == prev_action and best_a == self.board_size_square)
            if is_terminal:
                break
            # check if not 
            # print(best_a)
            if curnode.children.get(best_a, None) is None: 
                # traverse to an unexpanded node
                break
            else:
                curnode = curnode.children[best_a]
                prev_action = best_a
        return node_path, action_path, is_terminal

    def handle_terminal(self, terminal_node):
        board = board_from_state(terminal_node.state)
        winner, score_diff = board.score()
        # node is valued as board.next player
        if self.model.CRITIC_OUTLAYER_ACTIVATION_FUNC == "tanh":
            value = 1.0 if board.next == BLACK else -1.0
        # print("terminal action value", value)
        return value
        
    def expand(self, node, action):
        board = board_from_state(node.state)
        node_playas = board.next
        # update game board
        if action >= self.board_size_square: # is pass
            board.pass_move()
            islegal = True
        else:
            x = action % self.board_size
            y = action // self.board_size
            add_stone = Stone(board, (x, y))
            islegal = add_stone.islegal
        
        # add node
        if islegal:
            # check eval
            # if BLACK did a move, eval should not be too low. if WHITE did a move, eval should not be high
            eval = board.eval()
            if (board.next == WHITE and eval >= node.eval - 1) or (board.next == BLACK and eval <= node.eval + 1):
                new_node = self.add_node(node, action, board, eval)
                value = new_node.value
                # switch size
                if node_playas == WHITE: 
                    if self.model.CRITIC_OUTLAYER_ACTIVATION_FUNC == "tanh":
                        value = -value
                    else:
                        value = 1 - value
                return value
        
        # delete this child from valid list and reset its info
        node.valid_actions[action] = False
        node.Q[action] = invalidQ
        node.N[action] = 0
        # re-dump board state for its illegal record,
        # it can reduce some illegal children in endgame point expand
        node.state = board.state
        #print(leaf_action, "is illegal")
        return None

    def add_node(self, parent, action, board, eval):
        policy, value = self.model.get_policy_value(board.grid, board.next)
        valid_actions = board.get_valid_move_mask()
        newnode = MonteCarloNode(state=board.state,
                        eval = eval,
                        value = value,
                        policy = policy,
                        valid_actions = valid_actions,
                        action_size = self.action_size)
        # print(newnode)
        # parent add child
        if parent: parent.children[action] = newnode
        return newnode
    
    """ 
    def simulation(self, board):
        # random moves, can pass
        sim_values = []
        for _ in range(self.simulation_num):
            sim_board = board_from_state(board.state)
            pass_count = 0
            simulation_depth_count = 0
            while simulation_depth_count < self.simulation_depth:
                a = np.random.randint(self.action_size)
                if a >= self.board_size_square:
                    sim_board.pass_move()
                    pass_count += 1
                    if pass_count == 2:
                        break
                else:
                    x = a % self.board_size
                    y = a // self.board_size
                    if np.any(board.grid[x,y]):
                        continue
                    added_stone = Stone(sim_board, (x, y))
                    if added_stone.islegal:
                        pass_count = 0
                simulation_depth_count += 1
            
            if pass_count == 2:
                winner, score_diff = sim_board.score()
                # node is valued as board.next player
                v = 1.0 if board.next == winner else -1.0
            else:
                b, w = sim_board.eval()
                v = np.tanh(b - w)
                if board.next == WHITE: v = -v
            sim_values.append(v)
        return np.max(sim_values)
        """

    def backpropagate(self, node_path, action_path, value):
        for node, action in zip(reversed(node_path), reversed(action_path)):
            node.Q[action] = (value + node.N[action] * node.Q[action]) / (node.N[action] + 1)
            node.N[action] += 1
            # switch side
            if self.model.CRITIC_OUTLAYER_ACTIVATION_FUNC == "tanh":
                value = -value # tanh
            else:
                value = 1 - value # sigmoid
