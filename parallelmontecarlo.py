import gc
import threading
from pickle import loads, dumps
from go import Stone, WHITE, BLACK
from time import time
import numpy as np

def masked_softmax(mask, x, temperature):
    if len(x.shape) != 1:
        print("softmax input must be 1-D numpy array")
        return
    # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
    # sometime the sum exceed 1.0 due to numerical rounding
    x = x.astype("float64")
    # do not consider i if mask[i] == True
    mask_indice = np.argwhere(~mask)
    masked_x = x[mask_indice]
    # stablize/normalize because if x too big will cause NAN when exp(x)
    normal_masked_x = masked_x - np.max(masked_x)
    masked_softmax_x = np.exp(normal_masked_x/temperature) / np.sum(np.exp(normal_masked_x/temperature))
    softmax = np.zeros(x.shape)
    softmax[mask_indice] = masked_softmax_x
    return softmax

class BatchInfo():
    def __init__ (self, node_path, action_path, state):
        self.node_path = node_path
        self.action_path = action_path
        self.state = state
        self.value = None

class MonteCarloNode():
    def __init__ (self, state, policies, value, children_keys):
        self.state = state
        self.policies = policies
        self.value = value
        # Q, N are updated in back propergate phrase
        self.Q = {k : 0 for k in children_keys} # expected value of an action from this node's perspective
        self.N = {k : 0 for k in children_keys} # number of times that this node take path to an action
        self.expanding = {k : False for k in children_keys} # True when selected to expand... set to False in back propagate phase
        self.visiting = {k : 0 for k in children_keys} # +1 when selected in path... -1 back propagate phase
        self.children = {k : None for k in children_keys}  # action_id : node

class ParallelMonteCarlo():
    def __init__ (self, model, batch_size, thread_max = 0):
        """
            model: playmodel
            batch_size: larger than 8 will not be too effective
            use_thread: True or False
            thread_max: if this parameter is set, thread_num will equal to the number of children of root
                approximately equal to degree
        """
        self.model = model
        self.size = model.size
        self.size_square = self.size**2
        self.action_size = self.size_square + 1
        self.dirichlet_alpha = [10 / self.size_square]*self.action_size
        self.root = None
        self.playout_limit = 0 # reset every time search method called
        self.batch_size = batch_size
        self.thread_max = thread_max
        self.treelock = threading.Lock()
        self.playout_count_lock = threading.Lock()
        print("Parallel Monte Carlo parameters: Batch size:", batch_size, "thread max:", thread_max)
    
    def clear_visit(self):
        self.root = None
        self.visited = {}
        # explicitly release memory
        gc.collect()

    def re_root(self, new_root_action):
        if self.root.children[new_root_action] is None:
            # remove ref of root delete whole tree
            self.root = None
            return
        # remove ref in root excpet new_root
        self.root = self.root.children[new_root_action]

    def search(self, root_board, prev_action, playout, temperature):
        self.record_time = []
        self.playout_limit = playout
        self.playout_count = 0
        if self.root is None:
            # clean out visited & make root from root baord
            self.root = self.add_node(loads(dumps(root_board)))
        else:
            # check root_board.hash = root node hash
            if root_board.grid_hash() != loads(self.root.state).grid_hash():
                raise "root_board.grid_hash() != self.root.state.hash. Doesn't know how to handle yet"
        self.prev_action = prev_action
        self.threaded_playout_loop()
        #print("playout time", np.sum(self.record_time), np.mean(self.record_time), np.std(self.record_time))

        N_sum = sum(self.root.N)
        children_action = list(self.root.N.keys())
        children_values = np.array([v / N_sum for v in self.root.N])
        if children_values.sum() > 0:
            # exp(100) is 1e43, so keep temperature > 0.01, otherwise it would overflow
            if temperature < 0.01: temperature = 0.01
            value_softmax = np.exp(children_values/temperature) / np.sum(np.exp(children_values/temperature))
            mcts_policy = np.zeros((self.action_size))
            mcts_policy[children_action] = value_softmax
            try:
                action = np.random.choice(self.action_size, p=mcts_policy)
            except Exception as e:
                print(temperature)
                print(children_values)
                print(value_softmax)
                print(mcts_policy)
                raise e
            # choose resign if value lower than resign_value
            if self.root.Q[action] < self.model.resign_value:
                return 0, -1, mcts_policy
        else:
            # some things wrong
            raise "No search in MCTS???"
        self.re_root(action)
        return action%self.size, action//self.size, mcts_policy

    def threaded_playout_loop(self):
        #print("threaded_playout_loop")
        while self.playout_count < self.playout_limit:
            self.batch_playout()
            if len(self.root.children) <= 1:
                break

    def playout(self):
        node_path, action_path, is_terminal = self.select()
        if is_terminal: 
            value = self.handle_terminal(node_path[-1])
        else:
            value = self.expand(node_path[-1], action_path[-1])
        if value:
            self.backpropagate(node_path, action_path, value)

    def batch_playout(self):
        batch_list = []
        for _ in range(self.batch_size):
            node_path, action_path, is_terminal = self.select(batching = True)
            if len(action_path) == 0: # no path available to choose
                break
            if is_terminal:
                batchinfo = None
                value = self.handle_terminal(node_path[-1])
            else:
                value = None
                batchinfo = self.delayed_expand(node_path, action_path)
            if batchinfo is not None: # if expand is legal
                batch_list.append(batchinfo)
            elif value is not None: # if this action path leads to visited or terminal
                self.backpropagate(node_path, action_path, value)
            #illegal action: do nothing

            # self.playout_count_lock.acquire()
            self.playout_count += 1
            # SHOULD BE CRITICAL SECTION
            # BUT for speed reason, we doesn't really need precise number of playout
            # it will end nonetheless
            # self.playout_count_lock.release()

        if len(batch_list) > 0:
            self.batch_add_node(batch_list)
        self.backpropagate_with_batch(batch_list)

    def select(self, batching = False):
        #print("selecting nodes")
        curnode = self.root
        node_path = []
        action_path = []
        is_terminal = False
        #fuse = self.playout_limit
        #while fuse > 0:
        self.treelock.acquire()
        while True:
            #fuse -= 1
            best_a = -1
            N_visiting_sum_sqrt = np.sqrt(sum(curnode.N.values()) + sum(curnode.visiting.values()))
            """
                UCT is upper confience bound of v
                UCT[a] = Q[a] + P[a] where P is some formula for controlling exploration-exploitation balance
                P[a] = c * sqrt(sum(N)) / (1 + N[a])
                This formula is used by AlphaGoZero
                c is coefficent, it controls the exploration rate
                1.4 is recommend in [0, 1] valued environment
                1.5~3.0 are used in [-1, 1] valued environment
            """
            """
                Virtual Loss
                is to discourage over-exploitaion in parallel, which change P's formula to:
                P[a] = c * sqrt(sum(N)+ sum(O)) / (1 + N[a] + O[a])
                where O[a] is the number of playout visiting action 'a' that's not finished back-propergation yet
            """
            cur_filtered_uct_dict = {
                k : curnode.Q.get(k,0)+2*curnode.policies[k]*N_visiting_sum_sqrt/(1+curnode.N[k]+curnode.visiting[k])
                for k in curnode.children
                if not curnode.expanding[k] and (k == self.size_square or not curnode.children[k] in node_path)
            }
            
            if len(cur_filtered_uct_dict) > 0:
                best_a = max(cur_filtered_uct_dict, key=cur_filtered_uct_dict.get)
            # if no valid children
            # although curnode is visited... in some sense this is a terminal node
            if best_a == -1:
                break
            node_path.append(curnode)
            action_path.append(best_a)
            # check two consecutive pass
            prev_action = action_path[-2] if len(action_path) > 1 else self.prev_action
            if best_a == prev_action and best_a == self.size_square:
                is_terminal = True
                break
            # check if not visited
            if curnode.children[best_a] is None:
                # tell other thread this is visited
                curnode.expanding[best_a] = True
                break
            else:
                curnode.visiting[best_a] += 1
                curnode = curnode.children[best_a]
        self.treelock.release()
        # traverse to an unexpanded node
        # print("selected path:", action_path)
        return node_path, action_path, is_terminal

    def handle_terminal(self, terminal_node):
        board = loads(terminal_node.state)
        winner, score_diff = board.score()
        # node is valued as board.next player
        value = 1.0 if board.next == winner else -1.0
        #print("terminal action value", value)
        return value

    def delayed_expand(self, node_path, action_path):
        leaf_node = node_path[-1]
        leaf_action = action_path[-1]
        board = loads(leaf_node.state)

        # update game board
        islegal = True
        if leaf_action >= self.size_square: # is pass
            board.pass_move()
        else:
            x = leaf_action % self.size
            y = leaf_action // self.size
            add_stone = Stone(board, (x, y))
            islegal = add_stone.islegal

        if islegal:
            # don't add children to parent node yet, that's done in batch_add_node
            # print(new_zhash, "is batching")
            return BatchInfo(node_path=node_path, action_path=action_path, 
                    state=dumps(board))
        else:
            self.treelock.acquire()
            del leaf_node.children[leaf_action]
            del leaf_node.Q[leaf_action]
            del leaf_node.N[leaf_action]
            # re-dump board state for its illegal record,
            # it can reduce some illegal children in endgame point expand
            leaf_node.state = dumps(board)
            self.treelock.release()
            # print(leaf_action, "is illegal")
        # return None, None so that it won't go to backpropergate
        return None
        

    def expand(self, leaf_node, leaf_action):
        board = loads(leaf_node.state)

        # update game board
        islegal = True
        if leaf_action >= self.size_square: # is pass
            board.pass_move()
        else:
            x = leaf_action % self.size
            y = leaf_action // self.size
            add_stone = Stone(board, (x, y))
            islegal = add_stone.islegal
        
        # add node
        if islegal:
            new_node = self.add_node(board)
            value = new_node.value
            # parent node add children
            self.treelock.acquire()
            leaf_node.children[leaf_action] = new_node
            self.treelock.release()
            return -value
        else:
            self.treelock.acquire()
            del leaf_node.children[leaf_action]
            del leaf_node.Q[leaf_action]
            del leaf_node.N[leaf_action]
            leaf_node.state = dumps(board)
            self.treelock.release()
            #print(leaf_action, "is illegal")
        return

    def batch_add_node(self, batch_list):
        # print("batch_add_node")
        # batch-y get value
        batched_mask = np.empty((len(batch_list), self.action_size), dtype=bool)
        batched_board_grid = np.empty((len(batch_list), self.size, self.size, 2))
        for i, binfo in enumerate(batch_list):
            board = loads(binfo.state)
            batched_mask[i] = self.model.get_invalid_mask(board)
            if board.next == WHITE:
                batched_board_grid[i] = board.grid[:,:,[1,0]]
            else:
                batched_board_grid[i] = board.grid

        # fix initialization problem in threading scheme in Ubuntu
        # with self.model.graph.as_default():
        #     with self.model.session.as_default():
        batched_logit = self.model.actor.predict_on_batch(batched_board_grid) # shape = (batch_size, action_size)
        batched_value = self.model.critic.predict_on_batch(batched_board_grid) # shape = (batch_size, 1)
        batched_noise = np.random.dirichlet(alpha=self.dirichlet_alpha, size=(len(batch_list)))
        batched_noised_logit = 0.75 * batched_logit + 0.25 * batched_noise

        for i, binfo in enumerate(batch_list):
            binfo.value = batched_value[i][0]
            # create new node
            masked_intuitions = masked_softmax(batched_mask[i], batched_noised_logit[i], 1.0)
            children_actions = masked_intuitions.nonzero()[0]
            new_node = MonteCarloNode(state=binfo.state,
                                    policies=masked_intuitions,
                                    value=batched_value[i][0],
                                    children_keys=children_actions)
            # update parent's node
            self.treelock.acquire()
            binfo.node_path[-1].children[binfo.action_path[-1]] = new_node
            self.treelock.release()
    # end def batch_add_node

    def add_node(self, board):
        #print("adding node id", zhash)
        masked_intuitions = self.model.get_masked_intuitions(board, 1.0)
        value = self.model.get_value(board.next, board.grid)
        children_actions = masked_intuitions.nonzero()[0]
        return MonteCarloNode(state=dumps(board),
                            policies=masked_intuitions,
                            value=value,
                            children_keys=children_actions)
    # end def add_node

    def backpropagate(self, node_path, action_path, value):
        # print("bp with value:", value)
        self.treelock.acquire()
        for rev_i, a in reversed(list(enumerate(action_path))):
            curnode = node_path[rev_i]
            Qa = curnode.Q[a]
            Na = curnode.N[a]
            curnode.Q[a] = (value + Na * Qa) / (Na + 1)
            curnode.N[a] += 1
            curnode.visiting[a] -= 1
            curnode.expanding[a] = False
            value = -value # to switch side
        self.treelock.release()

    def backpropagate_with_batch(self, batch_list):
        # print("batch bp with value:")
        self.treelock.acquire()
        for binfo in batch_list:
            node_path, action_path, value = binfo.node_path, binfo.action_path, binfo.value
            for rev_i, a in reversed(list(enumerate(action_path))):
                curnode = node_path[rev_i]
                Qa = curnode.Q[a]
                Na = curnode.N[a]
                curnode.Q[a] = (value + Na * Qa) / (Na + 1)
                curnode.N[a] += 1
                curnode.visiting[a] -= 1
                curnode.expanding[a] = False
                value = -value # to switch side
        self.treelock.release()
