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
    valid_mask[:action_size-1] = np.transpose(np.sum(board.grid, axis=2)==0).flatten()
    for p in board.suicide_illegal.union(board.same_state_illegal):
        valid_mask[p[0]+p[1]*size] = False
    if not np.all(valid_mask): # not empty
        valid_mask[action_size-1] = True # can pass
    return valid_mask

class BatchInfo():
    def __init__ (self, node_path, action_path, state):
        self.node_path = node_path
        self.action_path = action_path
        self.state = state
        self.value = None

class MonteCarloNode():
    def __init__ (self, state, policies, value, children_actions):
        self.state = state
        self.policies = policies
        self.value = value
        # Q, N, UCT are updated in back propergate phrase
        self.children_actions = children_actions
        self.Q = np.full(policies.shape, -float("inf"), dtype=float) # expected value of an action from this node's perspective
        self.Q[children_actions] = 0
        self.sumN = 0
        self.N = np.zeros(policies.shape, dtype="int16") # number of times that this node take path to an action
        self.expanding = np.zeros(policies.shape, dtype=bool)
        self.visiting = np.zeros(policies.shape, dtype="int16") # +1 when selected in path... -1 back propagate phase
        self.children = {k : None for k in children_actions}  # action_id : node

class MonteCarlo():
    def __init__ (self, model, batch_size):
        """
            model: playmodel
            batch_size: larger than 8 will not be too effective
        """
        self.model = model
        self.size = model.size
        self.size_square = self.size**2
        self.action_size = self.size_square + 1
        self.dirichlet_alpha = [10 / self.size_square]*self.action_size
        self.root = None
        self.playout_limit = 0 # reset every time search method called
        self.batch_size = batch_size
        print("Monte Carlo parameters: Batch size:", batch_size)

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
            self.root = self.add_node(board_from_state(root_board.to_state()))
        else:
            # check root_board.hash == root node hash
            if root_board.grid_hash() != board_from_state(self.root.state).grid_hash():
                self.root = self.add_node(board_from_state(root_board.to_state()))
        self.prev_action = prev_action
        self.playout_loop()
        # print("playout time", np.sum(self.record_time), np.mean(self.record_time), np.std(self.record_time))

        children_values = self.root.N[self.root.children_actions] / self.root.sumN

        # if no search is done
        if children_values.sum() == 0:
            print(self.root.children_actions)
            print(self.root.N)
            print(self.root.sumN)
            # No search -- use policy
            mask = np.zeros((1, self.action_size), dtype=bool)
            mask[self.root.children_actions] = True
            policy_softmax = masked_softmax(mask, self.root.policies)[0]
            action = np.random.choice(self.action_size, p=policy_softmax)
            return action%self.size, action//self.size, policy_softmax
            
        # exp(100) is 1e43, so keep temperature > 0.01, otherwise it would overflow
        if temperature < 0.01: temperature = 0.01
        value_softmax = np.exp(children_values/temperature) / np.sum(np.exp(children_values/temperature))
        mcts_policy = np.zeros((self.action_size))
        mcts_policy[self.root.children_actions] = value_softmax
        try:
            action = np.random.choice(self.action_size, p=mcts_policy)
        except Exception as e:
            print(mcts_policy)
            raise e

        # choose resign if value too low OR value is action value is much lower than root
        print("Q", self.root.Q[action])
        if self.root.Q[action] < self.model.resign_value:
            return 0, -1, mcts_policy
        
        self.re_root(action)
        return action%self.size, action//self.size, mcts_policy

    def playout_loop(self):
        if self.batch_size > 1:
            while self.playout_count < self.playout_limit:
                #t = time()
                self.batch_playout()
                if len(self.root.children) <= 1:
                    break
                #self.record_time.append(time()-t)
        else:
            for _ in range(self.playout_limit):
                self.playout()
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
            self.playout_count += 1
        #end while
        if len(batch_list) > 0:
            self.batch_add_node(batch_list)
        self.batch_backpropagate(batch_list)

    def select(self, batching = False):
        # print("selecting nodes")
        curnode = self.root
        node_path = []
        action_path = []
        is_terminal = False
        #fuse = self.playout_limit
        #while fuse > 0:
        while True:
            #fuse -= 1
            best_a = -1
            """
                UCT is upper confience bound of v
                UCT[a] = Q[a] + P[a] where P is exploitation score
                P[a] = c_put * sqrt(sum(N)) / (1 + N[a])
                This formula is used by AlphaGoZero
                c_put is coefficent, it controls the exploration rate
                1.4 is recommend in [0, 1] valued environment
                1.5~3.0 are used in [-1, 1] valued environment
            """
            """
                Virtual Loss
                is to discourage over-exploitaion in parallel, which change P's formula to:
                P[a] = c * sqrt(sum(N)+ sum(O)) / (1 + N[a] + O[a])
                where O[a] is the number of playout visiting action 'a' that's not finished back-propergation yet
            """
            c_put = 2
            U = curnode.Q + c_put * (np.sqrt(curnode.sumN + np.sum(curnode.visiting))) * curnode.policies / (1 + curnode.N + curnode.visiting)
            U[curnode.expanding] = -float("inf") # can only choose from not expanding children
            best_a = np.argmax(U)
            if best_a not in curnode.children_actions:
                break
            node_path.append(curnode)
            action_path.append(best_a)
            # check two consecutive pass
            prev_action = action_path[-2] if len(action_path) > 1 else self.prev_action
            is_terminal = (best_a == prev_action and best_a == self.size_square)
            if is_terminal:
                break
            # check if not visited
            if curnode.children[best_a] is None: 
                # mark this not visited children
                # this mark will be erased (set to False) in backpropagate
                curnode.expanding[best_a] = True
                curnode.visiting[best_a] += 1
                #print("batching action", best_a)
                break
            else:
                curnode.visiting[best_a] += 1
                curnode = curnode.children[best_a]
        # traverse to an unexpanded node
        # print("selected path:", action_path)
        return node_path, action_path, is_terminal

    def handle_terminal(self, terminal_node):
        board = board_from_state(terminal_node.state)
        winner, score_diff = board.score()
        # node is valued as board.next player
        value = 1.0 if board.next == winner else -1.0
        #print("terminal action value", value)
        return value
    
    def delayed_expand(self, node_path, action_path):
        leaf_node = node_path[-1]
        leaf_action = action_path[-1]
        board = board_from_state(leaf_node.state)
        #print("leaf node state id:", board.grid_hash())

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
            #print(new_zhash, "is batching")
            return BatchInfo(node_path=node_path, action_path=action_path, 
                    state=board.to_state())
        else:
            # delete this action's info
            del leaf_node.children[leaf_action]
            del_mask = (leaf_node.children_actions != leaf_action)
            leaf_node.children_actions = leaf_node.children_actions[del_mask]
            leaf_node.Q[leaf_action] = -float("inf")
            leaf_node.N[leaf_action] = 0
            # re-dump board state for its illegal record,
            # it can reduce some illegal children in endgame point expand
            leaf_node.state = board.to_state()
            #print(action_path, leaf_action, "is illegal")
        # return None so that it won't go to backpropergate
        return None
        
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
            new_node = self.add_node(board)
            value = new_node.value
            # parent node add children
            leaf_node.children[leaf_action] = new_node
            return -value
        else:
            # delete this child's info
            del leaf_node.children[leaf_action]
            # re-dump board state for its illegal record,
            # it can reduce some illegal children in endgame point expand
            leaf_node.state = board.to_state()
            #print(leaf_action, "is illegal")
        return

    def batch_add_node(self, batch_list):
        #print("batch_add_node")
        # batch-y get value
        batched_mask = np.empty((len(batch_list), self.action_size), dtype=bool)
        batched_board_grid = np.empty((len(batch_list), self.size, self.size, 2))
        for i, binfo in enumerate(batch_list):
            board = board_from_state(binfo.state)
            batched_mask[i] = get_valid_mask(board, self.size)
            batched_board_grid[i] = board.grid[:,:,[1,0]] if board.next == WHITE else board.grid

        batched_logit = self.model.actor.predict_on_batch(batched_board_grid) # shape = (batch_size, action_size)
        batched_value = self.model.critic.predict_on_batch(batched_board_grid) # shape = (batch_size, 1)
        batched_noise = np.random.dirichlet(alpha=self.dirichlet_alpha, size=(len(batch_list)))
        batched_noised_logit = 0.75 * batched_logit + 0.25 * batched_noise
        masked_intuitions = masked_softmax(batched_mask, batched_noised_logit)

        for i, binfo in enumerate(batch_list):
            binfo.value = -batched_value[i][0] # binfo value is to update parent so negtived because board is viewed from enemy
            # create new node
            children_actions = batched_mask[i].nonzero()[0]
            #children_actions = fast_rand_int_sample(size=self.degree, p=masked_intuitions)
            new_node = MonteCarloNode(state=binfo.state,
                                    policies=masked_intuitions[i],
                                    value=batched_value[i][0], # new node value is view from "next"
                                    children_actions=children_actions)
            # update parent's node
            binfo.node_path[-1].children[binfo.action_path[-1]] = new_node
    # end def batch_add_node

    def add_node(self, board):
        # model generated data re-use
        #print("adding node id", zhash)
        masked_intuitions = self.model.get_masked_intuitions(board, 1.0)
        value = self.model.get_value(board.next, board.grid)
        children_actions = masked_intuitions.nonzero()[0]
        return MonteCarloNode(state=board.to_state(),
                            policies=masked_intuitions,
                            value=value,
                            children_actions=children_actions)
    # end def add_node

    def backpropagate(self, node_path, action_path, value):
        #print("bp with value:", value)
        for rev_i, a in reversed(list(enumerate(action_path))):
            curnode = node_path[rev_i]
            Qa = curnode.Q[a]
            Na = curnode.N[a]
            curnode.Q[a] = (value + Na * Qa) / (Na + 1)
            curnode.sumN += 1
            curnode.N[a] += 1
            curnode.expanding[a] = False
            curnode.visiting[a] -= 1
            value = -value # to switch side

    def batch_backpropagate(self, batch_list):
        #print("batch bp with value:")
        for binfo in batch_list:
            node_path, action_path, value = binfo.node_path, binfo.action_path, binfo.value
            #print(value)
            for rev_i, a in reversed(list(enumerate(action_path))):
                curnode = node_path[rev_i]
                Qa = curnode.Q[a]
                Na = curnode.N[a]
                curnode.Q[a] = (value + Na * Qa) / (Na + 1)
                curnode.sumN += 1
                curnode.N[a] += 1
                curnode.expanding[a] = False
                curnode.visiting[a] -= 1
                value = -value # to switch side
