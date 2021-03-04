import threading
import pickle
import go
from time import time
import numpy as np

WHITE = go.WHITE
BLACK = go.BLACK

def fast_rand_int_sample(size, p):
    """
    This function has problem: it samples 0 weighted 
        a: int. The sample would be np.arange(a)
        size: sample size
        p: 1-D array of weight
    """
    cumsum = np.cumsum(p)
    rand_uniform = np.random.random(size)
    return np.searchsorted(cumsum, rand_uniform)

class BatchInfo():
    def __init__ (self, node_path, action_path, state, id):
        self.node_path = node_path
        self.action_path = action_path
        self.state = state
        self.id = id
        self.value = None

class MonteCarloNode():
    def __init__ (self, parent_id, id, state, policies, value, children_keys):
        self.parent_id = parent_id
        self.id = id
        self.state = state
        self.policies = policies
        self.value = value
        self.Q = {} # expected value of child from this node's perspective
        self.N = {} # number of times that this node take path to its child
        self.marked = {} # could be marked in batching and/or threading... set to False in back propagate phase
        self.children = {k : -1 for k in children_keys}  # action_id : node_id

class MonteCarlo():
    def __init__ (self, model, batch_size, thread_num):
        self.model = model
        self.visited = {}
        self.playout_number = 0 # reset every time search method called
        self.batch_size = batch_size
        self.thread_num = thread_num
        print("Monte Carlo parameters: Batch size:", batch_size, "Tread Number", thread_num)

    def clear(self):
        self.visited = {}
        self.prev_visited = {}

    def threaded_search(self, root_board, prev_action, degree, playout):
        self.record_time = []
        self.prev_visited = self.visited
        self.visited = {}
        self.degree = degree
        self.playout_number = playout
        # add root
        self.root_node = self.add_node(None, root_board)
        self.prev_action = prev_action
        self.selectlock = threading.Lock()
        self.backpropagatelock = threading.Lock()
        thrd_list = []
        # begin playout
        for i in range(self.thread_num):
            thrd_list.append(threading.Thread(target=self.threaded_search_playout))
        for i in range(self.thread_num):
            thrd_list[i].setDaemon(True)
            thrd_list[i].start()
        for i in range(self.thread_num):
            thrd_list[i].join()
        #print("playout time", np.sum(self.record_time), np.mean(self.record_time), np.std(self.record_time))
        return list(self.root_node.Q.keys()), list(self.root_node.Q.values())

    def threaded_search_playout(self):
        for _ in range(self.playout_number//self.batch_size):
            self.batch_playout()
            if len(self.root_node.children) <= 1:
                break

    def search(self, root_board, prev_action, degree, playout):
        self.record_time = []
        self.prev_visited = self.visited
        self.visited = {}
        self.degree = degree
        self.playout_number = playout
        # add root
        self.root_node = self.add_node(None, root_board)
        self.prev_action = prev_action
        self.selectlock = None
        self.backpropagatelock = None
        # begin playout
        if self.batch_size > 1:
            for _ in range(playout//self.batch_size):
                #t = time()
                self.batch_playout()
                #self.record_time.append(time()-t)
                if len(self.root_node.children) <= 1:
                    break
        else:
            for _ in range(playout):
                self.playout()
                if len(self.root_node.children) <= 1:
                    break
        # ...or find best root path
        """ 
        action_path = []
        cur_node = self.root_node
        while True:
            a = max(self.cur_node.Q, key=self.cur_node.Q.get)
            action_path.append(a)
            next_id = cur_node.children[a]
            if next_id != -1:
                cur_node = self.visited[next_id]
            else:
                break
        best_a = path[0]
        max_Q = self.root_node.Q[best_a]
        # print(action_path, max_Q))        
        """
        #print("playout time", np.sum(self.record_time), np.mean(self.record_time), np.std(self.record_time))
        return list(self.root_node.Q.keys()), list(self.root_node.Q.values())

    def playout(self):
        node_path, action_path, is_terminal = self.select()
        leaf_node = node_path[-1]
        leaf_action = action_path[-1]
        value = self.expand(leaf_node, leaf_action, is_terminal)
        if value:
            self.backpropagate(node_path, action_path, value)

    # batch_size larger then degree/2 is not recommended
    # keep it small like around 4
    def batch_playout(self):
        batch_list = []
        try_time = self.batch_size + 1
        while len(batch_list) < self.batch_size and try_time > 0:
            try_time -= 1
            if len(self.root_node.children) == 0: # no need to search any more
                return
            node_path, action_path, is_terminal = self.select(batching = True)
            if len(action_path) == 0: # no path available to choose
                break
            batchinfo, value = self.delayed_expand(node_path, action_path, is_terminal)
            if batchinfo is not None: # if expand is legal
                batch_list.append(batchinfo)
            elif value is not None: # if this action path leads to visited or terminal
                self.backpropagate(node_path, action_path, value)
            # else: illegal action: do nothing
        if len(batch_list) > 0:
            self.batch_add_node(batch_list)
        for binfo in batch_list:
            self.backpropagate(binfo.node_path, binfo.action_path, binfo.value)

    def batch_add_node(self, batch_list):
        #print("batch_add_node")
        # batch-y get value
        batched_mask = []
        batched_board_grid = []
        for i, binfo in enumerate(batch_list):
            board = pickle.loads(binfo.state)
            batched_mask.append(self.model.get_invalid_mask(board))
            boardgrid = board.grid.copy()
            if board.next == WHITE:
                boardgrid[:,:,[0,1]] = boardgrid[:,:,[1,0]]
            batched_board_grid.append(boardgrid)
        
        batched_mask = np.array(batched_mask)
        batched_board_grid = np.array(batched_board_grid)
        batched_logit = self.model.actor.predict(batched_board_grid) # shape = (batch_size, action_size)
        alpha = 10 / self.model.size_square
        batched_noise = np.random.default_rng().dirichlet(alpha=[alpha]*self.model.action_size, size=(len(batch_list)))
        batched_noised_logit = 0.8 * batched_logit + 0.2 * batched_noise
        batched_value = self.model.critic.predict(batched_board_grid) # shape = (batch_size, 1)

        for i, binfo in enumerate(batch_list):
            binfo.value = batched_value[i][0]
            parent_node = binfo.node_path[-1]
            leaf_action = binfo.action_path[-1]
            # create new node
            masked_intuitions = self.model.masked_softmax(batched_mask[i], batched_noised_logit[i], 1.0)
            # children_actions = np.random.default_rng().choice(self.model.action_size,
            #                             size=self.degree,
            #                             p=masked_intuitions)
            children_actions = fast_rand_int_sample(size=self.degree, p=masked_intuitions)
            self.visited[binfo.id] = MonteCarloNode(parent_id=binfo.node_path[-1],
                                            id=binfo.id,
                                            state=pickle.dumps(board),
                                            policies=masked_intuitions,
                                            value=binfo.value,
                                            children_keys=children_actions)
            # update parent's node
            parent_node.children[leaf_action] = binfo.id
    # end def batch_add_node

    def node_reuse(self, parent_id, zhash):
        # children_actions = np.random.choice(self.model.action_size,
        #                         size=self.degree,
        #                         p=self.visited[zhash].policies)
        children_actions = fast_rand_int_sample(size=self.degree, p=self.prev_visited[zhash].policies)
        self.visited[zhash] = MonteCarloNode(parent_id=parent_id,
                                            id=zhash,
                                            state=self.prev_visited[zhash].state,
                                            policies=self.prev_visited[zhash].policies,
                                            value=self.prev_visited[zhash].value,
                                            children_keys=children_actions) 

    def add_node(self, parent_id, board):
        zhash = board.grid_hash()
        # model generated data re-use
        if zhash in self.prev_visited:
            self.node_reuse(parent_id, zhash)
        else:
            #print("adding node id", zhash)
            intuitions = self.model.get_masked_intuitions(board, 1.0)
            value = self.model.get_value(board.next, board.grid)
            children_actions = np.random.choice(self.model.action_size, size=self.degree, p=intuitions)
            self.visited[zhash] = MonteCarloNode(parent_id=parent_id,
                                            id=zhash,
                                            state=pickle.dumps(board),
                                            policies=intuitions,
                                            value=value,
                                            children_keys=children_actions)
        return self.visited[zhash]
    # end def add_node

    def select(self, batching = False):
        #print("selecting nodes")
        cur_node = self.root_node
        node_path = []
        action_path = []
        is_terminal = False
        fuse = self.playout_number
        while fuse > 0:
            fuse -= 1
            #print(cur_node.id)

            best_a, best_u = -1, -float("inf")
            # u is upper confience bound of v
            # it is used for exploration-exploitation balance
            # formula taken from AlphaGoZero
            # c is coefficent, it controls the exploration rate
            # 1.4 is recommend in [0, 1] valued environment
            # 1.5~3.0 are used in [-1, 1] valued environment
            N_sum = sum([cur_node.N[k] for k in cur_node.N])
            if self.selectlock:
                self.selectlock.acquire()
            for a in cur_node.children:
                # loop that not caused by pass is not allowed
                if a != self.model.size_square and any([cur_node.children[a] == n.id for n in node_path]):
                    continue
                Qa = cur_node.Q.get(a, 0)
                Na = cur_node.N.get(a, 0)
                u = Qa + 1.5 * cur_node.policies[a] * np.sqrt(N_sum) / (1 + Na)
                # discourage marked node
                if cur_node.marked.get(a):
                    u -= 1
                #print(zhash, a)
                if u > best_u:
                    best_a, best_u = a, u
            # end for children actions
            if self.selectlock:
                cur_node.marked[best_a] = True
                self.selectlock.release()
            # if no valid children
            # although cur_node is visited... in some sense this is a terminal node
            if best_a == -1:
                break
            node_path.append(cur_node)
            action_path.append(best_a)
            # check two consecutive pass
            prev_action = action_path[-2] if len(action_path) > 1 else self.prev_action
            if best_a == prev_action and best_a == self.model.size_square:
                is_terminal = True
                break
            # check if not visited
            if cur_node.children[best_a] == -1: 
                if batching:
                    # mark this not visited children for batching and threading
                    # this mark will be erased in batch_add_node or add_node
                    cur_node.marked[best_a] = True
                    #print("batching action", best_a, "from", cur_node.id)
                break
            else:
                cur_node = self.visited[cur_node.children[best_a]]
        # traverse to an unexpanded node
        #print("selected path:", action_path)
        return node_path, action_path, is_terminal

    def delayed_expand(self, node_path, action_path, is_terminal):
        leaf_node = node_path[-1]
        leaf_action = action_path[-1]
        board = pickle.loads(leaf_node.state)
        zhash = board.grid_hash()

        if is_terminal:
            board.pass_move()
            winner, b, w = board.score()    
            # node is valued as board.next player
            value = 1 if board.next == winner else -1
            #print("terminal action value", value)
            return None, -value

        # update game board
        islegal = True
        if leaf_action >= self.model.size_square: # is pass
            board.pass_move()
        else:
            x = leaf_action % self.model.size
            y = leaf_action // self.model.size
            add_stone = go.Stone(board, (x, y))
            islegal = add_stone.islegal

        if islegal:
            # no parent node update... that's done in batch_add_node
            new_zhash = board.grid_hash()
            # if it's a loop... but not terminal. that is handle up there
            # in this case, it needs not to be batched
            # so update parent and return value
            if new_zhash in self.visited:
                #print(new_zhash, "is visited")
                if len(self.visited[new_zhash].Q) > 0:
                    value = max(self.visited[new_zhash].Q.values())
                else:
                    value = self.visited[new_zhash].value
                leaf_node.children[leaf_action] = new_zhash
                return None, -value
            # now this is conpletly new node
            else    :
                # we can still use node_reuse to speed up
                if new_zhash in self.prev_visited:
                    self.node_reuse(zhash, new_zhash)
                    return None, -self.visited[new_zhash].value
                else:
                    return BatchInfo(node_path=node_path, action_path=action_path, 
                            state=pickle.dumps(board), id=new_zhash), None
        else:
            # because it is illegal, mark leaf_action from leave node
            # return None so that it would go to back propergate and unmark
            leaf_node.marked[leaf_action] = True
            # re-dump board state for its illegal record,
            # it can reduce some illegal children in endgame point expand
            leaf_node.state = pickle.dumps(board)
            #print(leaf_action, "is illegal")
        return None, None
        

    def expand(self, leaf_node, leaf_action, is_terminal):
        board = pickle.loads(leaf_node.state)
        zhash = board.grid_hash()

        if is_terminal:
            board.pass_move()
            winner, b, w, output = board.score()
            # node is valued as board.next player
            value = 1 if board.next == winner else -1
            #print("terminal action value", value)
            leaf_node.Q[leaf_action] = -value
            leaf_node.N[leaf_action] = 1
            return -value

        # update game board
        islegal = True
        if leaf_action >= self.model.size_square: # is pass
            board.pass_move()
        else:
            x = leaf_action % self.model.size
            y = leaf_action // self.model.size
            add_stone = go.Stone(board, (x, y))
            islegal = add_stone.islegal
        
        # add node
        if islegal:
            new_zhash = board.grid_hash()
            # if it's a loop... but not terminal. that is handle up there
            if new_zhash in self.visited:
                if len(self.visited[new_zhash].Q) > 0:
                    value = max(self.visited[new_zhash].Q.values())
                else:
                    value = self.visited[new_zhash].value
            else:
                new_node = self.add_node(zhash, board)
                value = new_node.value
            leaf_node.children[leaf_action] = new_zhash
            return -value
        else:
            # because it is illegal, mark leaf_action from leave node
            # return None so that it would go to back propergate and unmark
            leaf_node.marked[leaf_action] = True
            # re-dump board state for its illegal record,
            # it can reduce some illegal children in endgame point expand
            leaf_node.state = pickle.dumps(board)
            #print(leaf_action, "is illegal")
        return

    def backpropagate(self, node_path, action_path, value):
        #print("bp with value:", value)
        if self.backpropagatelock:
            self.backpropagatelock.acquire()
        for i in reversed(range(len(node_path))):
            a = action_path[i]
            if a not in node_path[i].N:
                node_path[i].N[a] = 0
            if a not in node_path[i].Q:
                node_path[i].Q[a] = 0
            Na = node_path[i].N[a]
            node_path[i].Q[a] = (value + Na * node_path[i].Q[a]) / (Na + 1)
            node_path[i].N[a] += 1
            node_path[i].marked[a] = False
            value = -value # to switch side
        if self.backpropagatelock:
            self.backpropagatelock.release()
