import threading
from pickle import loads, dumps
from go import Stone, WHITE, BLACK
from time import time
import numpy as np

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
    def __init__ (self, node_path, action_path, state, id):
        self.node_path = node_path
        self.action_path = action_path
        self.state = state
        self.id = id
        self.value = None

class MonteCarloNode():
    def __init__ (self, state, policies, value, children_keys):
        self.state = state
        self.policies = policies
        self.value = value
        # Q, N are updated in back propergate phrase
        self.Q = {} # expected value of child from this node's perspective
        self.N = {k : 0 for k in children_keys} # number of times that this node take path to its child
        self.expanding = {k : False for k in children_keys} # True when selected to expand... set to False in back propagate phase
        self.visiting = {k : 0 for k in children_keys} # +1 when selected in path... -1 back propagate phase
        self.children = {k : None for k in children_keys}  # action_id : node
        self.nodelock = threading.Lock()

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
        self.visited = {} # node id : node instance
        self.playout_limit = 0 # reset every time search method called
        self.batch_size = batch_size
        self.thread_max = thread_max
        print("Parallel Monte Carlo parameters: Batch size:", batch_size, "thread max:", thread_max)

    def clear_visit(self):
        self.visited = {}

    def search(self, root_board, prev_action, degree, playout):
        self.record_time = []
        self.prev_visited = self.visited
        self.visited = {}
        self.degree = degree
        self.playout_limit = playout
        self.playout_count = 0
        # add root
        self.root_node = self.add_node(loads(dumps(root_board)))
        self.prev_action = prev_action
        self.globaltreelock = threading.Lock()
        self.playout_count_lock = threading.Lock()
        thrd_list = []
        if self.thread_max > 0:
            thread_num = min(self.thread_max, len(self.root_node.children))
        else:
            thread_num = len(self.root_node.children)
        # this is to compensate the critical section problem of playout_count
        self.playout_limit = int(self.playout_limit * (thread_num - 1) / thread_num)
        # begin playout
        for i in range(thread_num):
            thrd_list.append(threading.Thread(target=self.threaded_playout_loop))
        for i in range(thread_num):
            thrd_list[i].setDaemon(True)
            thrd_list[i].start()
        for i in range(thread_num):
            thrd_list[i].join()
        # find best root path
        """ 
        action_path = []
        curnode = self.root_node
        while curnode is not None:
            a = max(self.curnode.Q, key=self.curnode.Q.get)
            action_path.append(a)
            curnode = curnode.children[a]
        best_a = path[0]
        max_Q = self.root_node.Q[best_a]
        # print(action_path, max_Q))        
        """
        # print(self.playout_count)
        #print("playout time", np.sum(self.record_time), np.mean(self.record_time), np.std(self.record_time))
        return list(self.root_node.Q.keys()), list(self.root_node.Q.values())

    def threaded_playout_loop(self):
        #print("threaded_playout_loop")
        while self.playout_count < self.playout_limit:
            self.batch_playout()
            if len(self.root_node.children) <= 1:
                break

    def playout(self):
        node_path, action_path, is_terminal = self.select()
        leaf_node = node_path[-1]
        leaf_action = action_path[-1]
        value = self.expand(leaf_node, leaf_action, is_terminal)
        if value:
            self.backpropagate(node_path, action_path, value)

    def batch_playout(self):
        batch_list = []
        no_batch_count = 0
        while len(batch_list) < self.batch_size and no_batch_count < (self.batch_size // 8) + 1:
            node_path, action_path, is_terminal = self.select(batching = True)
            if len(action_path) == 0: # no path available to choose
                break
            batchinfo, value = self.delayed_expand(node_path, action_path, is_terminal)
            if batchinfo is not None: # if expand is legal
                batch_list.append(batchinfo)
            elif value is not None: # if this action path leads to visited or terminal
                self.backpropagate(node_path, action_path, value)
                no_batch_count += 1
            else: #illegal action: do nothing
                no_batch_count += 1

            #self.playout_count_lock.acquire()
            self.playout_count += 1 # SHOULD BE CRITICAL SECTION
            #self.playout_count_lock.release()
            # but for speed reason, we doesn't really need precise number of playout
            # it will end nonetheless

        if len(batch_list) > 0:
            self.batch_add_node(batch_list)
        self.backpropagate_with_batch(batch_list)

    def select(self, batching = False):
        #print("selecting nodes")
        curnode = self.root_node
        node_path = []
        action_path = []
        is_terminal = False
        #fuse = self.playout_limit
        #while fuse > 0:
        while True:
            #fuse -= 1
            best_a = -1
            curnode.nodelock.acquire()
            N_sum_sqrt = np.sqrt(sum(curnode.N.values()))
            """
                UCT is upper confience bound of v
                UCT[a] = Q[a] + P[a] where P is some formula for controlling exploration-exploitation balance
                P[a] = c * sqrt(sum(N)) / (1 + N[a])
                This formula is used by AlphaGoZero
                c is coefficent, it controls the exploration rate
                1.4 is recommend in [0, 1] valued environment
                1.5~3.0 are used in [-1, 1] valued environment
            """
            cur_filtered_uct_dict = {
                k : curnode.Q.get(k,0)+2*curnode.policies[k]*N_sum_sqrt/(1+curnode.N[k]) - curnode.visiting[k] * 0.1
                # virtual loss = UTC - visiting_count * (1 / degree)
                for k in curnode.children
                if not curnode.expanding[k] and (k == self.size_square or not curnode.children[k] in node_path)
            }
            curnode.nodelock.release()
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
                curnode.nodelock.acquire()
                curnode.expanding[best_a] = True
                curnode.nodelock.release()
                break
            else:
                curnode.nodelock.acquire()
                curnode.visiting[best_a] += 1
                curnode.nodelock.release()
                curnode = curnode.children[best_a]
        # traverse to an unexpanded node
        # print("selected path:", action_path)
        return node_path, action_path, is_terminal

    def delayed_expand(self, node_path, action_path, is_terminal):
        leaf_node = node_path[-1]
        leaf_action = action_path[-1]
        board = loads(leaf_node.state)

        if is_terminal:
            board.pass_move()
            winner, b, w = board.score()    
            # node is valued as board.next player
            value = 1 if board.next == winner else -1
            # print("terminal action value", value)
            return None, -value

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
                #self.globaltreelock.acquire()
                leaf_node.nodelock.acquire()
                leaf_node.children[leaf_action] = self.visited[new_zhash]
                #self.globaltreelock.release()
                leaf_node.nodelock.release()
                return None, -value
            # now this is conpletly new node
            # not add children to parent node... that's done in batch_add_node
            else:
                # we can still use node_reuse to speed up
                if new_zhash in self.prev_visited:
                    #print(new_zhash, "is in pre-visited")
                    self.node_reuse(new_zhash)
                    return None, -self.visited[new_zhash].value
                else:
                    # print(new_zhash, "is batching")
                    return BatchInfo(node_path=node_path, action_path=action_path, 
                            state=dumps(board), id=new_zhash), None
        else:
            #self.globaltreelock.acquire()
            leaf_node.nodelock.acquire()
            leaf_node.children.pop(leaf_action, None)
            #self.globaltreelock.release()
            leaf_node.nodelock.release()
            # print(leaf_action, "is illegal")
        # return None, None so that it won't go to backpropergate
        return None, None
        

    def expand(self, leaf_node, leaf_action, is_terminal):
        board = loads(leaf_node.state)
        if is_terminal:
            board.pass_move()
            winner, b, w = board.score()
            # node is valued as board.next player
            value = 1 if board.next == winner else -1
            #print("terminal action value", value)
            return -value

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
            new_zhash = board.grid_hash()
            # if it's a loop... but not terminal. that is handle up there
            if new_zhash in self.visited:
                if len(self.visited[new_zhash].Q) > 0:
                    value = max(self.visited[new_zhash].Q.values())
                else:
                    value = self.visited[new_zhash].value
            else:
                new_node = self.add_node(board)
                value = new_node.value
            # parent node add children
            #self.globaltreelock.acquire()
            leaf_node.nodelock.acquire()
            leaf_node.children[leaf_action] = self.visited[new_zhash]
            #self.globaltreelock.release()
            leaf_node.nodelock.release()
            return -value
        else:
            #self.globaltreelock.acquire()
            leaf_node.nodelock.acquire()
            leaf_node.children.pop(leaf_action, None)
            #self.globaltreelock.release()
            leaf_node.nodelock.release()
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
        with self.model.graph.as_default():
            with self.model.session.as_default():
                batched_logit = self.model.actor.predict_on_batch(batched_board_grid) # shape = (batch_size, action_size)
                batched_value = self.model.critic.predict_on_batch(batched_board_grid) # shape = (batch_size, 1)
        batched_noise = np.random.dirichlet(alpha=self.dirichlet_alpha, size=(len(batch_list)))
        batched_noised_logit = 0.8 * batched_logit + 0.2 * batched_noise

        for i, binfo in enumerate(batch_list):
            bid = binfo.id
            binfo.value = batched_value[i][0]
            # create new node
            masked_intuitions = masked_softmax(batched_mask[i], batched_noised_logit[i], 1.0)
            # children_actions = np.random.choice(self.model.action_size,
            #                             size=self.degree,
            #                             p=masked_intuitions)
            children_actions = fast_rand_int_sample(size=self.degree, p=masked_intuitions)
            new_node = MonteCarloNode(state=binfo.state,
                                            policies=masked_intuitions,
                                            value=batched_value[i][0],
                                            children_keys=children_actions)
            self.visited[bid] = new_node
            # update parent's node
            #self.globaltreelock.acquire()
            binfo.node_path[-1].nodelock.acquire()
            binfo.node_path[-1].children[binfo.action_path[-1]] = new_node
            #self.globaltreelock.release()
            binfo.node_path[-1].nodelock.release()
    # end def batch_add_node

    def node_reuse(self, zhash):
        # children_actions = np.random.choice(self.model.action_size,
        #                         size=self.degree,
        #                         p=self.prev_visited[zhash].policies)
        children_actions = fast_rand_int_sample(size=self.degree, p=self.prev_visited[zhash].policies)
        self.visited[zhash] = MonteCarloNode(state=self.prev_visited[zhash].state,
                                            policies=self.prev_visited[zhash].policies,
                                            value=self.prev_visited[zhash].value,
                                            children_keys=children_actions) 

    def add_node(self, board):
        zhash = board.grid_hash()
        # model generated data re-use
        if zhash in self.prev_visited:
            self.node_reuse(zhash)
        else:
            #print("adding node id", zhash)
            intuitions = self.model.get_masked_intuitions(board, 1.0)
            value = self.model.get_value(board.next, board.grid)
            children_actions = np.random.choice(self.action_size, size=self.degree, p=intuitions)
            self.visited[zhash] = MonteCarloNode(state=dumps(board),
                                            policies=intuitions,
                                            value=value,
                                            children_keys=children_actions)
        return self.visited[zhash]
    # end def add_node

    def backpropagate(self, node_path, action_path, value):
        # print("bp with value:", value)
        for rev_i, a in reversed(list(enumerate(action_path))):
            curnode = node_path[rev_i]
            curnode.nodelock.acquire()
            Qa = curnode.Q.get(a, 0)
            Na = curnode.N[a]
            curnode.Q[a] = (value + Na * Qa) / (Na + 1)
            curnode.N[a] += 1
            curnode.visiting[a] -= 1
            curnode.expanding[a] = False
            curnode.nodelock.release()
            value = -value # to switch side

    def backpropagate_with_batch(self, batch_list):
        # print("batch bp with value:")
        for binfo in batch_list:
            node_path, action_path, value = binfo.node_path, binfo.action_path, binfo.value
            #self.globaltreelock.acquire()
            for rev_i, a in reversed(list(enumerate(action_path))):
                curnode = node_path[rev_i]
                curnode.nodelock.acquire()
                Qa = curnode.Q.get(a, 0)
                Na = curnode.N[a]
                curnode.Q[a] = (value + Na * Qa) / (Na + 1)
                curnode.N[a] += 1
                curnode.visiting[a] -= 1
                curnode.expanding[a] = False
                curnode.nodelock.release()
                value = -value # to switch side
            #self.globaltreelock.release()
