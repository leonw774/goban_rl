import numpy as np
import go
from montecarlo import MonteCarlo
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input
# if there is some weird bug, try this
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

WHITE = go.WHITE
BLACK = go.BLACK

SHARE_HIDDEN_LAYER_NUM = 4
SHARE_HIDDEN_LAYER_CHANNEL = 64

A_LR = 1e-4
A_LR_DECAY = 0
C_LR = 1e-4
C_LR_DECAY = 0

class StepRecord():
    def __init__(self, player=None):
        self.actions = []
        self.old_state = []
        self.old_state_value = []
        self.new_state = []
        self.rewards = []
    
    def push(self, _action, _old_state, _new_state, _reward):
        self.actions.append(_action)
        self.old_state.append(_old_state)
        self.new_state.append(_new_state)
        self.rewards.append(_reward)

    def pop(self):
        self.actions.pop()
        self.old_state.pop()
        self.new_state.pop()
        self.rewards.pop()
    
    def clear(self, player=None):
        if player is not None: self.player = player
        self.actions = [] 
        self.old_state = []
        self.new_state = []
        self.rewards = []
    
    @property
    def length(self) :
        return len(self.actions)
    
    def get_arrays(self, beg=0, size=None) :
        if not beg:
            beg = 0
        if size:
            to = beg + size
        else:
            to = None
        return np.array(self.actions[beg:to]), np.array(self.old_state[beg:to]), np.array(self.new_state[beg:to]), np.array(self.rewards[beg:to])

"""
    Critic learn: (Q-learning)
        loss = (TDError)^2 (Mean Square Error)
        = (reward(s') + value(s') * GAMMA - value(s)) ^ 2
        s -> a -> s'
        
    Actor learn: (Policy Gradient)
        loss = - TDError * log(logits(a))
"""

def actor_loss(a_true, y_pred):
    # a_true.shape=(batch_size, action_size + 1)
    # a_true[:,0] = tderror; a_true[:,1:] = rec_a (one-hot-ed)
    log_pred = K.log(y_pred + K.epsilon())
    log_a = K.sum(a_true[:,1:] * log_pred, axis=1) # select action taken
    # log_a.shape=(batch_size, 1)
    basis_loss = -log_a * a_true[:,0]
    # basis_loss.shape=(batch_size, 1)
    entropy = K.mean(y_pred * log_pred, axis=1)
    # entropy.shape=(batch_size, 1)
    return basis_loss - 0.01 * entropy

""" def critic_loss(new_v, y_pred):
    # y_true is new_v
    return K.mean(K.square(new_v - y_pred), axis=-1) """

def switch_side(grid):
    switched_grid = grid.copy()
    switched_grid[:,:,[0,1]] = switched_grid[:,:,[1,0]]
    return switched_grid

class ActorCritic():
    def __init__ (self, size, step_records_length_max, load_model_file) :    
        self.actor = None
        self.critic = None
        self.step_records = [StepRecord()]
        self.size = size
        self.size_square = size**2
        self.action_size = size**2 + 1

        self.step_records_length_max = step_records_length_max
        self.minimax_hash_record = set()
        self.monte_carlo = MonteCarlo(self, batch_size=4, thread_num=12)
        
        if load_model_file:
            self.actor = load_model("actor_" + load_model_file, custom_objects={"actor_loss": actor_loss})
            self.critic = load_model("critic_" + load_model_file)
        else :
            self.init_models()
    
    def init_models(self):
        input_grid = Input((self.size, self.size, 2))
        conv = input_grid
        for _ in range(SHARE_HIDDEN_LAYER_NUM):
            conv = Conv2D(SHARE_HIDDEN_LAYER_CHANNEL, (3, 3), padding="same")(conv)
            conv = BatchNormalization()(conv)
            conv = Activation("relu")(conv)
        
        share = conv

        a = Conv2D(2, (1, 1))(share)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Flatten()(a)
        logits = Dense(self.action_size)(a)
        probs = Activation("softmax")(logits)
        #self.actor_logit = Model(input_grid, logits)
        self.actor = Model(input_grid, probs)


        c = Conv2D(2, (1, 1))(share)
        c = BatchNormalization()(c)
        c = Activation("relu")(c)
        c = Flatten()(c)
        c = Dense(256, activation = "relu")(c)
        #value = Dense(2, activation = "relu")(c)
        # this could be an auxiliry output
        # score_value = [0-indexed player's score, 1-indexed player's score]
        # score_values are positive
        # loss of score_value should be MSE
        #value = Dense(1, activation = "sigmoid")(c)
        value = Dense(1, activation = "tanh")(c)
        # in thoery this works better than score value
        # winrate_value = win rate of 0-indexed player
        # loss: if use sigmoid: binary_crossentropy
        #       if use tanh: MSE
        # use tanh as activation has one convinient thing that opponent's value is just negtive
        self.critic = Model(input_grid, value)
        self.actor.summary()

        self.a_optimizer = optimizers.RMSprop(lr = A_LR, decay = A_LR_DECAY)
        self.actor.compile(loss = actor_loss, optimizer = self.a_optimizer)

        self.c_optimizer = optimizers.RMSprop(lr = C_LR, decay = C_LR_DECAY)
        self.critic.compile(loss = "mse", optimizer = self.c_optimizer)

    def get_value(self, playas, boardgrid):
        if playas == WHITE:
            boardgrid = switch_side(boardgrid)
        return self.critic.predict(boardgrid[np.newaxis])[0,0]

    def get_logits(self, playas, boardgrid):
        if playas == WHITE:
            boardgrid = switch_side(boardgrid)
        return self.actor.predict(boardgrid[np.newaxis])[0]

    def masked_softmax(self, mask, x, temperature):
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
        normal_masked_x = masked_x - max(masked_x)
        masked_softmax_x = np.exp(normal_masked_x/temperature) / np.sum(np.exp(normal_masked_x/temperature))
        softmax = np.zeros(x.shape)
        softmax[mask_indice] = masked_softmax_x
        return softmax

    def get_invalid_mask(self, board):
        invalid_mask = np.zeros((self.size_square + 1), dtype=bool)
        # transpose so that [x, y] become [x+y*size]
        invalid_mask[:self.size_square] = np.transpose(np.sum(board.grid, axis=2)==1).flatten()
        for p in board.suicide_illegal.union(board.same_state_illegal):
            invalid_mask[p[0]+p[1]*self.size] = True
        if not np.any(invalid_mask): # empty
            invalid_mask[self.size_square] = True # can't pass
        return invalid_mask
    
    def get_masked_intuitions(self, board, temperature):
        invalid_mask = self.get_invalid_mask(board)
        logits = self.get_logits(board.next, board.grid)
        # Dirichlet's alpha = 10 / (averge legal move of every state of game)
        # in 19*19 it is 250 so alpha = 0.03
        # in 9*9 it maybe 60 so alpha = 0.1
        alpha = 10 / self.size_square
        noise = np.random.dirichlet(alpha=[alpha]*self.action_size)
        noised_logit = 0.8 * logits + 0.2 * noise
        masked_intuitions = self.masked_softmax(invalid_mask, noised_logit, temperature)
        #print(all(np.logical_or(np.logical_not(invalid_mask), (instinct < 1e-4)))) # if output False, something is wrong
        return masked_intuitions

    def decide_instinct(self, board, temperature):
        masked_intuitions = self.get_masked_intuitions(board, temperature)
        act = np.random.choice(self.action_size, 1, p=masked_intuitions)[0]
        return act%self.size, act//self.size, masked_intuitions[act]

    def decide_monte_carlo(self, board, playout=60, temperature=0.1):
        degree = self.size * 2
        prev_point = board.log[-1][1] if len(board.log) else (-1, 0)
        prev_action = prev_point[0] + prev_point[1] * self.size
        #candidates, values= self.monte_carlo.search(board, prev_action, degree, int(playout))
        candidates, values = self.monte_carlo.threaded_search(board, prev_action, degree, int(playout))
        # use value to softmax the candidates
        if len(candidates):
            values = np.array(values)
            if temperature > 0.01:
                weights = np.exp(values/temperature) / np.sum(np.exp(values/temperature))
                i = np.random.choice(len(candidates), p=weights)
                action = candidates[i]
            else:
                i = np.argmax(values)
                action = candidates[i]
            return action%self.size, action//self.size, values[i]
        else: # GG
            x, y, i = self.decide_instinct(board, 0.5)
            return x, y, i

    def decide_minimax(self, board, depth, kth):
        self.search_hash_record = set()
        action, value = self.minimax(board, depth, kth, -1e16, 1e16)
        return action%self.size, action//self.size, value

    '''    # Minimax search tree
    def minimax(self, board, depth, kth, alpha, beta):
        """
        alpha and beta are list [alpha], [beta] so that it can pass down as one object
        """
        if depth <= 0:
            # value should be returned as opponent's value
            return -1, -self.get_value(board.next, board.grid)

        #playas = board.next
        masked_intuitions = self.get_masked_intuitions(board, 0.4)
        best_v = -float("inf")
        best_a = -1
        for i in range(kth):
            a = np.random.choice(self.action_size, 1, p=masked_intuitions)[0]
            #print("depth", depth, "branch", i, "best_v", best_v, "alpha, beta", alpha, beta)
            islegal = True
            board_cpy = pickle.loads(pickle.dumps(board))
            if a >= self.size_square: # is pass
                board_cpy.pass_move()
            else:
                x = a % self.size
                y = a // self.size
                add_stone = go.Stone(board_cpy, (x, y))
                islegal = add_stone.islegal
            if islegal and (board_cpy.grid_hash() not in self.minimax_hash_record):
                a, v = self.minimax(board_cpy, depth-1, kth, -beta, -alpha)
                if best_v < v * (1 + (np.random.random() - 0.5) * 0.2):
                    best_v, best_a = v, a
                alpha = max(best_v, alpha)
                # pruning happen when max player best >= min player best
                if alpha >= beta:
                    #print("pruning at depth", depth, "branch", i, ":", alpha, beta)
                    break
            if len(self.search_hash_record) > 16:
                break
        #print("depth", depth, "find best action", best_a, "with value", best_v, ":", alpha, beta)
        # value should be returned as opponent's value
        return best_a, -best_v'''
    
    def pop_step(self):
        self.step_records[-1].pop()
    
    def push_step(self, point, old_grid, new_grid, reward):
        action = point[0] + point[1] * self.size
        action_one_hot = np.zeros((self.action_size), dtype="int32")
        action_one_hot[action] = 1
        # because new grid would be seen from opponent:
        new_grid = switch_side(new_grid)
        self.step_records[-1].push(action_one_hot, old_grid, new_grid, reward)

    def enqueue_new_record(self):
        self.monte_carlo.clear()
        if self.step_records[-1].length > 0:
            if len(self.step_records) >= self.step_records_length_max:
                self.step_records = self.step_records[1:]
            self.step_records.append(StepRecord())
    
    # def reward_normalization(self, x):
    #     return (x - x.mean()) / (np.std(x) + 1e-9)

    def learn(self, learn_record_size, verbose=True):
        """
            new_v: shape=(record_length, 1)
            [B, W, B, W, B, W], reward = win => [lose, win, lose, win, lose, win]
            [B, W, B, W, B], reward = win => [win, lose, win, lose, win]
            old_v: shape=(record_length, 1)
                 = value(t)
            a_true: shape=(record_length, action_size + 1)
                a_true[:,0] = tderror; a_true[:,1:] = rec_a
        """
        closs, aloss = 0, 0
        random_id = np.random.randint(len(self.step_records), size=learn_record_size)
        for id in random_id:
            rec_a, rec_os, rec_ns, rec_r = self.step_records[id].get_arrays()
            rec_len = self.step_records[id].length
            if rec_a.shape[0] == 0: continue
            #print(rec_r.shape)
            old_v = self.critic.predict(rec_os)
            new_v = np.repeat(rec_r[-1], rec_len)[:,np.newaxis]
            new_v[(rec_len+1)%2::2] = -new_v[(rec_len+1)%2::2]
            # shape = (record_length, 1)
            #print(new_v.shape, old_v.shape)
            tderror = new_v - old_v
            #print("rec_a:", rec_a.shape, "tderror:", tderror.shape)
            a_true = np.concatenate((tderror, rec_a), axis=1)
            #print("a_true:", a_true.shape)
            closs += self.critic.fit(rec_os, new_v, verbose=0).history["loss"][0]
            aloss += self.actor.fit(rec_os, a_true, verbose=0).history["loss"][0]
        if verbose:
            print("avg_c_loss: %.3f avg_a_loss: %.3f"%(closs/learn_record_size, aloss/learn_record_size))
        
    def save(self, save_weight_name):
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)
