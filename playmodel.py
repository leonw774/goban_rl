import numpy as np
from time import time
import go
from montecarlo import MonteCarlo
from parallelmontecarlo import ParallelMonteCarlo
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input
# if there is some weird bug, try this
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

WHITE = go.WHITE
BLACK = go.BLACK

SHARE_HIDDEN_LAYER_NUM = 2
SHARE_HIDDEN_LAYER_CHANNEL = 64

A_LR = 1e-4
A_LR_DECAY = 0
C_LR = 1e-4
C_LR_DECAY = 0

class StepRecord():
    def __init__(self, player=None):
        self.old_state = []
        self.new_policy = []
        self.game_result = 0
    
    def push(self, _old_state, _new_policy):
        self.old_state.append(_old_state)
        self.new_policy.append(_new_policy)

    def pop(self):
        self.old_state.pop()
        self.new_policy.pop()
    
    def clear(self, player=None):
        if player is not None: self.player = player
        self.old_state = []
        self.new_policy = []
        self.game_result = 0
    
    @property
    def length(self) :
        return len(self.old_state)
    
    def get_arrays(self) :
        return np.array(self.old_state), np.array(self.new_policy), self.game_result

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

def masked_softmax(mask, x, temperature):
    # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
    # sometime the sum exceed 1.0 due to numerical rounding
    x = x.astype("float64")
    # do not consider i if mask[i] == True
    masked_x = x[mask]
    # stablize/normalize because if x too big will cause NAN when exp(x)
    normal_masked_x = masked_x - max(masked_x)
    masked_softmax_x = np.exp(normal_masked_x/temperature) / np.sum(np.exp(normal_masked_x/temperature))
    softmax = np.zeros(x.shape)
    softmax[mask] = masked_softmax_x
    return softmax

def switch_side(grid):
    switched_grid = grid.copy()
    switched_grid[:,:,[0,1]] = switched_grid[:,:,[1,0]]
    return switched_grid

class ActorCritic():
    def __init__ (self, size, step_records_length_max, load_model_file) :    
        self.actor = None
        self.critic = None
        self.size = size
        self.size_square = size**2
        self.action_size = size**2 + 1

        self.step_records = [StepRecord()]
        self.step_records_length_max = step_records_length_max

        self.resign_value = -0.8 # resign if determined value is low than this value

        if load_model_file:
            self.actor = load_model("actor_" + load_model_file, custom_objects={"actor_loss": actor_loss})
            self.critic = load_model("critic_" + load_model_file)
        else :
            self.init_models()
        # initalize predict function ahead of possible threading in monte carlo 
        # to prevent multi initializtion
        # self.actor._make_predict_function()
        # self.critic._make_predict_function()
        # for fixing initialization problem in parallel monte carlo's threading scheme in Ubuntu
        #self.session = K.get_session()
        #self.graph = tf.compat.v1.get_default_graph()
        #self.graph.finalize()
        # batch_size too large will cause exploration too high
        self.monte_carlo = MonteCarlo(self, batch_size=8)
        # thread_max should not bigger than core number of the running machine, or it could slow down
        # self.monte_carlo = ParallelMonteCarlo(self, batch_size=4, thread_max=4)
    
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
        self.actor.compile(loss = "categorical_crossentropy", optimizer = self.a_optimizer)

        self.c_optimizer = optimizers.RMSprop(lr = C_LR, decay = C_LR_DECAY)
        self.critic.compile(loss = "mse", optimizer = self.c_optimizer)

    # predict_on_batch seem to be faster than predict if only one batch is feeded
    def get_value(self, playas, boardgrid):
        if playas == WHITE:
            boardgrid = switch_side(boardgrid)
        return self.critic.predict_on_batch(boardgrid[np.newaxis])[0,0]

    def get_logits(self, playas, boardgrid):
        if playas == WHITE:
            boardgrid = switch_side(boardgrid)
        return self.actor.predict_on_batch(boardgrid[np.newaxis])[0]

    def eval_to_value(self, board):
        b, w = board.eval()
        v = np.tanh((b - w) / (self.size_square - b - w))
        if board.next == WHITE: v = -v
        return v

    def get_valid_mask(self, board):
        valid_mask = np.zeros((self.action_size), dtype=bool)
        # transpose so that [x, y] become [x+y*size]
        valid_mask[:self.size_square] = np.transpose(np.sum(board.grid, axis=2)==0).flatten()
        for p in board.suicide_illegal.union(board.same_state_illegal):
            valid_mask[p[0]+p[1]*self.size] = False
        if not np.all(valid_mask): # not empty
            valid_mask[self.action_size-1] = True # can pass
        return valid_mask
    
    def get_masked_intuitions(self, board, temperature):
        invalid_mask = self.get_valid_mask(board)
        logits = self.get_logits(board.next, board.grid)
        # Dirichlet's alpha = 10 / (averge legal move of every state of game)
        # in 19*19 it is 250 so alpha = 0.03
        # in 9*9 it maybe 60 so alpha = 0.1
        alpha = 10 / self.size_square
        noise = np.random.dirichlet(alpha=[alpha]*self.action_size)
        noised_logit = 0.75 * logits + 0.25 * noise
        masked_intuitions = masked_softmax(invalid_mask, noised_logit, temperature)
        #print(all(np.logical_or(np.logical_not(invalid_mask), (instinct < 1e-4)))) # if output False, something is wrong
        return masked_intuitions

    def decide_instinct(self, board, temperature):
        masked_intuitions = self.get_masked_intuitions(board, temperature)
        act = np.random.choice(self.action_size, 1, p=masked_intuitions)[0]
        return act%self.size, act//self.size, masked_intuitions[act]

    def decide_monte_carlo(self, board, playout, temperature=0.1):
        prev_point = board.log[-1][1] if len(board.log) else (-1, 0)
        prev_action = prev_point[0] + prev_point[1] * self.size
        x, y, pi = self.monte_carlo.search(board, prev_action, int(playout), temperature)
        return x, y, pi
    
    def pop_step(self):
        self.step_records[-1].pop()
    
    def push_step(self, old_grid, mcts_policy):
        self.step_records[-1].push(old_grid, mcts_policy)

    def enqueue_new_record(self, reward):
        self.step_records[-1].game_result = reward
        if self.step_records[-1].length > 0:
            if len(self.step_records) >= self.step_records_length_max:
                self.step_records = self.step_records[1:]
            self.step_records.append(StepRecord())
    
    # def reward_normalization(self, x):
    #     return (x - x.mean()) / (np.std(x) + 1e-9)

    def learn(self, learn_record_size, verbose=True):
        closs, aloss = 0, 0
        random_id = np.random.randint(len(self.step_records), size=learn_record_size)
        for id in random_id:
            rec_s, rec_p, rec_r = self.step_records[id].get_arrays()
            rec_len = self.step_records[id].length
            # new_v: shape=(record_length, 1)
            # reward is viewed from BLACK
            # [B, W, B, W, B, W] & reward = win => [win, lose, win, lose, win, lose]
            new_v = np.repeat(rec_r, rec_len)[:,np.newaxis]
            new_v[1::2] = -new_v[1::2]
            closs += self.critic.fit(rec_s, new_v, batch_size=4, verbose=0).history["loss"][0]
            aloss += self.actor.fit(rec_s, rec_p, batch_size=4, verbose=0).history["loss"][0]
        if verbose:
            print("avg_c_loss: %.3f avg_a_loss: %.3f"%(closs/learn_record_size, aloss/learn_record_size))
        
    def save(self, save_weight_name):
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)
