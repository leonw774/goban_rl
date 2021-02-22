import numpy as np
import go
from copy import copy, deepcopy
from keras import backend as K
from keras import optimizers, losses
from keras.models import Model, load_model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, Reshape
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

WHITE = go.WHITE
BLACK = go.BLACK

GAMMA = 0.9

STEP_QUEUE_MAX_LENGTH = 40

TRAIN_RECORDS = 10
TRAIN_RECORDS_THRESHOLD = 20
BATCH_SIZE = 8

A_LR = 1e-4
A_LR_DECAY = 0
C_LR = 1e-3
C_LR_DECAY = 0

def actor_loss(a_true, y_pred):
    # a_true: shape=(action_size + 1)
    # a_true[:,0] = tderror; a_true[:,1:] = rec_a (one-hot-ed)
    action_weight = K.sum(a_true[:,1:] * y_pred, axis=1)
    return -(K.log(action_weight + K.epsilon())) * a_true[:,0]

def critic_loss(new_v, y_pred):
    # y_true is new_v
    return K.mean(K.square(new_v - y_pred), axis=-1)

class StepRecord():
    def __init__(self, player=None):
        self.player = player
        self.actions = []
        self.old_state = []
        self.old_state_value = []
        self.new_state = []
        self.rewards = []
    
    def push(self, _action, _old_state, _old_state_value, _new_state, _reward):
        self.actions.append(_action)
        self.old_state.append(_old_state)
        self.old_state_value.append(_old_state_value)
        self.new_state.append(_new_state)
        self.rewards.append(_reward)

    def pop(self):
        self.actions.pop()
        self.old_state.pop()
        self.old_state_value.pop()
        self.new_state.pop()
        self.rewards.pop()
    
    def clear(self, player=None):
        if player is not None: self.player = player
        self.actions = [] 
        self.old_state = []
        self.old_state_value = []
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
        return np.array(self.actions[beg:to]), np.array(self.old_state[beg:to]), np.array(self.old_state_value[beg:to]), np.array(self.new_state[beg:to]), np.array(self.rewards[beg:to])

class ActorCritic :
    def __init__ (self, size, load_model_file) :    
        self.w_actor = None
        self.b_actor = None
        self.critic = None
        self.step_records = [StepRecord()]
        self.size = size
        self.size_square = size**2
        self.action_size = size**2 + 1

        #self.action = K.placeholder(shape=(None, self.action_size))
        #self.advantage = K.placeholder(shape=(None,))
        
        if load_model_file:
            self.w_actor = load_model("w_actor_" + load_model_file, custom_objects={"actor_loss": actor_loss})
            self.b_actor = load_model("b_actor_" + load_model_file, custom_objects={"actor_loss": actor_loss})
            self.critic = load_model("critic_" + load_model_file, custom_objects={"critic_loss": critic_loss})
        else :
            self.init_models(size)
            
        self.a_optimizer = optimizers.adam(lr = A_LR, decay = A_LR_DECAY)
        self.w_actor.compile(loss = actor_loss, optimizer = self.a_optimizer)
        self.b_actor.compile(loss = actor_loss, optimizer = self.a_optimizer)

        self.c_optimizer = optimizers.adam(lr = C_LR, decay = C_LR_DECAY)
        self.critic.compile(loss = critic_loss, optimizer = self.c_optimizer)

    def init_models(self, size):
        input_grid = Input((size, size, 2))
        strides_size = (2, 2) if size > 13 else (1, 1)
        x1 = Conv2D(64, (size//2, size//2), strides=strides_size, padding="valid", activation="relu")(input_grid)        
        x2 = Conv2D(64, (size//3, size//3), strides=strides_size, padding="valid", activation="relu")(input_grid)
        
        x0 = Flatten()(input_grid)
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x = Concatenate()([x0, x1, x2])
        shared = Dropout(0.1)(x)
        
        #shared = Dense(int(8*size), activation = "relu")(x)
        #shared = Dropout(0.1)(shared)
        
        w_logits = Dense(self.action_size, activation="softmax")(shared)
        b_logits = Dense(self.action_size, activation="softmax")(shared)
        self.w_actor = Model(input_grid, w_logits)
        self.b_actor = Model(input_grid, b_logits)

        value = Dense(1)(shared)
        # value can be negtive
        self.critic = Model(input_grid, value)
        self.w_actor.summary()
    
    def get_value(self, boardgrid):
        return self.critic.predict(np.expand_dims(boardgrid, axis=0))[0,0]

    def get_instincts(self, playas, boardgrid):
        if playas == WHITE:
            return self.w_actor.predict(np.expand_dims(boardgrid, axis=0))[0]
        else:
            return self.b_actor.predict(np.expand_dims(boardgrid, axis=0))[0]

    def masked_softmax(self, mask, x):
        # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
        # sometime the sum exceed 1.0 due to numerical rounding
        x = x.astype("float64")
        # this normalize x
        #x = (x - x.mean()) / (np.std(x) + 1e-9)
        x[mask] = -1e16 # very large negtive number
        if len(x.shape) != 1:
            print("softmax input must be 1-D numpy array")
            return
        return np.exp(x)/np.sum(np.exp(x))

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))
    
    def get_masked_instincts(self, board):
        playas = board.next
        invalid_move = np.zeros((self.size_square + 1), dtype=bool)
        # transpose so that [x, y] become [x+y*size]
        invalid_move[:self.size_square] = np.transpose(np.sum(board.grid, axis=2)==1).flatten()
        for p in board.suicide_illegal.union(board.same_state_illegal):
            invalid_move[p[0]+p[1]*self.size] = True

        if playas == WHITE:
            logits = self.w_actor.predict(np.expand_dims(board.grid, axis=0))[0]
        else:
            logits = self.b_actor.predict(np.expand_dims(board.grid, axis=0))[0]
        
        masked_instincts = self.masked_softmax(invalid_move, logits)
        #print(all(np.logical_or(np.logical_not(invalid_move), (instinct < 1e-4)))) # if output False, something is wrong
        return masked_instincts

    def decide_instinct(self, board):
        masked_instincts = self.get_masked_instincts(board)
        value = self.critic.predict(np.expand_dims(board.grid, axis=0))[0,0]
        act = np.random.choice(self.size_square+1, 1, p=masked_instincts)[0]
        return act%self.size, act//self.size, masked_instincts[act], value

    def decide_tree(self, board, depth, kth):
        action, instinct, value = self.tree_search(board, depth, kth, [-1e16, 1e16])
        return action%self.size, action//self.size, instinct, value

    def tree_search(self, board, depth, kth, alphabeta):
        """
        alphabeta is a list [alpha, beta] so that it can pass down as one object
        """
        if depth <= 0:
            return -1, 0, self.get_value(board.grid)
        playas = board.next
        masked_instincts = self.get_masked_instincts(board)
        candidates = np.argpartition(masked_instincts, -kth)[-kth:]
        np.random.shuffle(candidates)
        best_v = -1e15 if playas == BLACK else 1e15
        best_a = -1
        for i, candidate in enumerate(candidates):
            #print("depth", depth, "branch", i, "best_v", best_v, "alphabeta", alphabeta)
            islegal = True
            board_cpy = deepcopy(board)
            if candidate >= self.size_square:
                # is pass
                board_cpy.pass_move()
            else:
                candidate_x = candidate%self.size
                candidate_y = candidate//self.size
                add_stone = go.Stone(board_cpy, (candidate_x, candidate_y))
                islegal = add_stone.islegal
            if islegal:
                a, inst, v = self.tree_search(board_cpy, depth-1, kth, alphabeta)
                if playas == BLACK:
                    if best_v < v * (np.random.random() * 0.5):
                        best_v = v
                        best_a = candidate
                    alphabeta[0] = max(best_v, alphabeta[0])
                else:
                    if best_v > v * (np.random.random() * 0.5):
                        best_v = v
                        best_a = candidate
                    alphabeta[1] = min(best_v, alphabeta[1])
                # pruning happen when max player best >= min player best
                if alphabeta[0] > alphabeta[1]:
                    #print("pruning at depth", depth, "branch", i, ":", alphabeta)
                    break
        #print("depth", depth, "find best action", best_a, "with value", best_v, ":", alphabeta)
        return best_a, masked_instincts[best_a], best_v
    
    def pop_step(self):
        self.step_records[-1].pop()
    
    def push_step(self, point, color, old_grid, old_value, new_grid, reward):
        action = point[0] + point[1] * self.size
        action_one_hot = np.zeros((self.action_size), dtype="int32")
        action_one_hot[action] = 1
        self.step_records[-1].push(action_one_hot, old_grid, old_value, new_grid, reward)
    
    def set_record_player(self, player):
        self.step_records[-1].player = player

    def enqueue_new_record(self):
        if self.step_records[-1].length > 0:
            if len(self.step_records) >= STEP_QUEUE_MAX_LENGTH:
                self.step_records = self.step_records[1:]
            self.step_records.append(StepRecord())
    
    # def reward_normalization(self, x):
    #     return (x - x.mean()) / (np.std(x) + 1e-9)

    def learn(self, verbose=True):
        """
            Critic learn: (Q-learning)
                loss = (TDError)^2 = (reward(s') + value(s') * GAMMA - value(s)) ^ 2
                s -> a -> s'

            Actor learn: (Policy Gradient)
                loss = - TDError * log(logits(a))
            
            new_v: shape=(train_length, 1)
                 = reward(t) + value(t+1) * GAMMA
            old_v: shape=(train_length, self.size)
                 = value(t)
            a_true: shape=(train_length, action_size + 1)
                a_true[:,0] = tderror; a_true[:,1:] = rec_a
        """
        closs = 0; aloss = 0
        if len(self.step_records) < TRAIN_RECORDS_THRESHOLD:
            return
        random_id = np.random.randint(len(self.step_records), size=TRAIN_RECORDS)
        for id in random_id:
            rec_p = self.step_records[id].player
            rec_a, rec_os, rec_ov, rec_ns, rec_r = self.step_records[id].get_arrays()
            if rec_p == None or rec_a.shape[0] == 0: continue
            #print(rec_ov.shape, rec_r.shape)
            new_v = np.squeeze(self.critic.predict(rec_ns))
            #old_v = np.squeeze(self.critic.predict(rec_os))
            new_v = rec_r + new_v * GAMMA
            new_v[-1] = rec_r[-1] # terminal
            #print("new_v:", new_v.shape)
            tderror = np.expand_dims(new_v - rec_ov, axis=1) # tderror.shape = (batch_length, 1)
            if rec_p == WHITE: # white is min_player so tderror turns negtive
                tderror = -tderror
            #print("rec_a:", rec_a.shape, "tderror:", tderror.shape)
            a_true = np.concatenate((tderror, rec_a), axis=1)
            # only update the recorded action
            closs += self.critic.fit(rec_os, new_v, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
            if rec_p == BLACK:
                aloss += self.b_actor.fit(rec_os, a_true, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
            else:
                aloss += self.w_actor.fit(rec_os, a_true, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
        if verbose:
            print("avg_c_loss: %e avg_a_loss: %e"%(closs/TRAIN_RECORDS, aloss/TRAIN_RECORDS))
        
    def save(self, save_weight_name):
        self.w_actor.save("w_actor_" + save_weight_name)
        self.b_actor.save("b_actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)
