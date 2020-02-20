import numpy as np
import go
from copy import copy, deepcopy
from keras import backend as K
from keras import optimizers, losses
from keras.models import Model, load_model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GAMMA = 0.9

STEP_QUEUE_MAX_LENGTH = 200

TRAIN_EPOCHS = 4
TRAIN_EPOCHS_THRESHOLD = 20
BATCH_SIZE = 8

A_LR = 5e-4
A_LR_DECAY = 1e-6
C_LR = 5e-4
C_LR_DECAY = 1e-6

class StepRecord() :
    def __init__(self) :
        self.actions = []
        self.old_states = []
        self.new_states = []
        self.rewards = []
        self.is_terminals = []
    
    def add(self, action, old_states, new_states, reward, is_terminal) :
        self.actions.append(action)
        self.old_states.append(old_states)
        self.new_states.append(new_states)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
    
    def clear(self) :
        self.actions = [] 
        self.states = []
        self.rewards = []
        self.is_terminals = []
    
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
        return np.array(self.actions[beg:to]), np.array(self.old_states[beg:to]), np.array(self.new_states[beg:to]), np.array(self.rewards[beg:to]), self.is_terminals
        

class ActorCritic :
    def __init__ (self, size, load_model_file) :    
        self.actor = None
        self.critic = None
        self.step_records = [StepRecord()]
        self.size = size
        
        if load_model_file:
            self.actor = load_model("actor_" + load_model_file)
            self.critic = load_model("critic_" + load_model_file)
        else :
            self.init_models(size)
            
        self.a_optimizer = optimizers.rmsprop(lr = A_LR, decay = A_LR_DECAY)
        self.actor.compile(loss = "mse", optimizer = self.a_optimizer)

        self.c_optimizer = optimizers.rmsprop(lr = C_LR, decay = C_LR_DECAY)
        self.critic.compile(loss = "mse", optimizer = self.c_optimizer)
    
    def init_models(self, size):
        input_map = Input((size, size, 2))
        x1 = Conv2D(32, (size//3, size//3), strides=(2, 2), padding="valid", activation="relu")(input_map)
        x1 = Conv2D(64, (size//9, size//9), activation="relu")(x1)
        
        x2 = Conv2D(32, (size//2, size//2), padding="valid", activation="relu")(input_map)
        x2 = Conv2D(64, (size//4, size//4), activation="relu")(x2)
        
        x3 = Conv2D(32, ((size*2)//3, (size*2)//3), padding="valid", activation="relu")(input_map)
        x3 = Conv2D(64, ((size*4)//9, (size*4)//9), activation="relu")(x3)
        
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x3 = Flatten()(x3)
        #x = Concatenate()([x1, x2])
        x = Concatenate()([x1, x2, x3])
        x = Dropout(0.2)(x)
        
        shared = Dense(int(2*size**2), activation = "relu")(x)
        shared = Dropout(0.2)(shared)
        
        logits = Dense(size**2)(shared)
        self.actor = Model(input_map, logits)

        values = Dense(size**2)(shared)
        self.critic = Model(input_map, values)
        self.critic.summary()
    
    def softmax(self, x, temperature=1.0):
        x = x.astype("float64")
        if len(x.shape) != 1:
            print("softmax input must be 1-D numpy array")
            return
        return np.exp(x/temperature)/np.sum(np.exp(x/temperature))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def decide_random(self, board, temperature=1.0):
        playas = board.next
        # transpose so that [x, y] become [x+y*size]
        illegal = np.transpose(board.illegal[:, :, 0 if playas==BLACK else 1]).flatten()
        has_stone = np.transpose(np.max(board.map, axis=2)==1).flatten()
        unplaceable = np.logical_or(illegal, has_stone)
        if np.where(unplaceable)[0].shape[0] == board.size**2:
            return -1, -1, 0.0
        
        logits = np.squeeze(self.actor.predict(np.expand_dims(board.map, axis=0)))
        
        # white's goal is to make value low
        if playas == WHITE:
            logits = -logits
        # just a big negtive number
        logits[unplaceable] = -10e6

        # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
        # sometime the sum exceed 1.0 due to numerical rounding
        instinct = self.softmax(logits, temperature)
        act = np.argmax(np.random.multinomial(1, instinct, 1))
        return act%self.size, act//self.size, instinct[act]
    
    def decide_tree_search(self, board, temperature=0.01, depth=2, kth=4):
        playas = board.next
        # transpose so that [x, y] become [x+y*size]
        illegal = np.transpose(board.illegal[:, :, 0 if playas==BLACK else 1]).flatten()
        has_stone = np.transpose(np.max(board.map, axis=2)==1).flatten()
        unplaceable = np.logical_or(illegal, has_stone)
        if np.where(unplaceable)[0].shape[0] == self.size**2:
            return -1, -1, 0.0
        
        logits = np.squeeze(self.actor.predict(np.expand_dims(board.map, axis=0)))
        values = np.squeeze(self.critic.predict(np.expand_dims(board.map, axis=0)))
        
        # white's goal is to make value low
        if playas == WHITE:
            logits = -logits
            values = -values
        # just a big negtive number
        logits[unplaceable] = -10e6
        
        win_rate = self.sigmoid(values)
        instinct = self.softmax(logits, temperature)
        
        if depth<1:
            act = np.argmax(win_rate)
            #print("depth 0 find", act, "w/ win_rate", win_rate[act])
        else:
            #print("depth", depth)
            candidates = np.argpartition(instinct, -kth)[-kth:]
            searched_win_rate = win_rate[candidates]
            for idx, candidate in enumerate(candidates):
                board_cpy = deepcopy(board)
                add_stone = go.Stone(board_cpy, 
                                  (candidate%self.size, candidate//self.size),
                                  playas)
                _x, _y, _inst, _wr = self.decide_tree_search(board_cpy, temperature, depth-1, kth)
                searched_win_rate[idx] = _wr
                #print("searched for candidate", candidate, "find win_rate", _wr)
            act = candidates[np.argmax(searched_win_rate)]
        return act%self.size, act//self.size, instinct[act], win_rate[act]
    
    def get_winrates(self, boardmap):
        return self.sigmoid(self.critic.predict(np.expand_dims(boardmap, axis=0))[0])
        
    def get_instincts(self, boardmap):
        return self.softmax(self.actor.predict(np.expand_dims(boardmap, axis=0))[0])
    
    def record(self, point, old_map, new_map, reward, is_terminal):
        self.step_records[-1].add(point[0]+point[1]*self.size, old_map, new_map, reward, is_terminal)
    
    def add_record(self):
        if len(self.step_records) >= STEP_QUEUE_MAX_LENGTH:
            self.step_records = self.step_records[1:]
        self.step_records.append(StepRecord())
    
    def reward_normalization(self, x):
        return (x - x.mean()) / np.std(x)

    def learn(self, verbose=True):
        closs = 0; aloss = 0
        if len(self.step_records) < TRAIN_EPOCHS_THRESHOLD:
            return
        random_idxs = np.random.randint(len(self.step_records), size=TRAIN_EPOCHS)
        for idx in random_idxs:
            rec_a, rec_os, rec_ns, rec_r, rec_terminal = self.step_records[idx].get_arrays()
            rec_r_norm = self.reward_normalization(rec_r)
            train_length = self.step_records[idx].length
            
            #new_r = np.zeros((train_length, 1))
            #new_a = np.zeros((train_length, self.size))
            advantages = np.zeros((train_length, self.size**2))

            new_r = self.critic.predict(rec_ns)
            old_r = self.critic.predict(rec_os)
            for i in range(train_length):
                if rec_terminal[i]:
                    new_r[i, rec_a[i]] = rec_r[i]
                else:
                    new_r[i, rec_a[i]] = rec_r[i] + new_r[i, rec_a[i]] * GAMMA
                advantages[i, rec_a[i]] = new_r[i, rec_a[i]] - old_r[i, rec_a[i]]
            #print("advantages:", advantages)
            new_a = self.actor.predict(rec_os) + advantages 
            closs += self.critic.fit(rec_os, advantages, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
            aloss += self.actor.fit(rec_os, new_a, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
        if verbose:
            print("avg_c_loss: %e, avg_a_loss: %e"%(closs/TRAIN_EPOCHS, aloss/TRAIN_EPOCHS))
        
    def save(self, save_weight_name) :
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)

