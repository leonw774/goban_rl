import numpy as np
import go
from keras import backend as K
from keras import optimizers, losses
from keras.models import Model, load_model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GAMMA = 0.9982

STEP_QUEUE_MAX_LENGTH = 100

TRAIN_EPOCHS = 10
TRAIN_EPOCHS_THRESHOLD = 20
BATCH_SIZE = 8

A_LR = 10e-4
A_LR_DECAY = 10e-6
C_LR = 10e-4
C_LR_DECAY = 10e-6

def softmax(_input, _temperature=1.0):
    if len(_input.shape) != 1:
        print("softmax input must be 1-D numpy array")
        return
    return np.exp(_input/_temperature)/np.sum(np.exp(_input/_temperature))

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
        x1 = Conv2D(16, (size//3, size//3), strides=(2, 2), padding="valid", activation="relu")(input_map)
        x1 = Conv2D(32, (3, 3), activation="relu")(x1)
        
        x2 = Conv2D(16, ((size*2)//3, (size*2)//3), padding="valid", activation="relu")(input_map)
        x2 = Conv2D(32, (3, 3), activation="relu")(x2)
        
        x3 = Conv2D(32, (size, size), padding="valid", activation="relu")(input_map)
        
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x3 = Flatten()(x3)
        #x = Concatenate()([x1, x2])
        x = Concatenate()([x1, x2, x3])
        
        shared = Dense(2*size**2, activation = "relu")(x)
        logits = Dense(size**2)(shared)
        self.actor = Model(input_map, logits)

        values = Dense(size**2)(shared)
        self.critic = Model(input_map, values)
        self.critic.summary()
    
    def decide(self, board, temperature=1.0) :
        playas = board.next
        logits = np.squeeze(self.actor.predict(np.expand_dims(board.map, axis=0)))
        
        # transpose so that [x, y] become [x+y*size]
        illegal = np.transpose(board.illegal[:, :, 0 if playas==BLACK else 1]).flatten()
        has_stone = np.transpose(np.max(board.map, axis=2)==1).flatten()
        not_placeable = np.logical_or(illegal, has_stone)
        print(np.where(not_placeable)[0].shape[0])
        if np.where(not_placeable)[0].shape[0] == board.size**2:
            return -1, -1, 0.0
        
        # white's goal is to make value low
        if playas == WHITE:
            logits = -logits
        # just a big negtive number
        logits[not_placeable] = -10e6

        # astype("float64") because numpy's multinomial convert array to float64 after pval.sum()
        # sometime the sum exceed 1.0 due to numerical rounding
        probs = softmax(logits.astype("float64"), temperature)
        act = np.argmax(np.random.multinomial(1, probs, 1))
        return act%self.size, act//self.size, probs[act]
    
    def get_value(self, boardmap):
        return self.critic.predict(np.expand_dims(boardmap, axis=0))[0]
    
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
            
            new_a = self.actor.predict(rec_os)
            new_a = new_a + advantages 
            #print("advantages:", advantages)
            closs += self.critic.fit(rec_os, advantages, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
            aloss += self.actor.fit(rec_os, new_a, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
        if verbose:
            print("avgcloss: %e, avgaloss: %e"%(closs/TRAIN_EPOCHS, aloss/TRAIN_EPOCHS))
        
    def save(self, save_weight_name) :
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)

