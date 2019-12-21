import numpy as np
from keras import backend as K
from keras import optimizers
from keras.models import Model, load_model
#from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, LeakyReLU, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input

GAMMA = 0.996

STEP_QUEUE_MAX_LENGTH = 10

TRAIN_EPOCHS = 2
TRAIN_EPOCHS_THRESHOLD = 2
BATCH_SIZE = 16

A_LEARNING_RATE = 10e-4
A_LEARNING_RATE_DECAY = 0.0
C_LEARNING_RATE = 10e-4
C_LEARNING_RATE_DECAY = 0.0

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
    def __init__ (self, load_model_name="") :    
        self.actor = None
        self.critic = None
        self.step_records = [StepRecord()]
        
        if load_model_name:
            self.actor = load_model("actor_" + load_model_name)
            self.critic = load_model("critic_" + load_model_name)
        else :
            self.actor = self.init_actor()
            self.critic = self.init_critic()
            
        self.actor_optimizer = optimizers.rmsprop(lr = A_LEARNING_RATE, decay = A_LEARNING_RATE_DECAY)
        self.actor.compile(loss = "categorical_crossentropy", optimizer = self.actor_optimizer)

        self.critic_optimizer = optimizers.rmsprop(lr = C_LEARNING_RATE, decay = C_LEARNING_RATE_DECAY)
        self.critic.compile(loss = "mse", optimizer = self.critic_optimizer)
        
    def init_actor(self) :
        input_map = Input((19, 19, 2))
        x1 = Conv2D(16, (7, 7), strides=(2, 2), padding = "valid", activation = "relu")(input_map)
        x1 = Conv2D(32, (3, 3), activation = "relu")(x1)
        
        x2 = Conv2D(12, (11, 11), strides=(2, 2), padding = "valid", activation = "relu")(input_map)
        x2 = Conv2D(24, (3, 3), activation = "relu")(x2)
        
        x3 = Conv2D(8, (15, 15), strides=(2, 2), padding = "valid", activation = "relu")(input_map)
        x3 = Conv2D(16, (3, 3), activation = "relu")(x3)
        
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x3 = Flatten()(x3)
        #x = Concatenate()([x1, x2])
        x = Concatenate()([x1, x2, x3])
        probs = Dense(361, activation = "softmax")(x)
        model = Model(input_map, probs)
        model.summary()
        return model
    
    def init_critic(self) :
        input_map = Input((19, 19, 2))
        x1 = Conv2D(16, (7, 7), strides=(2, 2), padding = "valid", activation = "relu")(input_map)
        x1 = Conv2D(32, (3, 3), activation = "relu")(x1)
        
        x2 = Conv2D(12, (11, 11), strides=(2, 2), padding = "valid", activation = "relu")(input_map)
        x2 = Conv2D(24, (3, 3), activation = "relu")(x2)
        
        x3 = Conv2D(8, (15, 15), strides=(2, 2), padding = "valid", activation = "relu")(input_map)
        x3 = Conv2D(16, (3, 3), activation = "relu")(x3)
        
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x3 = Flatten()(x3)
        #x = Concatenate()([x1, x2])
        x = Concatenate()([x1, x2, x3])
        scores = Dense(361, activation = "sigmoid")(x)
        model = Model(input_map, scores)
        #model.summary()
        return model
        
    def decide(self, cur_map, temperature=1.0, test=False) :
        EPS = 10e-8
        preds = np.squeeze(self.actor.predict(np.expand_dims(cur_map, axis=0)))
        # delete illegal moves
        has_stone_id = np.argwhere(np.max(cur_map, axis=2) == 1)
        preds[has_stone_id] = 0
        if test:
            values = np.squeeze(self.critic.predict(np.expand_dims(cur_map, axis=0)))
            preds_values = values * preds
            preds_values[has_stone_id] = 0
            act = np.argmax(preds_values)
            return act%19, act//19, values[act]
        else:
            # softmax again
            exp_preds = np.exp((preds + EPS) / temperature)
            preds = exp_preds/np.sum(exp_preds)
            act = np.argmax(np.random.multinomial(1, preds, 1))
            return act%19, act//19, preds[act]
    
    def record(self, point, old_map, new_map, reward, is_terminal):
        self.step_records[-1].add(point[0]+point[1]*19, old_map, new_map, reward, is_terminal)
    
    def add_record(self):
        if len(self.step_records) >= STEP_QUEUE_MAX_LENGTH:
            self.step_records = self.step_records[1:]
        self.step_records.append(StepRecord())
    
    def learn(self):
        closs = 0; aloss = 0
        if len(self.step_records) < TRAIN_EPOCHS_THRESHOLD:
            return
        random_idxs = np.random.randint(len(self.step_records), size=TRAIN_EPOCHS)
        for idx in random_idxs:
            rec_a, rec_os, rec_ns, rec_r, rec_ister = self.step_records[idx].get_arrays()
            #print(rec_a.shape, rec_os.shape, rec_ns.shape, rec_r.shape)
            train_length = self.step_records[idx].length
            
            new_r = np.zeros((train_length, 361))
            advantages = np.zeros((train_length, 361))

            new_r = self.critic.predict(rec_ns)
            for i in reversed(range(train_length)):
                if rec_ister[i]:
                    new_r[i, rec_a[i]] = rec_r[i]
                else:
                    new_r[i, rec_a[i]] = rec_r[i] + new_r[i, rec_a[i]] * GAMMA
                
            old_r = self.critic.predict(rec_os)

            for i in range(train_length):
                advantages[i, rec_a[i]] = new_r[i, rec_a[i]] - old_r[i, rec_a[i]]
            #print("advantages:", advantages)
            closs += self.critic.fit(rec_os, new_r, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
            aloss += self.actor.fit(rec_os, advantages, batch_size=BATCH_SIZE, verbose=0).history["loss"][0]
        closs /= TRAIN_EPOCHS
        aloss /= TRAIN_EPOCHS
        print("avgcloss: %.3e, avgaloss: %.3e"%(closs, aloss))
        
    def save(self, save_weight_name) :
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)

    