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
# if there is some weird bug, try this:
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

WHITE = go.WHITE
BLACK = go.BLACK

class StepRecord():
    def __init__(self):
        """
            prev_states = list of prev_grid
            cur_states = list of cur_grid
        """
        self.prev_states = []
        self.cur_states = []
        self.actions = []
        self.game_result = None

    def push(self, _prev_states, _action, _cur_states):
        self.prev_states.append(_prev_states)
        self.actions.append(_action)
        self.cur_states.append(_cur_states)

    def pop(self):
        self.prev_states.pop()
        self.actions.pop()
        self.cur_states.pop()

    def clear(self):
        self.prev_states = []
        self.cur_states = []
        self.actions = []
        self.game_result = None

    @property
    def length(self) :
        return len(self.prev_states)

    def get_arrays(self) :
        return np.array(self.prev_states), np.array(self.actions),  np.array(self.cur_states), self.game_result

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
    def __init__ (self, size, load_model_file = None) :
        self.actor = None
        self.critic = None
        self.size = size
        self.size_square = size ** 2
        self.action_size = size ** 2 + 1

        self.SHARE_HIDDEN_LAYER_NUM = 1
        self.SHARE_HIDDEN_LAYER_CHANNEL = 32

        self.CRITIC_OUTLAYER_ACTIVATION_FUNC = "tanh"
        if self.CRITIC_OUTLAYER_ACTIVATION_FUNC == "tanh":
            self.WIN_REWARD = 1
            self.LOSE_REWARD = -1
        elif self.CRITIC_OUTLAYER_ACTIVATION_FUNC == "sigmoid":
            self.WIN_REWARD = 1
            self.LOSE_REWARD = 0

        self.GAMMA = 0.5
        self.LEARNING_RATE = 1e-3

        self.step_records = [StepRecord()]
        self.STEP_RECORD_AMOUNT_MAX = 4
        self.LEARN_RECORD_SIZE = 2

        self.resign_value = -0.8 # resign if determined value is low than this value

        if load_model_file:
            self.actor = load_model("actor_" + load_model_file)
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

        self.monte_carlo = MonteCarlo(self)
        # self.monte_carlo = ParallelMonteCarlo(self, batch_size = 4, thread_max = 4)

    def init_models(self):
        input_grid = Input((self.size, self.size, 2))
        input_playas = Input((1,))
        conv = input_grid
        for _ in range(self.SHARE_HIDDEN_LAYER_NUM):
            conv = Conv2D(self.SHARE_HIDDEN_LAYER_CHANNEL, (3, 3), padding="same")(conv)
            conv = BatchNormalization()(conv)
            conv = Activation("relu")(conv)

        share = conv

        a = Conv2D(2, (1, 1))(share)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Flatten()(a)
        a = Concatenate()([a, input_playas])
        logits = Dense(self.action_size)(a)
        policy = Activation("softmax")(logits)
        self.actor = Model([input_grid, input_playas], policy)

        c = Conv2D(2, (1, 1))(share)
        c = BatchNormalization()(c)
        c = Activation("relu")(c)
        c = Flatten()(c)
        a = Concatenate()([a, input_playas])
        c = Dense(256, activation = "relu")(c)

        if self.CRITIC_OUTLAYER_ACTIVATION_FUNC == "sigmoid":
            value = Dense(1, activation = "sigmoid")(c)
        elif self.CRITIC_OUTLAYER_ACTIVATION_FUNC == "tanh":
            value = Dense(1, activation = "tanh")(c)
        # paper says this works better than score value
        # winrate_value = win rate of 0-indexed player
        # loss: if use sigmoid: binary_crossentropy
        #       if use tanh: MSE
        # use tanh as activation has one convinient thing that opponent's value is just negtive
        
        #score = Dense(2, activation = "relu")(c)
        # this could be an auxiliry output
        # score_value = [0-indexed player's score, 1-indexed player's score]
        # score_values are positive
        # loss of score_value should be MSE
        self.critic = Model([input_grid, input_playas], value)


        self.combined = Model([input_grid, input_playas], [policy, value]) # this is for training use
        self.combined.summary()

        self.optimizer = optimizers.RMSprop(lr = self.LEARNING_RATE)
        self.actor.compile(optimizer = self.optimizer)
        self.critic.compile(optimizer = self.optimizer)

    # predict_on_batch seem to be faster than predict if only one batch is feeded
    def get_value(self, boardgrid, playas):
        return self.critic.predict_on_batch([boardgrid[np.newaxis], np.array([playas]) ])[0,0]

    def get_policy(self, boardgrid, playas):
        return self.actor.predict_on_batch([boardgrid[np.newaxis], np.array([playas]) ])[0]

    def get_masked_policy(self, board, temperature):
        invalid_mask = board.get_valid_move_mask()
        policy = self.get_policy(board.grid, board.next)
        if board.next == WHITE:
            policy = -policy
        # Dirichlet's alpha = 10 / (averge legal move of every state of game)
        # in 19*19 it is 250 so alpha = 0.03
        # in 9*9 it maybe 60 so alpha = 0.1
        alpha = 10 / self.size_square
        noise = np.random.dirichlet(alpha = [alpha] * self.action_size)
        noised_policy = 0.75 * policy + 0.25 * noise
        masked_policy = masked_softmax(invalid_mask, noised_policy, temperature)
        #print(all(np.logical_or(np.logical_not(invalid_mask), (instinct < 1e-4)))) # if output False, something is wrong
        return masked_policy

    def decide(self, board, temperature):
        masked_policy = self.get_masked_policy(board, temperature)
        act = np.random.choice(self.action_size, 1, p = masked_policy)[0]
        return act % self.size, act // self.size

    def decide_monte_carlo(self, board, playout, temperature = 0.1):
        prev_point = board.log[-1][1] if len(board.log) else (-1, 0)
        prev_action = prev_point[0] + prev_point[1] * self.size
        x, y = self.monte_carlo.search(board, prev_action, int(playout), temperature)
        return x, y

    def pop_step(self):
        self.step_records[-1].pop()

    def push_step(self, prev_state, action, cur_state):
        self.step_records[-1].push(prev_state, action, cur_state)

    def enqueue_new_record(self, reward):
        self.step_records[-1].game_result = reward
        if self.step_records[-1].length > 0:
            if len(self.step_records) >= self.STEP_RECORD_AMOUNT_MAX:
                self.step_records = self.step_records[1:]
            self.step_records.append(StepRecord())

    # def reward_normalization(self, x):
    #     return (x - x.mean()) / (np.std(x) + 1e-9)

    def learn(self, verbose=True):
        if len(self.step_records) < self.LEARN_RECORD_SIZE: return
        random_id = np.random.randint(len(self.step_records), size = self.LEARN_RECORD_SIZE)
        for id in random_id:
            rec_ps, rec_a, rec_cs, rec_r = self.step_records[id].get_arrays()
            rec_len = self.step_records[id].length
            if rec_len < 2: continue

            play_as_array = np.zeros((rec_len, 1))
            play_as_array[1::2] = WHITE
            play_as_array[1::2] = BLACK

            """
                Critic Loss = 1/2 * (Adventage)^2 
                            = 1/2 * (reward(s') + value(s') * GAMMA - value(s)) ^ 2

                Actor Loss = -Adventage * log(logits[a])

                Combined loss = Critic Loss + Actor Loss - Actor Entropy
            """
            with tf.GradientTape() as tape:

                cs_policy, cs_value = self.combined([rec_cs, play_as_array])

                # critic loss
                discounted_reward_sum = np.zeros((rec_len, 1))
                discounted_reward_sum[-1] = rec_r
                for i in range(1, rec_len):
                    discounted_reward_sum[-1-i] = discounted_reward_sum[-i] * self.GAMMA
                cs_value = self.critic([rec_cs, play_as_array])
                adventage = tf.convert_to_tensor(discounted_reward_sum, dtype = np.float32) - cs_value # convert as float32 because keras uses float32
                critic_loss = tf.reduce_mean(0.5 * (adventage ** 2)) # Huber loss

                # actor loss
                action_one_hot = np.zeros((rec_len, self.action_size), dtype = np.float32)
                tf.one_hot(rec_a, self.action_size, dtype=tf.float32)
                log_policy = tf.math.log(action_one_hot * cs_policy + 1e-20)
                entropy = tf.reduce_sum(cs_policy * tf.math.log(cs_policy + 1e-20), axis=1) * 0.01
                actor_loss = tf.reduce_mean(-adventage * log_policy)

                # optimize
                total_loss = critic_loss + actor_loss + entropy
                grads = tape.gradient(total_loss, self.combined.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.combined.trainable_weights))
        
        if verbose:
            print("closs: %.3f, aloss: %.3f" % (critic_loss, actor_loss))

    def save(self, save_weight_name):
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)
