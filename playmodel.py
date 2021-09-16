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
#tf.compat.v1.disable_eager_execution()

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
    # masked_x = masked_x - max(masked_x)
    exp_masked_x = np.exp(masked_x / temperature)
    masked_softmax_x = exp_masked_x / np.sum(exp_masked_x)
    softmax = np.zeros(x.shape)
    softmax[mask] = masked_softmax_x
    return softmax

class ActorCritic():
    def __init__ (self, size, load_model_file = None) :
        self.actor = None
        self.critic = None
        self.combined = None

        self.target_actor = None
        self.target_critic = None
        self.target_combined = None

        self.SHARE_HIDDEN_LAYER_NUM = 2
        self.SHARE_HIDDEN_LAYER_CHANNEL = 64

        self.CRITIC_OUTLAYER_ACTIVATION_FUNC = "tanh"
        if self.CRITIC_OUTLAYER_ACTIVATION_FUNC == "tanh":
            self.WIN_REWARD = 1
            self.LOSE_REWARD = -1
        elif self.CRITIC_OUTLAYER_ACTIVATION_FUNC == "sigmoid":
            self.WIN_REWARD = 1
            self.LOSE_REWARD = 0

        self.GAMMA = 0.9
        self.LEARNING_RATE = 1e-5

        self.step_records = [StepRecord()]
        self.STEP_RECORD_AMOUNT_MAX = 4
        self.LEARN_RECORD_SIZE = 2

        self.RESIGN_THRESHOLD = (self.WIN_REWARD - self.LOSE_REWARD) * 0.05 + self.LOSE_REWARD # resign if determined value is low than this value

        if load_model_file:
            self.actor = load_model("actor_" + load_model_file + ".h5")
            self.critic = load_model("critic_" + load_model_file + ".h5")
            self.board_size = self.actor.layers[0].output_shape[0][1]
            self.board_size_square = self.board_size ** 2
            self.action_size = self.board_size ** 2 + 1
        else :
            self.board_size = size
            self.board_size_square = size ** 2
            self.action_size = size ** 2 + 1
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
        input_grid = Input((self.board_size, self.board_size, 2))
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
        c = Concatenate()([c, input_playas])
        c = Dense(256)(c)

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
        # self.actor.compile(optimizer = self.optimizer)
        # self.critic.compile(optimizer = self.optimizer)

    # predict_on_batch seem to be faster than predict if only one batch is feeded
    def get_policy_value(self, boardgrid, playas):
        return self.actor.predict_on_batch([boardgrid[np.newaxis], np.array([playas])])[0], self.critic.predict_on_batch([boardgrid[np.newaxis], np.array([playas])])[0,0]

    def get_masked_policy(self, board, temperature):
        invalid_mask = board.get_valid_move_mask()
        policy = self.actor.predict_on_batch([board.grid[np.newaxis], np.array([board.next])])[0]
        if board.next == WHITE:
            policy = -policy

        # Dirichlet's alpha = 10 / (averge legal move of every state of game)
        # in 19*19 it is 250 so alpha = 0.03
        # in 9*9 it maybe 60 so alpha = 0.1
        # alpha = 10 / self.board_size_square
        # noise = np.random.dirichlet(alpha = [alpha] * self.action_size)
        # noised_policy = 0.75 * policy + 0.25 * noise
        # masked_policy = masked_softmax(invalid_mask, noised_policy, temperature)

        masked_policy = masked_softmax(invalid_mask, policy, temperature)

        #print(all(np.logical_or(np.logical_not(invalid_mask), (instinct < 1e-4)))) # if output False, something is wrong
        return masked_policy

    def decide(self, board, temperature):
        masked_policy = self.get_masked_policy(board, temperature)
        act = np.random.choice(self.action_size, 1, p = masked_policy)[0]
        return act % self.board_size, act // self.board_size

    def decide_monte_carlo(self, board, playout, output = False):
        prev_point = board.log[-1][1] if len(board.log) else (-1, 0)
        prev_action = prev_point[0] + prev_point[1] * self.board_size
        x, y = self.monte_carlo.search(board, prev_action, int(playout), output)
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
        
        if len(self.step_records) < self.LEARN_RECORD_SIZE * 2:
             return

        random_id = np.random.randint(len(self.step_records), size = self.LEARN_RECORD_SIZE)
        critic_loss = 0
        actor_loss = 0
        learn_epoch = 0

        for id in random_id:
            rec_ps, rec_a, rec_cs, rec_r = self.step_records[id].get_arrays()
            rec_len = self.step_records[id].length
            if rec_len <= 1: continue # impossible?!
            learn_epoch += 1
            
            prev_playas = np.zeros((rec_len, 1))
            cur_playas = np.zeros((rec_len, 1))
            prev_playas[0::2] = BLACK
            prev_playas[1::2] = WHITE
            cur_playas[0::2] = WHITE
            cur_playas[1::2] = BLACK

            """
                Adventage Actor Critic

                Adventage = Expected_Reward(s, a) - Value(s)
                Expected_Reward ~= Reward(s) + Value(s') * GAMMA
                Adventage = Reward(s) + Value(s') * GAMMA - Value(s)

                (Expected_Reward似乎可以用Discounted_Reward代替)

                Critic Loss = 1/2 * Adventage ^ 2 (Huber Loss)

                Actor Loss = -Adventage * log(Policy(a|s))

                Combined loss = Critic Loss + Actor Loss - Actor Entropy * ALPHA (expect to maximize the entrpoy of actor) (ALPHA is small)
            """
            ps_value = self.critic.predict([rec_ps, prev_playas])[0]
            with tf.GradientTape() as tape:

                cs_policy, cs_value = self.combined([rec_cs, cur_playas])

                # critic loss
                reward_array = np.zeros((rec_len, 1))
                reward_array[-1] = rec_r
                expect_reward = tf.where(reward_array != 0, reward_array, cs_value * self.GAMMA)
                adventage = expect_reward - tf.convert_to_tensor(ps_value, dtype = tf.float32) # convert as float32 because keras uses float32
                critic_loss = tf.reduce_mean((adventage ** 2) / 2) # Huber loss
                # critic_loss = tf.reduce_mean(adventage ** 2) # MSE

                # actor loss
                action_one_hot = tf.one_hot(rec_a, self.action_size, dtype = tf.float32)
                entropy = tf.reduce_sum(cs_policy * tf.math.log(cs_policy + 1e-20))
                actor_loss = tf.reduce_mean(-adventage * tf.math.log(action_one_hot * cs_policy + 1e-20))

                # optimize
                total_loss = critic_loss + actor_loss + entropy * 1e-3
                # total_loss = critic_loss + actor_loss  # no entropy
                grads = tape.gradient(total_loss, self.combined.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.combined.trainable_weights))
        
        if verbose and learn_epoch > 0:
            print("c loss: %.3f\t a loss: %.3f\t entropy: %.3f\t combined loss: %.3f" % (float(critic_loss), float(actor_loss), float(entropy), float(total_loss)))

    def save(self, save_weight_name):
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)
