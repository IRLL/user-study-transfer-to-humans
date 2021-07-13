import os
import math
from copy import deepcopy

import learnrl as rl
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


class Memory():

    def __init__(self, max_memory_len):
        self.max_memory_len = max_memory_len
        self.memory_len = 0
        self.MEMORY_KEYS = ('observation', 'action', 'reward', 'done', 'next_observation')
        self.datas = {key:None for key in self.MEMORY_KEYS}

    def remember(self, observation, action, reward, done, next_observation):
        for val, key in zip((observation, action, reward, done, next_observation), self.MEMORY_KEYS):
            batched_val = tf.expand_dims(val, axis=0)
            if self.memory_len == 0:
                self.datas[key] = batched_val
            else:
                self.datas[key] = tf.concat((self.datas[key], batched_val), axis=0)
            self.datas[key] = self.datas[key][-self.max_memory_len:]
        
        self.memory_len = len(self.datas[self.MEMORY_KEYS[0]])

    def sample(self, sample_size, method='random'):
        if method == 'random':
            indexes = tf.random.shuffle(tf.range(self.memory_len))[:sample_size]
            datas = [tf.gather(self.datas[key], indexes) for key in self.MEMORY_KEYS]
        elif method == 'last':
            datas = [self.datas[key][-sample_size:] for key in self.MEMORY_KEYS]
        else:
            raise ValueError(f'Unknowed method {method}')
        return datas
    
    def __len__(self):
        return self.memory_len


class LegalControl():

    def __init__(self, exploration=0, exploration_decay=0, exploration_minimum=0):
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_minimum = exploration_minimum
    
    def update_exploration(self):
        self.exploration *= 1 - self.exploration_decay
        self.exploration = max(self.exploration, self.exploration_minimum)

    def act(self, Q, legal_actions):
        raise NotImplementedError('You must define act(self, Q) when subclassing LegalControl')

    def __call__(self, Q, legal_actions, greedy):
        Q = tf.where(legal_actions, Q, -math.inf)
        if greedy:
            actions = tf.argmax(Q, axis=-1, output_type=tf.int32)
        else:
            actions = self.act(Q, legal_actions)
        return actions

def batched_random_valid_index(valid):
    batch_size = valid.shape[0]
    indexes = tf.broadcast_to(tf.range(valid.shape[1]), valid.shape)
    ragged_valid_indexes = tf.ragged.boolean_mask(indexes, valid)
    row_lengths = ragged_valid_indexes.nested_row_lengths()[0]
    float_row_lengths = tf.cast(row_lengths, tf.float32)
    randomly_selected = tf.cast(tf.random.uniform(row_lengths.shape, maxval=float_row_lengths), tf.int32)
    selected_indexes = tf.concat((tf.range(batch_size), randomly_selected), axis=-1)
    return tf.gather_nd(ragged_valid_indexes, selected_indexes)

class EpsGreedy(LegalControl):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.exploration <= 1 and self.exploration >= 0, \
            "Exploration must be in [0, 1] for EpsGreedy"

    def act(self, Q, legal_actions):
        batch_size = Q.shape[0]
        action_size = Q.shape[1]

        actions_random = batched_random_valid_index(legal_actions)
        actions_greedy = tf.argmax(Q, axis=-1, output_type=tf.int32)

        rd = tf.random.uniform((batch_size,), 0, 1)
        actions = tf.where(rd <= self.exploration, actions_random, actions_greedy)

        return actions


class Evaluation():

    def __init__(self, discount):
        self.discount = discount

    def eval(self, rewards, dones, next_observations, action_value):
        raise NotImplementedError('You must define eval when subclassing Evaluation')

    def __call__(self, rewards, dones, next_observations, action_value):
        return self.eval(rewards, dones, next_observations, action_value)

class QLearning(Evaluation):

    def eval(self, rewards, dones, next_observations, action_value):
        futur_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_values = tf.reduce_max(action_value(next_observations[ndones]), axis=-1)

            ndones_indexes = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(futur_rewards, ndones_indexes, self.discount * next_values)
        
        return futur_rewards


class DQLegalAgent(rl.Agent):

    def __init__(self, action_value:tf.keras.Model=None,
                    control:LegalControl=None,
                    memory:Memory=None,
                    evaluation:Evaluation=None,
                    sample_size=32,
                    learning_rate=1e-4,
                    cql_weight=0,
                    cql_weight_softmax=1.,
                    cql_weight_action=1.,
                    n_replays=1,
                    freezed_steps=0):

        self.action_value = action_value
        self.action_value_learner = deepcopy(action_value)
        self.action_value_opt = tf.keras.optimizers.Adam(learning_rate)

        self.control = LegalControl() if control is None else control
        self.memory = memory
        self.evaluation = evaluation

        self.sample_size = sample_size
        self.n_replays = n_replays
        self.cql_weight = cql_weight
        self.cql_weight_softmax = cql_weight_softmax
        self.cql_weight_action = cql_weight_action

        self.freezed_steps = freezed_steps
        self._freezed_steps = freezed_steps

    @tf.function
    def act(self, observation, greedy=False):
        observation, legal_actions = observation

        observations = tf.expand_dims(observation, axis=0)
        observations = self.preprocess(observations)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        Q = self.action_value(observations)
        action = self.control(Q, legal_actions, greedy)[0]
        return action

    def preprocess(self, observations):
        observations = tf.cast(observations, tf.float32)
        return tf.math.log1p(observations) / (5 * tf.math.log(2.))

    def learn(self):
        metrics = {}

        if len(self.memory) > self.sample_size:

            for _ in range(self.n_replays):
                observations, actions, rewards, dones, next_observations = self.memory.sample(self.sample_size)
                expected_futur_rewards = self.evaluation(rewards, dones, next_observations, self.action_value)

                with tf.GradientTape() as tape:
                    Q = self.action_value_learner(observations)

                    action_index = tf.stack( (tf.range(len(actions)), actions) , axis=-1)
                    Q_action = tf.gather_nd(Q, action_index)

                    max_Qs = tf.math.reduce_max(Q, axis=-1, keepdims=True)
                    Q_softmax = tf.math.log(tf.reduce_sum(tf.math.exp(Q), axis=-1))
                    cql_softmax = tf.reduce_mean(Q_softmax)
                    cql_action = tf.reduce_mean(Q_action)
                    cql_loss = self.cql_weight_softmax * cql_softmax - self.cql_weight_action * cql_action
                    mse_loss = tf.keras.losses.mse(expected_futur_rewards, Q_action)
                    loss = mse_loss + self.cql_weight * cql_loss

                grads = tape.gradient(loss, self.action_value_learner.trainable_weights)
                self.action_value_opt.apply_gradients(zip(grads, self.action_value_learner.trainable_weights))

            if self._freezed_steps == 0:
                self.action_value.set_weights(self.action_value_learner.get_weights())
                self._freezed_steps = self.freezed_steps
            else:
                self._freezed_steps -= 1

            metrics = {
                'value': tf.reduce_mean(Q_action).numpy(),
                'mse_loss': mse_loss.numpy(),
                'cql_softmax': cql_softmax.numpy(),
                'cql_action': cql_action.numpy(),
                'cql_loss': cql_loss.numpy(),
                'loss': loss.numpy(),
                'exploration': self.control.exploration,
                'learning_rate': self.action_value_opt.lr.numpy()
            }

            self.control.update_exploration()
        return metrics

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        observation, _ = observation
        observation = self.preprocess(observation)
        next_observation, _ = next_observation
        next_observation = self.preprocess(next_observation)
        self.memory.remember(observation, action, tf.cast(reward, tf.float32), done, next_observation)
    
    def save(self, filename):
        filename += '.h5'
        tf.keras.models.save_model(self.action_value, filename, save_format='h5')
        print(f'Model saved at {filename}')
    
    def load(self, filename):
        self.action_value = tf.keras.models.load_model(filename)


class RewardScaler(rl.RewardHandler):

    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def reward(self, **kwargs):
        return self.scaling_factor * kwargs.get('reward')

if __name__ == "__main__":
    from crafting import MineCraftingEnv
    from callbacks import WandbCallback, ValidationCallback, CheckpointCallback
    import wandb
    kl = tf.keras.layers

    default_config = {
        'max_memory_len': 5000,

        'exploration': 0.2,
        'exploration_decay': 0,
        'exploration_minimum': 0.2,

        'discount': 0.99,

        'dense_1_size': 256,
        'dense_1_activation': 'relu',
        'dense_2_size': 256,
        'dense_2_activation': 'relu',
        'dense_3_size': 128,
        'dense_3_activation': 'tanh',

        'sample_size': 512,
        'learning_rate': 1e-1,
        'n_replays': 2,

        'freezed_steps': 200,

        'cql_weight': 0.1,
        'cql_weight_softmax': 1,
        'cql_weight_action': 2,
    }

    run = wandb.init(config=default_config, entity='mathisfederico', project="options-metrics")
    config = run.config

    env = MineCraftingEnv(
        max_step=50,
        observe_legal_actions=True,
        tasks=['obtain_diamond'],
        tasks_can_end=[True],
        fps=60
    )

    memory = Memory(config.max_memory_len)
    control = EpsGreedy(
        config.exploration,
        config.exploration_decay,
        config.exploration_minimum
    )
    evaluation = QLearning(config.discount)

    action_value = tf.keras.models.Sequential((
        kl.Dense(config.dense_1_size, activation=config.dense_1_activation),
        kl.Dense(config.dense_2_size, activation=config.dense_2_activation),
        kl.Dense(config.dense_3_size, activation=config.dense_3_activation),
        kl.Dense(env.action_space.n, activation='linear', use_bias=False)
    ))

    agent = DQLegalAgent(
        action_value=action_value,
        control=control,
        memory=memory,
        evaluation=evaluation,
        sample_size=config.sample_size,
        n_replays=config.n_replays,
        learning_rate=config.learning_rate,
        cql_weight=config.cql_weight,
        cql_weight_action=config.cql_weight_action,
        cql_weight_softmax=config.cql_weight_softmax,
        freezed_steps=config.freezed_steps,
    )

    metrics=[
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~agent-rwd', {'steps': 'sum', 'episode': 'sum'}),
        'cql_loss~cql',
        'cql_action~cqlA',
        'cql_softmax~cqlS',
        'mse_loss~mse',
        'loss',
        'exploration~exp',
        'value~Q'
    ]


    class DebugCallback(rl.Callback):

        def on_step_end(self, step, logs):
            print(logs['reward'], logs['handled_reward'])

    valid = ValidationCallback()
    check = CheckpointCallback(os.path.join('models', 'DQLegalAgent'))
    wandbcallback = WandbCallback(run, metrics)

    pg = rl.Playground(env, agent)
    pg.run(
        1000, verbose=2, render=True, metrics=metrics,
        callbacks=[wandbcallback, check],
        reward_handler=RewardScaler(0.5)
    )
