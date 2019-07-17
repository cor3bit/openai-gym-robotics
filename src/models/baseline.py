'''
Based on keras-rl example
https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_mujoco.py
'''

import numpy as np

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)


def tf_train(env, n_episodes, save_weights):
    agent = _create_agent(env)

    # Training
    agent.fit(env, nb_steps=n_episodes, visualize=False, verbose=1)

    # Save weights
    if save_weights:
        agent.save_weights('../../models/keras_ddpg_weights.h5f', overwrite=True)


def tf_predict(env, n_episodes, render, weight_path=None):
    agent = _create_agent(env)

    path = weight_path if weight_path is not None else '../../models/keras_ddpg_weights.h5f'
    agent.load_weights(path)

    agent.test(env, nb_episodes=n_episodes, visualize=render, nb_max_episode_steps=200)


def _create_agent(env):
    # Get the environment
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Declare model
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(40))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)

    x = Dense(40)(flattened_observation)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(30)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)

    critic = Model(inputs=[action_input, observation_input], outputs=x)
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)

    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=MujocoProcessor())
    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    return agent
