'''
Based on keras-rl example
https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_mujoco.py
'''

import numpy as np

from keras.models import Sequential, Model
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
    # Get the environment and extract the number of actions.
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(40))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    # print(actor.summary())

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
    # print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)

    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=MujocoProcessor())
    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    # Okay, now it's time to learn something!
    agent.fit(env, nb_steps=n_episodes, visualize=False, verbose=1)

    # After training is done, we save the final weights.
    if save_weights:
        agent.save_weights('../../models/keras_ddpg_weights.h5f', overwrite=True)


def tf_predict(env, n_episodes, render):
    # load agent
    agent = DDPGAgent.load_weights(filepath='../../models/keras_ddpg_weights.h5f')

    agent.test(env, nb_episodes=n_episodes, visualize=render, nb_max_episode_steps=200)
