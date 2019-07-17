import gym
from gym.wrappers import Monitor

from src.models.rlagent import RlAgent
from src.models.constants import ModelType

ENV = 'Ant-v2'
METHOD = ModelType.EXT_BASELINE
N_EPISODES = 100

if __name__ == '__main__':
    # initializes env
    env = gym.make(ENV)

    # initializes an RL agent with pre-trained weights
    agent = RlAgent(method=METHOD)

    # runs trained agent
    agent.run(env, n_episodes=N_EPISODES, render=True, weight_path='models/keras_ddpg_weights.h5f')

    # post-processing
    env.close()
