import gym

from src.models.rlagent import RlAgent
from src.models.constants import ModelType

ENV = 'Ant-v2'

if __name__ == '__main__':
    # initializes env
    env = gym.make(ENV)

    # initializes an RL agent with pre-trained weights
    agent = RlAgent(method=ModelType.TF_BASELINE)

    # runs trained agent
    agent.run(env, n_episodes=5, render=True)

    # post-processing
    env.close()
