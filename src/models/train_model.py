import gym

from src.models.rlagent import RlAgent
from src.models.constants import ModelType

ENV = 'Ant-v2'


def train_robot():
    # initializes env
    env = gym.make(ENV)

    # initializes an RL agent with pre-trained weights
    agent = RlAgent(method=ModelType.TF_BASELINE)
    agent.train(env, n_episodes=10000, save_weights=True)

    # post-processing
    env.close()


def test_robot():
    # initializes env
    env = gym.make(ENV)

    # initializes an RL agent with pre-trained weights
    agent = RlAgent(method=ModelType.TF_BASELINE)
    agent.run(env, n_episodes=5, render=True)

    # post-processing
    env.close()


# ----------------- Script -----------------


if __name__ == '__main__':
    # train
    train_robot()

    # test
    test_robot()
