import gym

from src.models.rlagent import RlAgent

if __name__ == '__main__':
    # initializes env
    env = gym.make('Ant-v2')

    # initializes an RL agent with pre-trained weights
    agent = RlAgent(method='dqn')
    agent.load_weights(path=r'models/test.test')

    # runs trained agent
    agent.run(env, n_episodes=5, render=True)

    # post-processing
    env.close()
