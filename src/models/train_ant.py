import gym

from src.models.rlagent import RlAgent

def train_robot():
    # initializes env
    env = gym.make('Ant-v2')

    # initializes an RL agent with pre-trained weights
    agent = RlAgent(method='dqn')
    agent.train(env, n_episodes=5, save_weights=False)

    # runs trained agent
    agent.run(env, n_episodes=5, render=True)

    # post-processing
    env.close()


# ----------------- Script -----------------

if __name__ == '__main__':
    train_robot()
