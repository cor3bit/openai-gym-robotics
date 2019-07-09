import gym


def train_robot():
    env = gym.make('Ant-v2')

    a = env.action_space
    b = env.observation_space
    print(b)

    for i_episode in range(5):
        observation = env.reset()
        for t in range(100):
            # env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()


# ----------------- Script -----------------

if __name__ == '__main__':
    train_robot()
