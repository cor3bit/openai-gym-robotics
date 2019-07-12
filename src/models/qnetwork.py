def dqn_train(env, n_episodes):
    a = env.action_space
    b = env.observation_space
    print(b)

    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(100):
            # env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def dqn_predict(env, n_episodes):
    pass
