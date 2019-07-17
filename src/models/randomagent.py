def rnd_train(env, n_episodes, save_weights):
    pass


def rnd_predict(env, n_episodes, render, weight_path=None):
    for i_episode in range(n_episodes):

        observation = env.reset()
        done = False
        while not done:
            if render:
                env.render()

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
