from joblib import dump, load

from src.models.qnetwork import dqn_train, dqn_predict


class RlAgent:
    _supported_engines = ['dqn']
    _weights_path = r'models/w.joblib'

    def __init__(self, method='dqn'):
        assert method in self._supported_engines

        if method == 'dqn':
            self._learn_func = dqn_train
            self._predict_func = dqn_predict
        else:
            raise NotImplementedError(f'Method {method} is not yet implemented!')

    def train(self, env, n_episodes, save_weights=True):
        self._w = self._learn_func(env, n_episodes)

        if save_weights:
            dump(self._w, filename=self._weights_path)

    def load_weights(self, path=None):
        self._w = load(filename=path if path is not None else self._weights_path)

    def run(self, env, n_episodes=10, render=False):
        pass
