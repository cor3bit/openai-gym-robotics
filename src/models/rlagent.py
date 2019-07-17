from joblib import dump, load

from src.models.mcpg import mcpg_train, mcpg_predict
from src.models.baseline import tf_predict, tf_train
from src.models.randomagent import rnd_predict, rnd_train
from src.models.constants import ModelType


class RlAgent:
    def __init__(self, method):
        if method == ModelType.MCPG:
            self._learn_func = mcpg_train
            self._predict_func = mcpg_predict
        elif method == ModelType.EXT_BASELINE:
            self._learn_func = tf_train
            self._predict_func = tf_predict
        elif method == ModelType.RANDOM:
            self._learn_func = rnd_train
            self._predict_func = rnd_predict
        else:
            raise NotImplementedError(f'Method {method} is not yet implemented!')

    def train(self, env, n_episodes, save_weights=True):
        self._learn_func(env, n_episodes, save_weights)

    def run(self, env, n_episodes=10, render=False, weight_path=None):
        self._predict_func(env, n_episodes, render, weight_path)
