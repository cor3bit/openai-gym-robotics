from joblib import dump, load

from src.models.mcpg import mcpg_train, mcpg_predict
from src.models.baseline import tf_predict, tf_train
from src.models.constants import ModelType


class RlAgent:
    _supported_engines = [ModelType.MCPG, ModelType.TF_BASELINE]

    def __init__(self, method):
        assert method in self._supported_engines

        if method == ModelType.MCPG:
            self._learn_func = mcpg_train
            self._predict_func = mcpg_predict
        elif method == ModelType.TF_BASELINE:
            self._learn_func = tf_train
            self._predict_func = tf_predict
        else:
            raise NotImplementedError(f'Method {method} is not yet implemented!')

    def train(self, env, n_episodes, save_weights=True):
        self._learn_func(env, n_episodes, save_weights)

    def run(self, env, n_episodes=10, render=False):
        self._predict_func(env, n_episodes, render)
