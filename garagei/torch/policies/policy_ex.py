import akro
import numpy as np
import torch

from garage.torch import global_device
from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garagei.torch.modules.multiplier import Multiplier


class PolicyEx(StochasticPolicy):
    def __init__(self,
                 env_spec,
                 name,
                 *,
                 module_cls,
                 module_kwargs,
                 clip_action=False,
                 omit_obs_idxs=None,
                 option_info=None,
                 ):
        super().__init__(env_spec, name)

        self.env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs

        self._option_info = option_info

        self._module = module_cls(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            **module_kwargs
        )

    def process_observations(self, observations):
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        return observations

    def forward(self, observations):
        observations = self.process_observations(observations)
        dist = self._module(observations)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, '_normal'):
            info.update(dict(
                normal_mean=dist._normal.mean,
                normal_std=dist._normal.variance.sqrt(),
            ))

        return dist, info

    def forward_mode(self, observations):
        observations = self.process_observations(observations)
        samples = self._module.forward_mode(observations)
        return samples, dict()

    def forward_with_transform(self, observations, *, transform):
        observations = self.process_observations(observations)
        dist, dist_transformed = self._module.forward_with_transform(observations, transform=transform)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            ret_mean_transformed = dist_transformed.mean.cpu()
            ret_log_std_transformed = (dist_transformed.variance.sqrt()).log().cpu()
            info = (dict(mean=ret_mean, log_std=ret_log_std),
                    dict(mean=ret_mean_transformed, log_std=ret_log_std_transformed))
        except NotImplementedError:
            info = (dict(),
                    dict())
        return (dist, dist_transformed), info

    def forward_with_chunks(self, observations, *, merge):
        observations = [self.process_observations(o) for o in observations]
        dist = self._module.forward_with_chunks(observations,
                                                merge=merge)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, info

    def get_mode_actions(self, observations):
        if not isinstance(observations[0], np.ndarray) and not isinstance(
                observations[0], torch.Tensor):
            observations = self._env_spec.observation_space.flatten_n(
                observations)

        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(self._env_spec.observation_space, akro.Image) and \
                len(observations.shape) < \
                len(self._env_spec.observation_space.shape):
            observations = self._env_spec.observation_space.unflatten_n(
                observations)
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(
                    global_device())
            samples, info = self.forward_mode(observations)
            return samples.cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_sample_actions(self, observations):
        """Based on super().get_actions()"""
        if not isinstance(observations[0], np.ndarray) and not isinstance(
                observations[0], torch.Tensor):
            observations = self._env_spec.observation_space.flatten_n(
                observations)

        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(self._env_spec.observation_space, akro.Image) and \
                len(observations.shape) < \
                len(self._env_spec.observation_space.shape):
            observations = self._env_spec.observation_space.unflatten_n(
                observations)
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(
                    global_device())
            dist, info = self.forward(observations)
            if isinstance(dist, TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['pre_tanh_value'] = pre_tanh_values.detach().cpu().numpy()
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            return actions, infos

    def get_actions(self, observations):
        actions, info = self.get_sample_actions(observations)
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )
        return actions, info
