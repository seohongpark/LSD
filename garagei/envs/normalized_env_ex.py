"""An environment wrapper that normalizes action, observation and reward."""
import akro
import gym
import gym.spaces
import gym.spaces.utils
import numpy as np

from garage.envs import EnvSpec

from garagei.envs.akro_wrapper import AkroWrapperTrait

class NormalizedEnvEx(AkroWrapperTrait, gym.Wrapper):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (garage.envs.GarageEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            expected_action_scale=1.,
            flatten_obs=True,
            obs_alpha=0.001,
            reward_alpha=0.001,
            initial_obs_mean=None,
            initial_obs_var=None,
            action_drop_type='state',
            action_drop_prob=None,
    ):
        super().__init__(env)

        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        flat_obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        if initial_obs_mean is not None:
            self._obs_mean = initial_obs_mean
        else:
            self._obs_mean = np.zeros(flat_obs_dim)
        if initial_obs_var is not None:
            self._obs_var = initial_obs_var
        else:
            self._obs_var = np.ones(flat_obs_dim)

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

        self._action_drop_type = action_drop_type
        assert self._action_drop_type in ['state', 'traj']
        self._action_drop_prob = action_drop_prob
        if self._action_drop_prob is not None and self._action_drop_type == 'traj':
            self._curr_traj_action_drop_mask = self._sample_action_drop_mask()

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_space = akro.Box(low=-self._expected_action_scale,
                                         high=self._expected_action_scale,
                                         shape=self.env.action_space.shape)
        else:
            self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.do_update = True

    #@property
    #def spec(self):
    #    return EnvSpec(action_space=self.action_space,
    #                   observation_space=self.observation_space)

    def _update_obs_estimate(self, obs):
        flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        # self._obs_mean = (
        #     1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        # self._obs_var = (
        #     1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(
        #         flat_obs - self._obs_mean)
        # https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
        #delta = obs - self._obs_mean
        delta = flat_obs - self._obs_mean
        self._obs_mean = self._obs_mean + self._obs_alpha * delta
        self._obs_var = (1 - self._obs_alpha) * (self._obs_var + self._obs_alpha * delta ** 2)

    def _update_reward_estimate(self, reward):
        delta = reward - self._reward_mean
        self._reward_mean = self._reward_mean + self._reward_alpha * delta
        self._reward_var = (1 - self._reward_alpha) * (self._reward_var + self._reward_alpha * delta ** 2)

    def _apply_normalize_obs(self, obs):
        """Compute normalized observation.

        Args:
            obs (np.ndarray): Observation.

        Returns:
            np.ndarray: Normalized observation.

        """
        if self.do_update:
            self._update_obs_estimate(obs)
        flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        normalized_obs = (flat_obs -
                          self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        if not self._flatten_obs:
            normalized_obs = gym.spaces.utils.unflatten(
                self.env.observation_space, normalized_obs)
        return normalized_obs

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.

        Args:
            reward (float): Reward.

        Returns:
            float: Normalized reward.

        """
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        ret = self.env.reset(**kwargs)
        self._cur_obs = ret
        if self._normalize_obs:
            ret = self._apply_normalize_obs(ret)
        return ret

    def step(self, action, **kwargs):
        """Feed environment with one step of action and get result.

        Args:
            action (np.ndarray): An action fed to the environment.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        if isinstance(self.env.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            lb, ub = self.env.action_space.low, self.env.action_space.high
            if np.all(lb != -np.inf) and np.all(ub != -np.inf):
                scaled_action = lb + (action + self._expected_action_scale) * (
                    0.5 * (ub - lb) / self._expected_action_scale)
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action, **kwargs)
        info['original_observations'] = self._cur_obs
        info['original_next_observations'] = next_obs

        self._cur_obs = next_obs

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info


normalize_ex = NormalizedEnvEx
