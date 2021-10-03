import akro
import numpy as np
import torch
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch import global_device
from garagei.torch.modules.multiplier import Multiplier


class LstmPolicy(StochasticPolicy):
    def __init__(self,
                 env_spec,
                 name,
                 *,
                 lstm_module_cls,
                 lstm_module_kwargs,
                 post_lstm_module_cls,
                 post_lstm_module_kwargs,
                 clip_action=False,
                 omit_obs_idxs=None,
                 input_dropout_prob=None,
                 input_multiplier=None,
                 option_info=None,
                 state_include_action=False,
                 force_use_mode_actions=False,
                 force_use_option_as_action_dims=None,
                 force_use_option_as_action_start_dim=0,
                 ):
        super().__init__(env_spec, name)

        self.env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs

        if input_dropout_prob is not None:
            self._input_dropout = torch.nn.Dropout(input_dropout_prob)
        else:
            self._input_dropout = torch.nn.Identity()

        if input_multiplier is not None:
            self._input_multiplier = Multiplier(input_multiplier)
        else:
            self._input_multiplier = torch.nn.Identity()

        self._option_info = option_info
        assert not force_use_mode_actions
        assert force_use_option_as_action_dims is None

        self._state_include_action = state_include_action

        self._lstm_num_layers = lstm_module_kwargs['num_layers']
        self._lstm_hidden_dim = lstm_module_kwargs['hidden_dim']

        self._post_lstm_module = post_lstm_module_cls(
            input_dim=self._lstm_hidden_dim,
            output_dim=self._action_dim,
            **post_lstm_module_kwargs
        )

        self._module = lstm_module_cls(
            input_dim=self._obs_dim if not self._state_include_action else self._obs_dim + self._action_dim,
            output_type='BatchTime_Hidden',
            post_lstm_module=self._post_lstm_module,
            bidirectional=False,
            **lstm_module_kwargs,
        )

        # Variables below are only used in get_actions.
        self._prev_actions = None
        self._prev_hiddens = None
        self._prev_cell_states = None

    def reset(self, dones=None, hidden_cell_state_tuples=None):
        """Reset the policy.

        Note:
            If `do_resets` is None, it will be by default np.array([True]),
            which implies the policy will not be "vectorized", i.e. number of
            parallel environments for training data sampling = 1.

        Args:
            dones (numpy.ndarray): Bool that indicates terminal state(s).

        """
        if dones is None:
            dones = np.array([True])
        if self._prev_hiddens is None or len(dones) != self._prev_hiddens.size(1):
            self._prev_actions = torch.zeros((len(dones), self.action_space.flat_dim))
            self._prev_hiddens = torch.zeros((self._lstm_num_layers, len(dones), self._lstm_hidden_dim))
            self._prev_cell_states = torch.zeros((self._lstm_num_layers, len(dones), self._lstm_hidden_dim))

        if hidden_cell_state_tuples is None:
            hidden, cell_state = 0., 0.
        else:
            hidden, cell_state = hidden_cell_state_tuples

        self._prev_actions[dones, :] = 0.
        self._prev_hiddens[:, dones, :] = torch.tensor(hidden, dtype=torch.float32)
        self._prev_cell_states[:, dones, :] = torch.tensor(cell_state, dtype=torch.float32)

    def process_observations(self, observations, actions, seq_lengths):
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        observations = self._input_dropout(observations)
        observations = self._input_multiplier(observations)

        # Split into len(seq_lengths) chunks
        observations = list(observations.split(seq_lengths))
        if actions is not None:
            actions = list(actions.split(seq_lengths))

        if self._state_include_action:
            if actions is not None:
                # Use given actions (goes here when it comes from compute_loss)
                for i in range(len(observations)):
                    # Shift actions by 1
                    cur_actions = torch.cat([torch.zeros((1, actions[i].size(1)), device=actions[i].device), actions[i][:-1]], dim=0)
                    observations[i] = torch.cat([observations[i], cur_actions], dim=1)
            else:
                # Use stored _prev_actions (goes here when it comes from get_actions)
                for i in range(len(observations)):
                    observations[i] = torch.cat([observations[i], self._prev_actions[[i]]], dim=1)

        return observations

    def forward(self, observations, actions, hidden_cell_state_tuple, seq_lengths=None):
        observations = self.process_observations(observations, actions, seq_lengths)
        dist, (hidden, cell_state) = self._module(
            observations,
            hidden_cell_state_tuple=hidden_cell_state_tuple,
        )

        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, (hidden, cell_state), info

    def forward_with_transform(self, observations, actions, hidden_cell_state_tuple, seq_lengths=None, *, transform):
        observations = self.process_observations(observations, actions, seq_lengths)
        (dist, dist_transformed), (hidden, cell_state) = self._module.forward_with_transform(
            observations,
            hidden_cell_state_tuple=hidden_cell_state_tuple,
            transform=transform,
        )

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
        return (dist, dist_transformed), (hidden, cell_state), info

    def forward_with_chunks(self, observations, actions, hidden_cell_state_tuple, seq_lengths, *, merge):
        observations = [
            self.process_observations(o, a, sl)
            for o, a, sl in zip(observations, actions, seq_lengths)
        ]
        dist, (hidden, cell_state) = self._module.forward_with_chunks(
                observations,
                hidden_cell_state_tuple=hidden_cell_state_tuple,
                merge=merge,
        )

        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, (hidden, cell_state), info


    def get_actions(self, observations):
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
            dist, (hidden, cell_state), info = self.forward(
                observations=observations,
                actions=None,
                hidden_cell_state_tuple=(self._prev_hiddens, self._prev_cell_states),
                seq_lengths=[1] * len(observations),
            )
            actions, info = dist.sample().cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )

        self._prev_actions = torch.as_tensor(actions).float().to(global_device())
        self._prev_hiddens = hidden
        self._prev_cell_states = cell_state

        return actions, info
