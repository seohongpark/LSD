import copy
import gc

import numpy as np
import torch
from matplotlib import cm

import aim_wrapper
import dowel_wrapper
from dowel import Histogram
from garage import TrajectoryBatch
from garage.np.algos.rl_algorithm import RLAlgorithm
from garagei import log_performance_ex
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import compute_total_norm, TrainContext
from iod.utils import draw_2d_gaussians, get_option_colors, FigManager, MeasureAndAccTime, record_video


class IOD(RLAlgorithm):
    def __init__(
            self,
            *,
            env_spec,
            normalizer,
            normalizer_type,
            normalizer_mean,
            normalizer_std,
            normalized_env_eval_update,
            option_policy,
            traj_encoder,
            optimizer,
            alpha,
            max_path_length,
            max_optimization_epochs,
            n_epochs_per_eval,
            n_epochs_per_first_n_eval,
            custom_eval_steps,
            n_epochs_per_log,
            n_epochs_per_tb,
            n_epochs_per_save,
            n_epochs_per_pt_save,
            n_epochs_per_pkl_update,
            dim_option,
            num_eval_options,
            num_eval_trajectories_per_option,
            num_random_trajectories,
            eval_record_video,
            video_skip_frames,
            eval_deterministic_traj,
            eval_deterministic_video,
            eval_plot_axis,
            eval_plot_walls,
            name='IOD',
            device=torch.device('cpu'),
            num_train_per_epoch=1,
            discount=0.99,
            record_metric_difference=True,
            te_max_optimization_epochs=None,
            te_trans_optimization_epochs=None,
            trans_minibatch_size=None,
            trans_optimization_epochs=None,
            discrete=False,
    ):
        self.discount = discount
        self.max_path_length = max_path_length
        self.max_optimization_epochs = max_optimization_epochs

        self.device = device
        self.normalizer = normalizer
        self.normalizer_type = normalizer_type
        self.normalized_env_eval_update = normalized_env_eval_update
        self.option_policy = option_policy.to(self.device)
        self.traj_encoder = traj_encoder.to(self.device)
        self.param_modules = {
            'traj_encoder': self.traj_encoder,
            'option_policy': self.option_policy,
        }

        self.alpha = alpha
        self.name = name

        self.dim_option = dim_option

        self._num_train_per_epoch = num_train_per_epoch
        self._env_spec = env_spec

        self.n_epochs_per_eval = n_epochs_per_eval
        self.n_epochs_per_first_n_eval = n_epochs_per_first_n_eval
        self.custom_eval_steps = custom_eval_steps
        self.n_epochs_per_log = n_epochs_per_log
        self.n_epochs_per_tb = n_epochs_per_tb
        self.n_epochs_per_save = n_epochs_per_save
        self.n_epochs_per_pt_save = n_epochs_per_pt_save
        self.n_epochs_per_pkl_update = n_epochs_per_pkl_update
        self.num_eval_options = num_eval_options
        self.num_eval_trajectories_per_option = num_eval_trajectories_per_option
        self.num_random_trajectories = num_random_trajectories
        self.eval_record_video = eval_record_video
        self.video_skip_frames = video_skip_frames
        self.eval_deterministic_traj = bool(eval_deterministic_traj)
        self.eval_deterministic_video = eval_deterministic_video
        self.eval_plot_axis = eval_plot_axis
        if (not eval_plot_walls) and eval_plot_axis is None:
            self.eval_plot_axis = 'nowalls'

        assert isinstance(optimizer, OptimizerGroupWrapper)
        self._optimizer = optimizer

        self._record_metric_difference = record_metric_difference
        self._cur_max_path_length = max_path_length

        self.te_max_optimization_epochs = te_max_optimization_epochs
        self.te_trans_optimization_epochs = te_trans_optimization_epochs

        self._trans_minibatch_size = trans_minibatch_size
        self._trans_optimization_epochs = trans_optimization_epochs

        self._cur_obs_mean = None
        self._cur_obs_std = None

        if self.normalizer_type == 'manual':
            self._cur_obs_mean = np.full(self._env_spec.observation_space.flat_dim, normalizer_mean)
            self._cur_obs_std = np.full(self._env_spec.observation_space.flat_dim, normalizer_std)
        else:
            # Set to the default value
            self._cur_obs_mean = np.full(self._env_spec.observation_space.flat_dim, 0.)
            self._cur_obs_std = np.full(self._env_spec.observation_space.flat_dim, 1.)

        self.discrete = discrete

        self.eval_options = {}
        self.gaussian_eval_options = {}
        self.eval_option_colors = {}
        self.video_eval_options = {}
        self.gaussian_video_eval_options = {}
        self._setup_evaluation(self.dim_option)

        self.traj_encoder.eval()  # for batch norm

    @property
    def policy(self):
        raise NotImplementedError()

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p

    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        # Actually itr + 1 is correct (to match with step_epochs' logging period)
        logging_enabled = ((runner.step_itr + 1) % self.n_epochs_per_log == 0)

        data = self.process_samples(paths, training=True, logging_enabled=logging_enabled)

        time_computing_metrics = [0.0]
        time_training = [0.0]

        if logging_enabled:
            metrics_from_processing = data.pop('metrics')

            with torch.no_grad(), MeasureAndAccTime(time_computing_metrics):
                tensors_before, _ = self._compute_common_tensors(data, compute_extra_metrics=True,
                                                                 op_compute_chunk_size=self._optimizer._minibatch_size)
                gc.collect()

        with MeasureAndAccTime(time_training):
            self._train_once_inner(data)

        performence = log_performance_ex(
            itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self.discount,
        )
        discounted_returns = performence['discounted_returns']
        undiscounted_returns = performence['undiscounted_returns']

        if logging_enabled:
            with torch.no_grad(), MeasureAndAccTime(time_computing_metrics):
                tensors_after, _ = self._compute_common_tensors(data, compute_extra_metrics=True,
                                                                op_compute_chunk_size=self._optimizer._minibatch_size)
                gc.collect()

            prefix_tabular, prefix_aim = aim_wrapper.get_metric_prefixes()
            with dowel_wrapper.get_tabular().prefix(prefix_tabular + self.name + '/'), dowel_wrapper.get_tabular(
                    'plot').prefix(prefix_tabular + self.name + '/'):
                def _record_scalar(key, val):
                    dowel_wrapper.get_tabular().record(key, val)
                    aim_wrapper.track(val, name=(prefix_aim + self.name + '__' + key), epoch=itr)

                def _record_histogram(key, val):
                    dowel_wrapper.get_tabular('plot').record(key, Histogram(val))

                for k in tensors_before.keys():
                    if tensors_before[k].numel() == 1:
                        _record_scalar(f'{k}Before', tensors_before[k].item())
                        if self._record_metric_difference:
                            _record_scalar(f'{k}After', tensors_after[k].item())
                            _record_scalar(f'{k}Decrease', (tensors_before[k] - tensors_after[k]).item())
                    else:
                        _record_histogram(f'{k}Before', tensors_before[k].detach().cpu().numpy())
                with torch.no_grad():
                    total_norm = compute_total_norm(self.all_parameters())
                    _record_scalar('TotalGradNormAll', total_norm.item())
                    for key, module in self.param_modules.items():
                        total_norm = compute_total_norm(module.parameters())
                        _record_scalar(f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}', total_norm.item())
                for k, v in extra_scalar_metrics.items():
                    _record_scalar(k, v)
                _record_scalar('TimeComputingMetrics', time_computing_metrics[0])
                _record_scalar('TimeTraining', time_training[0])

                path_lengths = [
                    len(path['actions'])
                    for path in paths
                ]
                _record_scalar('PathLengthMean', np.mean(path_lengths))
                _record_scalar('PathLengthMax', np.max(path_lengths))
                _record_scalar('PathLengthMin', np.min(path_lengths))

                _record_histogram('ExternalDiscountedReturns', np.asarray(discounted_returns))
                _record_histogram('ExternalUndiscountedReturns', np.asarray(undiscounted_returns))

                for k, v in metrics_from_processing.items():
                    _record_scalar(k, v)

        return np.mean(undiscounted_returns)

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunnerTraj): LocalRunnerTraj is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        with aim_wrapper.AimContext({'phase': 'train', 'policy': 'sampling'}):
            for _ in runner.step_epochs(
                    full_tb_epochs=0,
                    log_period=self.n_epochs_per_log,
                    tb_period=self.n_epochs_per_tb,
                    pt_save_period=self.n_epochs_per_pt_save,
                    pkl_update_period=self.n_epochs_per_pkl_update,
                    new_save_period=self.n_epochs_per_save,
            ):
                for p in self.policy.values():
                    p.eval()
                self.traj_encoder.eval()

                eval_policy = (
                        (self.n_epochs_per_eval != 0 and runner.step_itr % self.n_epochs_per_eval == 0)
                        or (self.n_epochs_per_eval != 0 and self.n_epochs_per_first_n_eval is not None
                            and runner.step_itr < self.n_epochs_per_eval and runner.step_itr % self.n_epochs_per_first_n_eval == 0)
                        or (self.custom_eval_steps is not None and runner.step_itr in self.custom_eval_steps)
                )

                if eval_policy:
                    eval_preparation = self._prepare_for_evaluate_policy(runner)
                    self._evaluate_policy(runner, **eval_preparation)

                    self._log_eval_metrics(runner)

                for p in self.policy.values():
                    p.train()
                self.traj_encoder.train()

                for _ in range(self._num_train_per_epoch):
                    time_sampling = [0.0]
                    with MeasureAndAccTime(time_sampling):
                        runner.step_path = self._get_train_trajectories(runner)
                    last_return = self.train_once(
                        runner.step_itr,
                        runner.step_path,
                        runner,
                        extra_scalar_metrics={
                            'TimeSampling': time_sampling[0],
                        },
                    )
                    gc.collect()

                runner.step_itr += 1

        return last_return

    def _get_trajectories(self,
                          runner,
                          sampler_key,
                          batch_size=None,
                          extras=None,
                          update_stats=False,
                          update_normalizer=False,
                          update_normalizer_override=False,
                          worker_update=None,
                          max_path_length_override=None,
                          env_update=None):
        if batch_size is None:
            batch_size = len(extras)
        policy_sampler_key = sampler_key[6:] if sampler_key.startswith('local_') else sampler_key
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories, infos = runner.obtain_exact_trajectories(
                runner.step_itr,
                sampler_key=sampler_key,
                batch_size=batch_size,
                agent_update=self._get_policy_param_values_cpu(policy_sampler_key),
                env_update=env_update,
                worker_update=worker_update,
                update_normalized_env_ex=update_normalizer if self.normalizer_type == 'garage_ex' else None,
                get_attrs=['env._obs_mean', 'env._obs_var'],
                extras=extras,
                max_path_length_override=max_path_length_override,
                update_stats=update_stats,
            )
        print(f'_get_trajectories({sampler_key}) {time_get_trajectories[0]}s')

        for traj in trajectories:
            for key in ['ori_obs', 'next_ori_obs', 'coordinates', 'next_coordinates']:
                if key not in traj['env_infos']:
                    continue

        if self.normalizer_type == 'garage_ex' and update_normalizer:
            self._set_updated_normalized_env_ex(runner, infos)
        if self.normalizer_type == 'consistent' and update_normalizer:
            self._set_updated_normalizer(runner, trajectories, update_normalizer_override)

        return trajectories

    def _get_train_trajectories(self, runner, burn_in=False):
        default_kwargs = dict(
            runner=runner,
            update_stats=not burn_in,
            update_normalizer=True,
            update_normalizer_override=burn_in,
            max_path_length_override=self._cur_max_path_length,
            worker_update=dict(
                    _deterministic_initial_state=False,
                    _deterministic_policy=False,
            ),
            env_update=dict(_action_noise_std=None),
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs(runner))

        paths = self._get_trajectories(**kwargs)

        return paths

    def process_samples(self, paths, training=False, logging_enabled=True):
        r"""Process sample data based on the collected paths."""
        def _to_torch_float32(x):
            if x.dtype == np.object:
                return np.array([torch.tensor(i, dtype=torch.float32, device=self.device) for i in x], dtype=np.object)
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        valids = np.asarray([len(path['actions'][:self._cur_max_path_length]) for path in paths])
        obs = np.asarray(
            [_to_torch_float32(path['observations'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        ori_obs = np.asarray(
            [_to_torch_float32(path['env_infos']['ori_obs'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        next_obs = np.asarray(
            [_to_torch_float32(path['next_observations'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        next_ori_obs = np.asarray(
            [_to_torch_float32(path['env_infos']['next_ori_obs'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        actions = np.asarray(
            [_to_torch_float32(path['actions'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        rewards = np.asarray(
            [_to_torch_float32(path['rewards'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        dones = np.asarray(
            [_to_torch_float32(path['dones'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)

        # XXX: Abuse next_far_obs to put s_0
        next_far_obs = []
        for ob in obs:
            next_far_obs.append(
                torch.cat([torch.full((ob.size(0) - 1, ob.size(1)), np.nan, device=ob.device), ob[[0]]], dim=0))

        data = dict(
            obs=obs,
            ori_obs=ori_obs,
            next_obs=next_obs,
            next_ori_obs=next_ori_obs,
            next_far_obs=next_far_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            valids=valids,
        )

        for key in paths[0]['agent_infos'].keys():
            data[key] = np.asarray([torch.tensor(path['agent_infos'][key][:self._cur_max_path_length],
                                                 dtype=torch.float32, device=self.device)
                                    for path in paths], dtype=np.object)
        for key in ['step_xi', 'option']:
            if key not in data:
                continue
            next_key = f'next_{key}'
            data[next_key] = copy.deepcopy(data[key])
            for i in range(len(data[next_key])):
                cur_data = data[key][i]
                data[next_key][i] = torch.cat([cur_data[1:], cur_data[-1:]], dim=0)

        if logging_enabled:
            data['metrics'] = dict()

        return data

    def _get_policy_param_values_cpu(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            param_dict[k] = param_dict[k].detach().cpu()
        return param_dict

    def _generate_option_extras(self, options):
        return [{'option': option} for option in options]

    def _gradient_descent(self, loss, optimizer_keys):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._optimizer.step(keys=optimizer_keys)

    def _get_mini_tensors(self, tensors, internal_vars, num_transitions, trans_minibatch_size):
        idxs = np.random.choice(num_transitions, trans_minibatch_size)
        mini_tensors = {}
        mini_internal_vars = {}

        for k, v in tensors.items():
            try:
                if len(v) == num_transitions:
                    mini_tensors[k] = v[idxs]
            except TypeError:
                pass
        for k, v in internal_vars.items():
            try:
                if len(v) == num_transitions:
                    mini_internal_vars[k] = v[idxs]
            except TypeError:
                pass

        return mini_tensors, mini_internal_vars

    def _compute_common_tensors(self, data, *, compute_extra_metrics=False, op_compute_chunk_size=None):
        tensors = {}  # contains tensors to be logged, including losses.
        internal_vars = {  # contains internal variables.
            'maybe_no_grad': {},
        }

        self._update_inputs(data, tensors, internal_vars)

        return tensors, internal_vars

    def _update_inputs(self, data, tensors, v):
        obs = list(data['obs'])
        next_obs = list(data['next_obs'])
        actions = list(data['actions'])
        valids = list(data['valids'])
        dones = list(data['dones'])
        rewards = list(data['rewards'])
        if 'log_prob' in data:
            log_probs = list(data['log_prob'])
        else:
            log_probs = None

        num_trajs = len(obs)
        valids_t = torch.tensor(data['valids'], device=self.device)
        valids_t_f32 = valids_t.to(torch.float32)
        max_traj_length = valids_t.max().item()

        obs_flat = torch.cat(obs, dim=0)
        next_obs_flat = torch.cat(next_obs, dim=0)
        actions_flat = torch.cat(actions, dim=0)
        dones_flat = torch.cat(dones, dim=0).to(torch.int)
        rewards_flat = torch.cat(rewards, dim=0)
        if log_probs is not None:
            log_probs_flat = torch.cat(log_probs, dim=0)
        else:
            log_probs_flat = None

        if 'pre_tanh_value' in data:
            pre_tanh_values = list(data['pre_tanh_value'])
            pre_tanh_values_flat = torch.cat(pre_tanh_values, dim=0)

        dims_action = actions_flat.size()[1:]

        assert obs_flat.ndim == 2
        dim_obs = obs_flat.size(1)
        num_transitions = actions_flat.size(0)

        traj_encoder_extra_kwargs = dict()

        cat_obs_flat = obs_flat
        next_cat_obs_flat = next_obs_flat

        v.update({
            'obs': obs,
            'obs_flat': obs_flat,
            'next_obs_flat': next_obs_flat,
            'cat_obs_flat': cat_obs_flat,
            'next_cat_obs_flat': next_cat_obs_flat,
            'actions_flat': actions_flat,
            'valids': valids,
            'valids_t': valids_t,
            'valids_t_f32': valids_t_f32,
            'dones_flat': dones_flat,
            'rewards_flat': rewards_flat,
            'log_probs_flat': log_probs_flat,

            'dim_obs': dim_obs,
            'dims_action': dims_action,
            'num_trajs': num_trajs,
            'num_transitions': num_transitions,
            'max_traj_length': max_traj_length,
            'traj_encoder_extra_kwargs': traj_encoder_extra_kwargs,
        })

        if 'pre_tanh_value' in data:
            v.update({
                'pre_tanh_values_flat': pre_tanh_values_flat,
            })

    def _set_updated_normalizer(self, runner, paths, override=False):
        original_obs = [torch.tensor(
            path['env_infos']['original_observations'], dtype=torch.float32
        ) for path in paths]
        original_obs_flat = torch.cat(original_obs, dim=0)

        self.normalizer.update(original_obs_flat, override)

        runner.set_hanging_env_update(
            dict(
                _obs_mean=self.normalizer.mean,
                _obs_var=self.normalizer.var,
            ),
            sampler_keys=['option_policy', 'local_option_policy'],
        )

    def _set_updated_normalized_env_ex(self, runner, infos):
        mean = np.mean(infos['env._obs_mean'], axis=0)
        var = np.mean(infos['env._obs_var'], axis=0)

        self._cur_obs_mean = mean
        self._cur_obs_std = var ** 0.5

        runner.set_hanging_env_update(
            dict(
                _obs_mean=mean,
                _obs_var=var,
            ),
            sampler_keys=['option_policy', 'local_option_policy'],
        )

    def _setup_evaluation(self, dim_latent):
        """Set up evaluation options."""
        if self.discrete:
            eval_disc_options = np.arange(0, self.dim_option)
            eval_options = np.eye(self.dim_option)
        else:
            if dim_latent == 2:
                eval_options = [[0., 0.]]
                for dist in [1.5, 3.0, 4.5, 0.75, 2.25, 3.75]:
                    for angle in [0, 4, 2, 6, 1, 5, 3, 7]:
                        eval_options.append([dist * np.cos(angle * np.pi / 4), dist * np.sin(angle * np.pi / 4)])
                eval_options = eval_options[:self.num_eval_options]
                eval_options = np.array(eval_options)
            else:
                eval_options = [[0] * dim_latent]
                for dist in [1.5, -1.5, 3.0, -3.0, 4.5, -4.5]:
                    for dim in range(dim_latent):
                        cur_option = [0] * dim_latent
                        cur_option[dim] = dist
                        eval_options.append(cur_option)
                eval_options = eval_options[:self.num_eval_options]
                eval_options = np.array(eval_options)

        eval_options = np.repeat(eval_options, self.num_eval_trajectories_per_option, axis=0)
        if self.discrete:
            eval_disc_options = np.repeat(eval_disc_options, self.num_eval_trajectories_per_option, axis=0)
            eval_option_colors = []
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            for eval_disc_option in eval_disc_options:
                if eval_disc_option >= 0:
                    eval_option_colors.extend([cm.get_cmap(cmap)(eval_disc_option)[:3]])
                else:
                    eval_option_colors.extend([np.array(cm.get_cmap(cmap)(-(eval_disc_option + 1))[:3]) * 0.7])
            eval_option_colors = np.array(eval_option_colors)
        else:
            eval_option_colors = get_option_colors(eval_options)

        self.eval_options[dim_latent] = eval_options
        self.eval_option_colors[dim_latent] = eval_option_colors

        self.gaussian_eval_options[dim_latent] = eval_options / 4.5 * 3

        if self.eval_record_video:
            if self.discrete:
                video_eval_options = np.eye(self.dim_option)
            else:
                if dim_latent == 2:
                    video_eval_options = []
                    for dist in [4.5]:
                        for angle in [3, 2, 1, 4]:
                            video_eval_options.append([dist * np.cos(angle * np.pi / 4), dist * np.sin(angle * np.pi / 4)])
                    video_eval_options.append([0, 0])
                    for dist in [4.5]:
                        for angle in [0, 5, 6, 7]:
                            video_eval_options.append([dist * np.cos(angle * np.pi / 4), dist * np.sin(angle * np.pi / 4)])
                    video_eval_options = np.array(video_eval_options)
                elif dim_latent <= 5:
                    video_eval_options = []
                    for dist in [2.25, -2.25, 4.5, -4.5]:
                        for dim in range(dim_latent):
                            cur_option = [0] * dim_latent
                            cur_option[dim] = dist
                            video_eval_options.append(cur_option)
                    video_eval_options.append([0.] * dim_latent)
                    video_eval_options = np.array(video_eval_options)
                else:
                    video_eval_options = np.random.randn(9, dim_latent) * 1.5

            # Record each option twice
            video_eval_options = np.repeat(video_eval_options, 2, axis=0)

            self.video_eval_options[dim_latent] = video_eval_options

            if not self.discrete:
                self.gaussian_video_eval_options[dim_latent] = video_eval_options / 4.5 * 1.25
            else:
                self.gaussian_video_eval_options[dim_latent] = video_eval_options

    def _get_coordinates_trajectories(self, trajectories, include_last):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].dtype == np.object:
                coords = np.concatenate(trajectory['env_infos']['coordinates'], axis=0)
                if include_last:
                    coords = np.concatenate([
                        coords,
                        [trajectory['env_infos']['next_coordinates'][-1][-1]],
                    ])
            elif trajectory['env_infos']['coordinates'].ndim == 2:
                coords = trajectory['env_infos']['coordinates']
                if include_last:
                    coords = np.concatenate([
                        coords,
                        [trajectory['env_infos']['next_coordinates'][-1]]
                    ])
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coords = trajectory['env_infos']['coordinates'].reshape(-1, 2)
                if include_last:
                    coords = np.concatenate([
                        coords,
                        trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                    ])
            coordinates_trajectories.append(np.asarray(coords))
        return coordinates_trajectories

    def _get_sp_options_at_timesteps(self, data, use_zero_options=False):
        sp_obs = data['obs']
        if use_zero_options:
            zero_options = np.zeros((len(sp_obs), self.dim_option))
            return zero_options, np.ones((len(sp_obs), self.dim_option)), zero_options
        last_obs = torch.stack([sp_ob[-1] for sp_ob in sp_obs])
        sp_option_dists, _ = self.traj_encoder(last_obs)

        sp_option_means = sp_option_dists.mean.detach().cpu().numpy()
        sp_option_stddevs = torch.ones_like(sp_option_dists.stddev.detach().cpu()).numpy()
        sp_option_samples = sp_option_dists.mean.detach().cpu().numpy()  # Plot from means

        return sp_option_means, sp_option_stddevs, sp_option_samples

    def _plot_sp_and_corresponding_op(self, runner, sp_trajectories, video_sp_trajectories):
        data = self.process_samples(sp_trajectories)
        use_zero_options = False
        sp_option_means, sp_option_stddevs, sp_option_samples = self._get_sp_options_at_timesteps(
                data,
                use_zero_options=use_zero_options,
        )

        sp_option_colors = get_option_colors(sp_option_means)
        sp_option_sample_colors = get_option_colors(sp_option_samples)

        fm_kwargs = dict()
        with FigManager(runner, f'EvalSp__TrajPlot', **fm_kwargs) as fm:
            runner._env.render_trajectories(
                sp_trajectories, sp_option_colors, self.eval_plot_axis, fm.ax
            )
        if hasattr(runner._env, 'is_fetch'):
            self._plot_fetch(f'EvalSp__TrajPlot', sp_trajectories, sp_option_colors, runner, fm_kwargs)

        if self.dim_option == 2:
            with FigManager(runner, f'EvalSp__CPlotFromSpPosterior') as fm:
                draw_2d_gaussians(sp_option_means, sp_option_stddevs, sp_option_colors, fm.ax)
                draw_2d_gaussians(
                    sp_option_samples,
                    [[0.03, 0.03]] * len(sp_option_samples),
                    sp_option_sample_colors,
                    fm.ax,
                    fill=True,
                    use_adaptive_axis=True,
                )
        else:
            with FigManager(runner, f'EvalSp__CPlotFromSpPosterior') as fm:
                draw_2d_gaussians(sp_option_means[:, :2], sp_option_stddevs[:, :2], sp_option_colors, fm.ax)
                draw_2d_gaussians(
                    sp_option_samples[:, :2],
                    [[0.03, 0.03]] * len(sp_option_samples),
                    sp_option_sample_colors,
                    fm.ax,
                    fill=True,
                )

        if self.eval_record_video:
            record_video(runner, 'EvalSp__Video', video_sp_trajectories, skip_frames=self.video_skip_frames)

        eval_metrics = runner._env.calc_eval_metrics(sp_trajectories, is_option_trajectories=False)
        with aim_wrapper.AimContext({'phase': 'eval', 'policy': 'sampling'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, sp_trajectories),
                discount=self.discount,
                additional_records=eval_metrics,
                additional_prefix=type(runner._env.unwrapped).__name__,
            )

    def _plot_op_from_preset(self, runner, random_options_normal=True):
        preset_op_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(self.eval_options[self.dim_option]),
            max_path_length_override=self._cur_max_path_length,
            worker_update=dict(
                    _deterministic_initial_state=self.eval_deterministic_traj,
                    _deterministic_policy=self.eval_deterministic_traj,
            ),
            env_update=dict(_action_noise_std=None),
        )

        random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
        random_option_colors = get_option_colors(random_options)
        random_op_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            max_path_length_override=self._cur_max_path_length,
            worker_update=dict(
                    _deterministic_initial_state=self.eval_deterministic_traj,
                    _deterministic_policy=self.eval_deterministic_traj,
            ),
            env_update=dict(_action_noise_std=None),
        )

        if self.dim_option == 2:
            with FigManager(runner, 'EvalSp__CPlotFromPreset') as fm:
                draw_2d_gaussians(
                    self.eval_options[self.dim_option],
                    [[0.03, 0.03]] * len(self.eval_options[self.dim_option]),
                    self.eval_option_colors[self.dim_option],
                    fm.ax,
                    fill=True,
                )

        if not getattr(runner._env, 'has_no_coordinates', False):
            with FigManager(runner, 'EvalOp__TrajPlotWithCFromPreset') as fm:
                runner._env.render_trajectories(
                    preset_op_trajectories, self.eval_option_colors[self.dim_option], self.eval_plot_axis, fm.ax
                )

        fm_kwargs = dict()
        with FigManager(runner, 'EvalOp__TrajPlotWithCFromRandom', **fm_kwargs) as fm:
            runner._env.render_trajectories(
                random_op_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        if self.eval_record_video:
            video_op_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(self.video_eval_options[self.dim_option]),
                worker_update=dict(
                        _render=True,
                        _deterministic_initial_state=False,
                        _deterministic_policy=False,
                ),
                env_update=dict(_action_noise_std=None),
            )

            record_video(runner, 'EvalOp__VideoWithCFromPreset', video_op_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics = runner._env.calc_eval_metrics(preset_op_trajectories, is_option_trajectories=True)
        with aim_wrapper.AimContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, preset_op_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
                additional_prefix=type(runner._env.unwrapped).__name__,
            )

    def _log_eval_metrics(self, runner):
        runner.eval_log_diagnostics()
        runner.plot_log_diagnostics()
