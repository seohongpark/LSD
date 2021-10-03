import numpy as np
import torch

from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs


class LSD(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            discount,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,
            alive_reward,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.tau = tau
        self.discount = discount

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.alive_reward = alive_reward

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha
        )

        if scale_reward:
            self._reward_scale_factor = 1 / (self.alpha + 1e-12)
        else:
            self._reward_scale_factor = 1
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            extras = self._generate_option_extras(random_options)

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _set_updated_normalized_env_ex_except_sampling_policy(self, runner, infos):
        mean = np.mean(infos['env._obs_mean'], axis=0)
        var = np.mean(infos['env._obs_var'], axis=0)

        self._cur_obs_mean = mean
        self._cur_obs_std = var ** 0.5

        runner.set_hanging_env_update(
            dict(
                _obs_mean=mean,
                _obs_var=var,
            ),
            sampler_keys=[],
        )

    def _update_inputs(self, data, tensors, v):
        super()._update_inputs(data, tensors, v)

        options = list(data['option'])
        traj_options = torch.stack([x[0] for x in options], dim=0)
        assert traj_options.size() == (v['num_trajs'], self.dim_option)
        options_flat = torch.cat(options, dim=0)

        cat_obs_flat = self._get_concat_obs(v['obs_flat'], options_flat)

        next_options = list(data['next_option'])
        next_options_flat = torch.cat(next_options, dim=0)
        next_cat_obs_flat = self._get_concat_obs(v['next_obs_flat'], next_options_flat)

        v.update({
            'traj_options': traj_options,
            'options_flat': options_flat,
            'cat_obs_flat': cat_obs_flat,
            'next_cat_obs_flat': next_cat_obs_flat,
        })

    def _setup_evaluation(self, dim_latent):
        super()._setup_evaluation(dim_latent)

        self.eval_options = self.gaussian_eval_options
        self.video_eval_options = self.gaussian_video_eval_options

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if isinstance(cur_list, torch.Tensor):
                        cur_list = cur_list.detach().cpu().numpy()
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    elif cur_list.ndim == 0:  # valids
                        continue
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1:
                value = np.squeeze(value, axis=1)
            data[key] = np.asarray([torch.from_numpy(value).float().to(self.device)], dtype=np.object)
        data['valids'] = [self._trans_minibatch_size]
        self._compute_reward(data)

        assert len(data['obs']) == 1
        assert self.normalizer_type not in ['consistent', 'garage_ex']

        tensors = {}
        internal_vars = {
            'maybe_no_grad': {},
        }

        self._update_inputs(data, tensors, internal_vars)

        return data, tensors, internal_vars

    def _train_once_inner(self, data):
        self._update_replay_buffer(data)

        self._compute_reward(data)

        for minibatch in self._optimizer.get_minibatch(data, max_optimization_epochs=self.max_optimization_epochs[0]):
            self._train_op_with_minibatch(minibatch)

        for minibatch in self._optimizer.get_minibatch(data, max_optimization_epochs=self.te_max_optimization_epochs):
            self._train_te_with_minibatch(minibatch)

        sac_utils.update_targets(self)

    def _train_te_with_minibatch(self, data):
        tensors, internal_vars = self._compute_common_tensors(data)

        if self.te_trans_optimization_epochs is None:
            assert self.replay_buffer is None
            self._optimize_te(tensors, internal_vars)
        else:
            if self.replay_buffer is None:
                num_transitions = internal_vars['num_transitions']
                for _ in range(self.te_trans_optimization_epochs):
                    mini_tensors, mini_internal_vars = self._get_mini_tensors(
                        tensors, internal_vars, num_transitions, self._trans_minibatch_size
                    )
                    self._optimize_te(mini_tensors, mini_internal_vars)
            else:
                if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                    for i in range(self.te_trans_optimization_epochs):
                        data, tensors, internal_vars = self._sample_replay_buffer()
                        self._optimize_te(tensors, internal_vars)

    def _train_op_with_minibatch(self, data):
        tensors, internal_vars = self._compute_common_tensors(data)

        if self._trans_optimization_epochs is None:
            assert self.replay_buffer is None
            self._optimize_op(tensors, internal_vars)
        else:
            if self.replay_buffer is None:
                num_transitions = internal_vars['num_transitions']
                for _ in range(self._trans_optimization_epochs):
                    mini_tensors, mini_internal_vars = self._get_mini_tensors(
                        tensors, internal_vars, num_transitions, self._trans_minibatch_size
                    )
                    self._optimize_op(mini_tensors, mini_internal_vars)
            else:
                if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                    for _ in range(self._trans_optimization_epochs):
                        data, tensors, internal_vars = self._sample_replay_buffer()
                        self._optimize_op(tensors, internal_vars)

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'],
            optimizer_keys=['qf1'],
        )
        self._gradient_descent(
            tensors['LossQf2'],
            optimizer_keys=['qf2'],
        )

        # LossSacp should be updated here because Q functions are changed by optimizers.
        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

    def _compute_common_tensors(self, data, *, compute_extra_metrics=False, op_compute_chunk_size=None):
        tensors = {}
        internal_vars = {}

        self._update_inputs(data, tensors, internal_vars)

        if compute_extra_metrics:
            self._update_loss_te(tensors, internal_vars)
            self._compute_reward(data, metric_tensors=tensors)
            self._update_loss_qf(tensors, internal_vars)
            self._update_loss_op(tensors, internal_vars)
            self._update_loss_alpha(tensors, internal_vars)

        return tensors, internal_vars

    def _get_te_option_log_probs(self, tensors, v):
        # Note: actually te_option_log_probs isn't a log prob

        cur_z = self.traj_encoder(v['obs_flat'])[0].mean
        next_z = self.traj_encoder(v['next_obs_flat'])[0].mean
        target_z = next_z - cur_z

        if self.discrete:
            target_z = target_z.reshape(target_z.size(0), 1, target_z.size(1))
            eye_z = torch.eye(self.dim_option, device=target_z.device).reshape(1, self.dim_option, self.dim_option).expand(target_z.size(0), -1, -1)
            logits = (eye_z * target_z).sum(dim=2)
            masks = (v['options_flat'] - v['options_flat'].mean(dim=1, keepdim=True)) * (self.dim_option) / (self.dim_option - 1 if self.dim_option != 1 else 1)
            te_option_log_probs = (logits * masks).sum(dim=1)
        else:
            inner = (target_z * v['options_flat']).sum(dim=1)
            te_option_log_probs = inner

        tensors.update({
            'PureTeOptionLogProbMean': te_option_log_probs.mean(),
            'PureTeOptionLogProbStd': te_option_log_probs.std(),
        })

        if self.alive_reward is not None:
            te_option_log_probs = te_option_log_probs + self.alive_reward

        return te_option_log_probs

    def _update_loss_te(self, tensors, v):
        te_option_log_probs = self._get_te_option_log_probs(tensors, v)
        te_option_log_prob_mean = te_option_log_probs.mean()

        loss_te = -te_option_log_prob_mean

        tensors.update({
            'TeOptionLogProbMean': te_option_log_prob_mean,
        })
        v.update({
            'te_option_log_probs': te_option_log_probs,
            'te_option_log_prob_mean': te_option_log_prob_mean,
        })

        tensors.update({
            'LossTe': loss_te,
        })

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs_flat = self.option_policy.process_observations(v['cat_obs_flat'])
        next_processed_cat_obs_flat = self.option_policy.process_observations(v['next_cat_obs_flat'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs_flat=processed_cat_obs_flat,
            actions_flat=v['actions_flat'],
            next_obs_flat=next_processed_cat_obs_flat,
            dones_flat=v['dones_flat'],
            rewards_flat=v['rewards_flat'] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs_flat': processed_cat_obs_flat,
            'next_processed_cat_obs_flat': next_processed_cat_obs_flat,
        })

    def _update_loss_op(self, tensors, v):
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs_flat=v['processed_cat_obs_flat'],
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    def _compute_reward(self, data, metric_tensors=None):
        tensors = {}
        v = {}

        self._update_inputs(data, tensors, v)

        with torch.no_grad():
            te_option_log_probs = self._get_te_option_log_probs(tensors, v)

            rewards = te_option_log_probs

            if metric_tensors is not None:
                metric_tensors.update({
                    'LsdTotalRewards': rewards.mean(),
                })
            rewards = rewards.split(v['valids'], dim=0)

            data['rewards'] = np.asarray(rewards, dtype=np.object)

    def _prepare_for_evaluate_policy(self, runner):
        return {}

    def _evaluate_policy(self, runner, **kwargs):
        random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
        sampling_op_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            update_normalizer=self.normalized_env_eval_update,
        )
        if self.eval_record_video:
            random_options = np.random.randn(len(self.video_eval_options[self.dim_option]), self.dim_option)
            video_sp_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(random_options),
                worker_update=dict(_render=True),
            )
        else:
            video_sp_trajectories = None
        self._plot_sp_and_corresponding_op(runner, sampling_op_trajectories, video_sp_trajectories)
        self._plot_op_from_preset(runner, random_options_normal=True)
        self._log_eval_metrics(runner)
