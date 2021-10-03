import torch
from torch.nn import functional as F


def _clip_actions(algo, actions):
    epsilon = 1e-6
    lower = torch.from_numpy(algo._env_spec.action_space.low).to(algo.device) + epsilon
    upper = torch.from_numpy(algo._env_spec.action_space.high).to(algo.device) - epsilon

    clip_up = (actions > upper).float()
    clip_down = (actions < lower).float()
    with torch.no_grad():
        clip = ((upper - actions) * clip_up + (lower - actions) * clip_down)

    return actions + clip


def update_loss_qf(
        algo, tensors, v,
        obs_flat,
        actions_flat,
        next_obs_flat,
        dones_flat,
        rewards_flat,
        policy,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    q1_pred = algo.qf1(obs_flat, actions_flat).flatten()
    q2_pred = algo.qf2(obs_flat, actions_flat).flatten()

    next_action_dists_flat, *_ = policy(next_obs_flat)
    if hasattr(next_action_dists_flat, 'rsample_with_pre_tanh_value'):
        new_next_actions_flat_pre_tanh, new_next_actions_flat = next_action_dists_flat.rsample_with_pre_tanh_value()
        new_next_action_log_probs = next_action_dists_flat.log_prob(new_next_actions_flat, pre_tanh_value=new_next_actions_flat_pre_tanh)
    else:
        new_next_actions_flat = next_action_dists_flat.rsample()
        new_next_actions_flat = _clip_actions(algo, new_next_actions_flat)
        new_next_action_log_probs = next_action_dists_flat.log_prob(new_next_actions_flat)

    target_q_values = torch.min(
        algo.target_qf1(next_obs_flat, new_next_actions_flat).flatten(),
        algo.target_qf2(next_obs_flat, new_next_actions_flat).flatten(),
    )
    target_q_values = target_q_values - alpha * new_next_action_log_probs
    target_q_values = target_q_values * algo.discount

    with torch.no_grad():
        q_target = rewards_flat + target_q_values * (1. - dones_flat)

    # critic loss weight: 0.5
    loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
    loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

    tensors.update({
        'QTargetsMean': q_target.mean(),
        'QTdErrsMean': ((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2,
        'LossQf1': loss_qf1,
        'LossQf2': loss_qf2,
    })


def update_loss_sacp(
        algo, tensors, v,
        obs_flat,
        policy,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    action_dists_flat, *_ = policy(obs_flat)
    if hasattr(action_dists_flat, 'rsample_with_pre_tanh_value'):
        new_actions_flat_pre_tanh, new_actions_flat = action_dists_flat.rsample_with_pre_tanh_value()
        new_action_log_probs = action_dists_flat.log_prob(new_actions_flat, pre_tanh_value=new_actions_flat_pre_tanh)
    else:
        new_actions_flat = action_dists_flat.rsample()
        new_actions_flat = _clip_actions(algo, new_actions_flat)
        new_action_log_probs = action_dists_flat.log_prob(new_actions_flat)

    min_q_values = torch.min(
        algo.qf1(obs_flat, new_actions_flat).flatten(),
        algo.qf2(obs_flat, new_actions_flat).flatten(),
    )

    loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

    tensors.update({
        'SacpNewActionLogProbMean': new_action_log_probs.mean(),
        'LossSacp': loss_sacp,
    })

    v.update({
        'new_action_log_probs': new_action_log_probs,
    })


def update_loss_alpha(
        algo, tensors, v,
):
    loss_alpha = (-algo.log_alpha.param * (
            v['new_action_log_probs'].detach() + algo._target_entropy
    )).mean()

    tensors.update({
        'Alpha': algo.log_alpha.param.exp(),
        'LossAlpha': loss_alpha,
    })


def update_targets(algo):
    """Update parameters in the target q-functions."""
    target_qfs = [algo.target_qf1, algo.target_qf2]
    qfs = [algo.qf1, algo.qf2]
    for target_qf, qf in zip(target_qfs, qfs):
        for t_param, param in zip(target_qf.parameters(), qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - algo.tau) +
                               param.data * algo.tau)
