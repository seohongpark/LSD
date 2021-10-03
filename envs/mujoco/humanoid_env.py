# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import defaultdict

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait


def mass_center(sim):
    mass = np.expand_dims(sim.model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


# pylint: disable=missing-docstring
class HumanoidEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 model_path=None,
                 task='forward',
                 goal=None,
                 fixed_initial_state=False,
                 num_action_repeats=None,
                 done_allowing_step_unit=None,
                 fixed_mpl=None,
                 original_env=False,
                 render_hw=100,
                 ):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = 'humanoid.xml'

        self._task = task
        self._goal = goal
        if self._task == "follow_goals":
            self._goal_list = [
                np.array([3.0, -0.5]),
                np.array([6.0, 8.0]),
                np.array([12.0, 12.0]),
            ]
            self._goal = self._goal_list[0]
            print("Following a trajectory of goals:", self._goal_list)

        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self.fixed_initial_state = fixed_initial_state
        self._num_action_repeats = num_action_repeats
        self._done_allowing_step_unit = done_allowing_step_unit
        self._fixed_mpl = fixed_mpl
        self._original_env = original_env
        self.render_hw = render_hw
        xml_path = "envs/mujoco/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)

    def _get_obs(self):
        data = self.sim.data
        if self._original_env:
            return np.concatenate([data.qpos.flat[2:],
                                   data.qvel.flat,
                                   data.cinert.flat,
                                   data.cvel.flat,
                                   data.qfrc_actuator.flat,
                                   data.cfrc_ext.flat])



        data = self.sim.data
        if self._expose_all_qpos:
            obs = np.concatenate([
                data.qpos.flat, data.qvel.flat,
                # data.cinert.flat, data.cvel.flat,
                # data.qfrc_actuator.flat, data.cfrc_ext.flat
            ])
        else:
            obs = np.concatenate([
                data.qpos.flat[2:], data.qvel.flat, data.cinert.flat, data.cvel.flat,
                data.qfrc_actuator.flat, data.cfrc_ext.flat
            ])

        if self._expose_obs_idxs is not None:
            obs = obs[self._expose_obs_idxs]

        return obs

    # def compute_reward(self, ob, next_ob, action=None):
    #     xposbefore = ob[:, 0]
    #     yposbefore = ob[:, 1]
    #     xposafter = next_ob[:, 0]
    #     yposafter = next_ob[:, 1]
    #
    #     forward_reward = (xposafter - xposbefore) / self.dt
    #     sideward_reward = (yposafter - yposbefore) / self.dt
    #
    #     if action is not None:
    #         ctrl_cost = .5 * np.square(action).sum(axis=1)
    #         survive_reward = 1.0
    #     if self._task == "forward":
    #         reward = forward_reward - ctrl_cost + survive_reward
    #     elif self._task == "backward":
    #         reward = -forward_reward - ctrl_cost + survive_reward
    #     elif self._task == "left":
    #         reward = sideward_reward - ctrl_cost + survive_reward
    #     elif self._task == "right":
    #         reward = -sideward_reward - ctrl_cost + survive_reward
    #     elif self._task in ["goal", "follow_goals"]:
    #         reward = -np.linalg.norm(
    #             np.array([xposafter, yposafter]).T - self._goal, axis=1)
    #     elif self._task in ["sparse_goal"]:
    #         reward = (-np.linalg.norm(
    #             np.array([xposafter, yposafter]).T - self._goal, axis=1) >
    #                   -0.3).astype(np.float32)
    #     return reward

    def compute_reward(self, **kwargs):
        return None

    def step(self, a, render=False):
        if hasattr(self, '_step_count'):
            self._step_count += 1

        obsbefore = self._get_obs()
        pos_before = mass_center(self.sim)
        xposbefore = self.sim.data.qpos.flat[0]
        yposbefore = self.sim.data.qpos.flat[1]
        if self._num_action_repeats is None:
            self.do_simulation(a, self.frame_skip)
        else:
            for i in range(self._num_action_repeats):
                self.do_simulation(a, self.frame_skip)
        obsafter = self._get_obs()
        pos_after = mass_center(self.sim)
        xposafter = self.sim.data.qpos.flat[0]
        yposafter = self.sim.data.qpos.flat[1]

        def _get_dads_humanoid_reward():
            alive_bonus = 5.0
            data = self.sim.data
            lin_vel_cost = 0.25 * (
                    pos_after - pos_before) / self.sim.model.opt.timestep
            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            return reward

        def _get_gym_humanoid_reward():
            # gym/envs/mujoco/humanoid.py
            alive_bonus = 5.0
            data = self.sim.data
            lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            return reward

        qpos = self.sim.data.qpos
        if hasattr(self, '_done_internally') and self._done_allowing_step_unit is not None:
            self._done_internally = (self._done_internally or bool((qpos[2] < 1.0) or (qpos[2] > 2.0)))
            done = (self._done_internally and self._step_count % self._done_allowing_step_unit == 0)
        else:
            done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        reward = self.compute_reward(xposbefore=xposbefore, yposbefore=yposbefore, xposafter=xposafter, yposafter=yposafter, cur_done=done)
        if reward is None:
            reward = _get_gym_humanoid_reward()

            if self._task == "follow_goals":
                xposafter = self.sim.data.qpos.flat[0]
                yposafter = self.sim.data.qpos.flat[1]
                reward = -np.linalg.norm(np.array([xposafter, yposafter]).T - self._goal)
                # update goal
                if np.abs(reward) < 0.5:
                    self._goal = self._goal_list[0]
                    self._goal_list = self._goal_list[1:]
                    print("Goal Updated:", self._goal)

            elif self._task == "goal":
                xposafter = self.sim.data.qpos.flat[0]
                yposafter = self.sim.data.qpos.flat[1]
                reward = -np.linalg.norm(np.array([xposafter, yposafter]).T - self._goal)

        ob = self._get_obs()
        info = dict(
            #reward_linvel=lin_vel_cost,
            #reward_quadctrl=-quad_ctrl_cost,
            #reward_alive=alive_bonus,
            #reward_impact=-quad_impact_cost,
            coordinates=np.array([xposbefore, yposbefore]),
            next_coordinates=np.array([xposafter, yposafter]),
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )

        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)

        return ob, reward, done, info

    def reset_model(self):
        self._step_count = 0
        self._done_internally = False

        c = 0.01
        if self.fixed_initial_state:
            self.set_state(
                self.init_qpos,
                self.init_qvel)
        else:
            self.set_state(
                self.init_qpos + np.random.uniform(
                    low=-c, high=c, size=self.sim.model.nq),
                self.init_qvel + np.random.uniform(
                    low=-c,
                    high=c,
                    size=self.sim.model.nv,
                ))

        if self._task == "follow_goals":
            self._goal = self._goal_list[0]
            self._goal_list = self._goal_list[1:]
            print("Current goal:", self._goal)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2.0

    def calc_eval_metrics(self, trajectories, is_option_trajectories, num_coord_dims=2):
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories, num_coord_dims)
        return eval_metrics
