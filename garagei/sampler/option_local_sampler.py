"""Sampler that runs workers in the main process."""
import copy
from collections import defaultdict

from garage import TrajectoryBatch
from garage.sampler import LocalSampler


class OptionLocalSampler(LocalSampler):
    def _update_workers(self, agent_update, env_update, worker_update):
        """Apply updates to the workers.

        Args:
            agent_update(object): Value which will be passed into the
                `agent_update_fn` before doing rollouts. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update(object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `factory.n_workers`, and will be
                spread across the workers.

        """
        agent_updates = self._factory.prepare_worker_messages(agent_update)
        env_updates = self._factory.prepare_worker_messages(
            env_update, preprocess=copy.deepcopy)
        worker_updates = self._factory.prepare_worker_messages(worker_update)
        for worker, agent_up, env_up, worker_up in zip(self._workers, agent_updates,
                                            env_updates, worker_updates):
            worker.update_agent(agent_up)
            worker.update_env(env_up)
            worker.update_worker(worker_up)

    def obtain_exact_trajectories(self,
                                  n_traj_per_worker,
                                  agent_update,
                                  env_update=None,
                                  worker_update=None,
                                  get_attrs=None):
        self._update_workers(agent_update, env_update, worker_update)
        batches = []
        for worker, n_traj in zip(self._workers, n_traj_per_worker):
            for _ in range(n_traj):
                batch = worker.rollout()
                batches.append(batch)

        infos = defaultdict(list)
        if get_attrs is not None:
            for i in range(self._factory.n_workers):
                contents = self._workers[i].get_attrs(get_attrs)
                for k, v in contents.items():
                    infos[k].append(v)

        return TrajectoryBatch.concatenate(*batches), infos
