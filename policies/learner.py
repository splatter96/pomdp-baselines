import os, sys
import csv
import time
import pickle
from utils import system, logger

import math
import numpy as np
import torch
from torch.nn import functional as F

from tqdm import tqdm

from hydra.utils import get_original_cwd, to_absolute_path

# import gym
import gymnasium as gym

from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer

# RNN policy on vector-based task
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import evaluation as utl_eval
from utils import logger

from copy import deepcopy


class Learner:
    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed

        self.init_env(**env_args)

        self.init_agent(**policy_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)


    def init_env(
        self,
        env_type,
        env_name,
        max_rollouts_per_task=None,
        num_tasks=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        **kwargs,
    ):
        # initialize environment
        assert env_type in [
            "meta",
            "pomdp",
            "credit",
            "rmdp",
            "generalize",
            "atari",
        ]
        self.env_type = env_type
        self.env_args = kwargs

        # keep the *un-split* env id for (re)building envs, e.g. the vector rollout env
        # (main.py may pass a hyphen-joined id such as "merge-single-agent-v0")
        self._env_name = env_name
        self.vec_train_env = None

        if self.env_type in [
            "pomdp",
            "credit",
        ]:  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp

            # import envs.credit_assign
            sys.path.append(to_absolute_path("highway-env"))
            import highway_env

            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name, config=kwargs)
            if "merge" in env_name:
                self.train_env.config.update(kwargs)
            self.train_env.reset(seed=self.seed)
            self.train_env.action_space.seed(self.seed)  # crucial


            self.eval_env = self.train_env
            # self.eval_env.seed(self.seed + 1)
            if "merge" in env_name:
                self.eval_env.config.update(kwargs)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

            # NOTE: self.train_env is deliberately NOT vectorized here — it is aliased to
            # self.eval_env and `evaluate`/`evaluate_parallel` step it directly. Rollout
            # collection uses a *separate* vectorized copy built in init_train.

        elif self.env_type == "atari":
            from envs.atari import create_env

            assert num_eval_tasks > 0
            self.train_env = create_env(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        else:
            raise ValueError

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False

        # TODO account for not flattened observations
        obs_space = self.train_env.observation_space.shape
        if len(obs_space) == 2:
            self.obs_dim = obs_space[0] * obs_space[1]
        else:
            self.obs_dim = obs_space[0]

        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
        seq_model,
        separate: bool = True,
        image_encoder=None,
        reward_clip=False,
        **kwargs,
    ):
        # initialize agent
        if seq_model == "mlp":
            agent_class = AGENT_CLASSES["Policy_MLP"]
            rnn_encoder_type = None
            assert separate == True
        elif "-mlp" in seq_model:
            agent_class = AGENT_CLASSES["Policy_RNN_MLP"]
            rnn_encoder_type = seq_model.split("-")[0]
            assert separate == True
        else:
            rnn_encoder_type = seq_model
            if separate == True:
                agent_class = AGENT_CLASSES["Policy_Separate_RNN"]
            else:
                agent_class = AGENT_CLASSES["Policy_Shared_RNN"]

        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)

        if image_encoder is not None:  # catch, keytodoor
            image_encoder_fn = lambda: ImageEncoder(
                image_shape=self.train_env.image_space.shape, **image_encoder
            )
        else:
            image_encoder_fn = lambda: None

        self.agent = agent_class(
            encoder=rnn_encoder_type,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            image_encoder_fn=image_encoder_fn,
            **kwargs,
        ).to(ptu.device)
        logger.log(self.agent)

        self.reward_clip = reward_clip  # for atari

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        num_rollouts_per_iter,
        num_rollout_workers=1,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        buffer_type=None,
        target_update_interval=None,
        **kwargs,
    ):
        # parallel rollout collection: 1 = serial (original), >1 = AsyncVectorEnv of N
        self.num_rollout_workers = int(num_rollout_workers)
        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if target_update_interval is None:
            self.target_update_interval = 1
        else:
            self.target_update_interval = target_update_interval

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            if buffer_type is None or buffer_type == SeqReplayBuffer.buffer_type:
                buffer_class = SeqReplayBuffer
            elif buffer_type == RAMEfficient_SeqReplayBuffer.buffer_type:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                observation_type=self.train_env.observation_space.dtype,
            )

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

        # optionally build a vectorized (multiprocessing) env for parallel rollout collection
        if self.num_rollout_workers > 1:
            self._build_vec_train_env(self.num_rollout_workers)

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        eval_stochastic=False,
        num_episodes_per_task=1,
        **kwargs,
    ):
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self):
        """
        training loop
        """

        self._start_training()


        if self.num_init_rollouts_pool > 0:
            logger.log("Collecting initial pool of data..")
            while (
                self._n_env_steps_total
                < self.num_init_rollouts_pool * self.max_trajectory_len
            ):
                self.collect_rollouts(
                    num_rollouts=1,
                    random_actions=True,
                )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:

            start = time.time()
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log("env steps", self._n_env_steps_total)
            end = time.time()
            print(f"Collect rollouts took {end-start}s")

            # TODO only update targets every target_update_interval timesteps
            start = time.time()
            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)
            end = time.time()

            print(f"update took {end-start}s\n\n")

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            print(f"Current num iters: {current_num_iters}")
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                start = time.time()
                perf = self.log()
                end = time.time()
                print(f"Log took {end-start}s\n\n\n")
                if (
                    self.save_interval > 0
                    #and self._n_env_steps_total > 0.75 * self.n_env_steps_total
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf)
        self.save_model(current_num_iters, perf)

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """
        # parallel (vectorized) rollout collection, if a vector env was set up in init_train
        if self.vec_train_env is not None:
            return self.collect_rollouts_vec(
                num_rollouts, random_actions=random_actions
            )

        before_env_steps = self._n_env_steps_total
        crashes = 0
        merges = 0
        for idx in range(num_rollouts):
            steps = 0


            obs = ptu.from_numpy(self.train_env.reset()[0])  # reset


            # TODO create more universal preprocessor for observations
            obs = obs.flatten()
            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info()

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=False,
                        )
                    else:
                        action, _, _, _ = self.agent.act(obs, deterministic=False)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                if self.reward_clip and self.env_type == "atari":
                    reward = torch.tanh(reward)

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                ## determine terminal flag per environment
                # term ignore time-out scenarios, but record early stopping
                term = (
                    False
                    if "TimeLimit.truncated" in info or steps >= self.max_trajectory_len
                    else done_rollout
                )

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

                if "crashed" in info and info["crashed"] == True:
                    crashes += 1
                elif "merged" in info and info["merged"] == True and done_rollout:
                    merges += 1

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                )
                print(
                    f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1

        logger.record_tabular("merge/crashrate", crashes / num_rollouts)
        logger.record_tabular("merge/mergerate", merges / num_rollouts)
        logger.dump_tabular()
        return self._n_env_steps_total - before_env_steps

    def _make_rollout_env(self):
        """factory used to build one rollout env inside an AsyncVectorEnv worker process.

        gymnasium's AsyncVectorEnv pickles this callable to each subprocess; each child
        then constructs its own SingleAgentMergeEnv. `highway_env` is only importable via
        the sys.path shim established in init_env, which does NOT propagate to forked
        children, so the shim is re-established inside make_env() (child-side).

        IMPORTANT: the returned closure captures only plain data (env_name, env_args) so
        it is picklable. It must NOT capture `self` (the Learner owns torch modules).
        """
        env_name = self._env_name
        env_args = dict(self.env_args)

        def make_env():
            import gymnasium as gym
            import os as _os
            import sys as _sys

            # re-establish the highway_env import shim in this child process
            _he = to_absolute_path("highway-env")
            if _he not in _sys.path:
                _sys.path.insert(0, _he)
            import envs.pomdp  # noqa: F401  ensure POMDP registrations are live in the child
            import highway_env  # noqa: F401  ensures merge-single-agent-v0 is registered

            env = gym.make(env_name, config=dict(env_args))
            # mirror the merge special-case from init_env (no-op for kwargs already passed)
            if "merge" in env_name:
                env.config.update(env_args)
            return env

        return make_env

    def _build_vec_train_env(self, num_workers):
        """Build a gymnasium AsyncVectorEnv of `num_workers` rollout envs (multiprocessing).

        Kept separate from self.train_env, which is aliased to self.eval_env and stepped
        directly by evaluate()/evaluate_parallel().
        """
        import gymnasium as gym

        if self.env_type != "pomdp":
            logger.log(f"[rollout] num_rollout_workers ignored: env_type={self.env_type}")
            return
        # ensure the env id is registered in THIS process (already done in init_env, but be safe)
        import envs.pomdp  # noqa: F401
        _he = to_absolute_path("highway-env")
        if _he not in sys.path:
            sys.path.insert(0, _he)
        import highway_env  # noqa: F401

        factory = self._make_rollout_env()
        self.vec_train_env = gym.vector.AsyncVectorEnv(
            [factory for _ in range(num_workers)],
        )
        logger.log(f"[rollout] built AsyncVectorEnv with {num_workers} workers")

    @torch.no_grad()
    def collect_rollouts_vec(self, num_rollouts, random_actions=False):
        """Collect `num_rollouts` episodes using a gymnasium AsyncVectorEnv of N parallel
        workers, with a single batched policy `act` (B=N) per step.

        Data is written to self.policy_storage exactly like the scalar collect_rollouts:
        add_sample() per step for Markov archs, add_episode() per finished episode for
        recurrent archs. Notes:
          - gymnasium auto-resets a worker on done; its terminal next_obs is the reset
            obs, which is safe (the buffer's valid_starts mask never samples across the
            terminal into it — same semantics as the scalar timeout path).
          - the raw merge env always returns truncated=False; TimeLimit(500) delivers the
            500-step cap as truncated=True + info["TimeLimit.truncated"]. `_is_terminal`
            (crash / position>500 / off-ramp) fires early, so most episodes terminate.
          - each finished worker's RNN hidden state is zeroed, matching the scalar's
            per-episode get_initial_info().
        """
        import numpy as _np

        vec = self.vec_train_env
        N = vec.num_envs
        before_env_steps = self._n_env_steps_total
        crashes = 0
        merges = 0
        collected = 0
        recurrent = self.agent_arch != AGENT_ARCHS.Markov

        # expand the B=1 initial info to B=N (identical to running N scalar episodes)
        prev_action = None
        reward = None
        internal = None
        if recurrent:
            pa0, r0, hs0 = self.agent.get_initial_info()
            prev_action = pa0.new_zeros((N,) + pa0.shape[1:])
            reward = r0.new_zeros((N,) + r0.shape[1:])
            if isinstance(hs0, (tuple, list)):
                internal = tuple(
                    s.new_zeros(s.shape[0], N, s.shape[2]) for s in hs0
                )
            else:
                internal = hs0.new_zeros(hs0.shape[0], N, hs0.shape[2])

        # per-worker current observation (fresh episode start); seed each worker
        # distinctly so their HDV/scene random streams differ. gymnasium >=0.28
        # reset() returns (obs, infos), so unpack the obs.
        cur_obs_np, _reset_infos = vec.reset(seed=[self.seed + b for b in range(N)])
        cur_obs = ptu.from_numpy(
            _np.asarray(cur_obs_np).reshape(N, -1)
        ).float()

        # per-worker episode accumulators (recurrent archs only)
        if recurrent:
            w_obs, w_act, w_rew, w_term, w_next, w_ret = [], [], [], [], [], []
            for _ in range(N):
                w_obs.append([]); w_act.append([]); w_rew.append([])
                w_term.append([]); w_next.append([]); w_ret.append(0.0)

        # gymnasium >=0.28 returns the vector `infos` as a flattened dict:
        #   infos[key] -> np.ndarray of shape (N,) (object dtype with None for
        #   sub-envs that omitted the key); infos["_"+key] -> bool presence mask.
        # Read a per-worker boolean flag defensively.
        def _info_flag(arr, b):
            if arr is None:
                return False
            try:
                return bool(arr[b])
            except (IndexError, TypeError, ValueError):
                return False

        while collected < num_rollouts:
            finished = []  # worker indices that completed an episode this step

            # 1) choose actions
            if random_actions:
                actions_in = _np.asarray(vec.action_space.sample())  # (N,) ints
                action = ptu.from_numpy(actions_in)
                if not self.act_continuous:
                    action = F.one_hot(action.long(), num_classes=self.act_dim).float()
            elif recurrent:
                (action, _, _, _), internal = self.agent.act(
                    prev_internal_state=internal,
                    prev_action=prev_action,
                    reward=reward,
                    obs=cur_obs,
                    deterministic=False,
                )  # action: (N, A) one-hot; internal: (layers, N, H)
                prev_action = action.clone()  # feed this step's action in on the next act
            else:
                action, _, _, _ = self.agent.act(cur_obs, deterministic=False)
            if not random_actions:
                # vec.step expects raw discrete ints (N,) or continuous values (N, A)
                if self.act_continuous:
                    actions_in = _np.asarray(action.cpu().numpy())
                else:
                    actions_in = _np.asarray(torch.argmax(action, dim=-1).cpu().numpy())

            # 2) step all workers in parallel
            next_obs, rewards, terminations, truncations, infos = vec.step(actions_in)
            rewards = _np.asarray(rewards, dtype=float)
            terminations = _np.asarray(terminations, dtype=bool)
            truncations = _np.asarray(truncations, dtype=bool)
            # flattened per-step info arrays (gymnasium >=0.28)
            crashed_arr = infos.get("crashed")
            merged_arr = infos.get("merged")
            if self.reward_clip and self.env_type == "atari":
                rewards = torch.tanh(ptu.from_numpy(rewards)).numpy()
            reward_b = ptu.from_numpy(rewards).view(-1, 1)  # (N, 1)
            cur_next = ptu.from_numpy(_np.asarray(next_obs).reshape(N, -1)).float()

            # 3) per-worker bookkeeping (cur_obs is still this step's PRE-step obs;
            #    finished workers' cur_next is their auto-reset fresh obs)
            for b in range(N):
                obs_b = cur_obs[b:b + 1]
                act_b = action[b:b + 1]
                rew_b = reward_b[b:b + 1]
                next_b = cur_next[b:b + 1]
                # env auto-reset means a terminated/truncated worker has a fresh episode
                # in flight; an episode is "done" for bookkeeping on either flag.
                # TimeLimit (gym.make) already delivers the 500-step cap as
                # truncations[b], so no separate TimeLimit.truncated lookup is needed.
                done_b = bool(terminations[b])
                is_trunc = bool(truncations[b])
                # term: early stop (terminated, not truncated) — mirrors scalar
                term_b = bool(done_b and not is_trunc)
                # per-step crash/merge accounting (mirrors scalar's info checks)
                if _info_flag(crashed_arr, b):
                    crashes += 1
                elif _info_flag(merged_arr, b) and done_b:
                    merges += 1
                if self.agent_arch == AGENT_ARCHS.Markov:
                    # per-step Markov sample (mirrors scalar collect_rollouts)
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs_b.squeeze(0)),
                        action=ptu.get_numpy(
                            act_b.squeeze(0)
                            if self.act_continuous
                            else torch.argmax(act_b.squeeze(0), dim=-1, keepdims=True)
                        ),
                        reward=ptu.get_numpy(rew_b.squeeze(0)),
                        terminal=_np.array([term_b], dtype=float),
                        next_observation=ptu.get_numpy(next_b.squeeze(0)),
                    )
                else:
                    # append every step to the per-worker episode accumulators
                    w_obs[b].append(obs_b)
                    w_act[b].append(act_b)
                    w_rew[b].append(rew_b)
                    w_term[b].append(term_b)
                    w_next[b].append(next_b)
                    w_ret[b] += float(rewards[b])
                if done_b or is_trunc:
                    if recurrent:
                        # reset this worker's recurrent state for the next episode.
                        # internal state tensors are (num_layers, N, H): worker b
                        # lives on the batch dim (1), not the layer dim (0).
                        prev_action[b] = 0.0
                        reward[b] = 0.0
                        if isinstance(internal, tuple):
                            for s in internal:
                                s[:, b, :] = 0.0
                        else:
                            internal[:, b, :] = 0.0
                    finished.append(b)
                    collected += 1

            # 4) push finished episodes to the replay buffer (recurrent archs only)
            if recurrent:
                for b in finished:
                    obs_arr = torch.cat(w_obs[b], dim=0)  # (L, dim)
                    act_arr = torch.cat(w_act[b], dim=0)
                    if not self.act_continuous:
                        act_arr = torch.argmax(act_arr, dim=-1, keepdims=True)  # (L, 1)
                    rew_arr = torch.cat(w_rew[b], dim=0)
                    term_arr = _np.array(w_term[b], dtype=float).reshape(-1, 1)
                    next_arr = torch.cat(w_next[b], dim=0)
                    if len(w_obs[b]) < 2:  # degenerate 1-step episode: pad (masked out at sample)
                        obs_arr = torch.cat([obs_arr, obs_arr[-1:]], dim=0)
                        act_arr = torch.cat([act_arr, act_arr[-1:]], dim=0)
                        rew_arr = torch.cat([rew_arr, rew_arr[-1:]], dim=0)
                        term_arr = _np.concatenate([term_arr, _np.zeros((1, 1), dtype=float)])
                        next_arr = torch.cat([next_arr, next_arr[-1:]], dim=0)
                    self.policy_storage.add_episode(
                        observations=ptu.get_numpy(obs_arr),
                        actions=ptu.get_numpy(act_arr),
                        rewards=ptu.get_numpy(rew_arr),
                        terminals=term_arr,
                        next_observations=ptu.get_numpy(next_arr),
                    )
                    print(f"steps: {len(w_obs[b])} term: {w_term[b][-1]} ret: {w_ret[b]:.2f}")
                    w_obs[b] = []; w_act[b] = []; w_rew[b] = []; w_term[b] = []
                    w_next[b] = []; w_ret[b] = 0.0

            # 5) advance observations for the next step (auto-reset workers now fresh)
            cur_obs = cur_next

            self._n_env_steps_total += N
            self._n_rollouts_total += len(finished)

        logger.record_tabular("merge/crashrate", crashes / num_rollouts)
        logger.record_tabular("merge/mergerate", merges / num_rollouts)
        logger.dump_tabular()
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)

        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate_parallel(self, tasks, deterministic=True, render=False, log=False):
        from joblib import Parallel, delayed, parallel_config

        def eval(task_idx):
            num_episodes = self.max_rollouts_per_task  # k
            crashes = 0
            merges = 0
            speed = 0
            road_speed = 0
            step = 0

            num_steps_per_episode = self.eval_env._max_episode_steps
            initial_veh = deepcopy(self.eval_env.road.vehicles)

            obs = ptu.from_numpy(self.eval_env.reset()[0])  # reset
            obs = obs.flatten()
            obs = obs.reshape(1, obs.shape[-1])

            obs = self.eval_env.observation_type.t = 0
            obs = self.eval_env.observation_type.observe()
            obs = ptu.from_numpy(obs)
            obs = obs.flatten()
            obs = obs.reshape(1, obs.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info()

            for episode_idx in range(num_episodes):
                for i in range(num_steps_per_episode):
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=deterministic,
                        )
                    else:
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=deterministic
                        )

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0), render
                    )

                    speed += info["average_speed"]
                    road_speed += info["average_road_speed"]

                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                    # set: obs <- next_obs
                    obs = next_obs.clone()

                    if "crashed" in info and info["crashed"] == True:
                        crashes += 1
                        with open(f"initial_veh{task_idx}.pkl", "wb") as f:
                            pickle.dump(initial_veh, f)
                    elif "merged" in info and info["merged"] == True and done_rollout:
                        merges += 1

                    if done_rollout:
                        break

            return crashes, merges, speed, road_speed, step

        start = time.time()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = list(
                tqdm(
                    Parallel(return_as="generator", n_jobs=8)(
                        delayed(eval)(i) for i in range(0, len(tasks))
                    ),
                    total=len(tasks),
                )
            )

            total_steps = 0
            total_crashes = 0
            total_merges = 0
            total_speed = 0
            total_road_speed = 0
            for r in res:
                total_crashes += r[0]
                total_merges += r[1]
                total_speed += r[2]
                total_road_speed += r[3]
                total_steps += r[4]

            print(f"Total merges {total_merges}")
            print(f"Total crashes {total_crashes}")
            print(f"Total steps {total_steps}")
            print(f"Total episodes {len(res)}")

            print(f"Crahrate: {total_crashes/len(tasks)}")
            print(f"Mergerate: {total_merges/len(tasks)}")
            print(f"Ego speed: {total_speed/total_steps}")
            print(f"Road speed: {total_road_speed/total_steps}")
            print(f"Took {time.time()-start}")

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True, render=False, log=False):
        num_episodes = self.max_rollouts_per_task  # k
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))
        crashes = 0
        merges = 0

        num_steps_per_episode = self.eval_env._max_episode_steps
        observations = None

        speed = 0
        road_speed = 0

        total_reward = 0

        total_affected_radars_data = []
        ego_positions = []

        ttm_values = []

        # CSV logging of per-step radar interference flags (only when log=True)
        radar_csv_file = None
        radar_csv_writer = None
        radar_csv_path = None

        if log:
            tasks = tqdm(tasks)

        start = time.time()
        for task_idx, task in enumerate(tasks):
            step = 0
            episode_reward = 0

            obs = ptu.from_numpy(self.eval_env.reset()[0])  # reset
            obs = obs.flatten()
            obs = obs.reshape(1, obs.shape[-1])
            initial_veh = deepcopy(self.eval_env.road.vehicles)

            affected_radars_episode = []

            # with open("initial_veh191.pkl", "rb") as f:
            #     self.eval_env.road.vehicles = pickle.load(f)
            #     self.eval_env.set_vehicle(self.eval_env.road.vehicles[0])
            # Need to reobserv when setting the initial state, as
            # the old observation was from the old initial state
            obs = self.eval_env.observation_type.t = 0
            obs = self.eval_env.observation_type.observe()
            obs = ptu.from_numpy(obs)
            obs = obs.flatten()
            obs = obs.reshape(1, obs.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info()

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for i in range(num_steps_per_episode):
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=deterministic,
                        )
                    else:
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=deterministic
                        )

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0), render
                    )

                    episode_reward += reward

                    speed += info["average_speed"]
                    road_speed += info["average_road_speed"]

                    # log which radars perceive interference this timestep to CSV
                    if log and "radars_affected_for_whole_timestep" in info:
                        radars = info["radars_affected_for_whole_timestep"]
                        if radar_csv_writer is None:
                            # open lazily so the header matches the real radar count
                            radar_csv_path = os.path.join(
                                logger.get_dir(), "radar_interference.csv"
                            )
                            radar_csv_file = open(radar_csv_path, "w", newline="")
                            radar_csv_writer = csv.writer(radar_csv_file)
                            radar_csv_writer.writerow(
                                [
                                    "task_idx",
                                    "episode_idx",
                                    "step",
                                    *[f"radar_{r}" for r in range(len(radars))],
                                ]
                            )
                        radar_csv_writer.writerow(
                            [task_idx, episode_idx, i, *radars.astype(int)]
                        )

                    # total_affected_radars_data.append(
                    #     self.eval_env.unwrapped.observation_type.affected_radars_data
                    # )
                    # affected_radars_episode.append(
                    #     self.eval_env.unwrapped.observation_type.affected_radars_data
                    # )
                    # ego_positions.append(self.eval_env.unwrapped.controlled_vehicles[0].position.copy())

                    # add raw reward
                    running_reward += reward.item()
                    # clip reward if necessary for policy inputs
                    if self.reward_clip and self.env_type == "atari":
                        reward = torch.tanh(reward)

                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                    # set: obs <- next_obs
                    obs = next_obs.clone()

                    if "crashed" in info and info["crashed"] == True:
                        crashes += 1
                        # save initial vehicles
                        # with open(f"initial_veh{task_idx}.pkl", "wb") as f:
                        #     pickle.dump(initial_veh, f)
                        # with open(f"affected_radars_episode_new{task_idx}.npy", "wb") as f:
                        #     np.save(f, affected_radars_episode)

                        # with open(f"random_state{task_idx}.pkl", "wb") as f:
                        #     pickle.dump(np.random.get_state(), f)

                    elif "merged" in info and info["merged"] == True and done_rollout:
                        merges += 1
                        if np.isfinite(float(info["time_to_merge"])):
                            ttm_values.append(float(info["time_to_merge"]))

                    if done_rollout:
                        if log:
                            tasks.set_description(
                                f"Crashrate {crashes/(task_idx+1)} Mergerate {merges/(task_idx+1)}"
                            )
                            # print(f"Reward {episode_reward}")
                            total_reward += episode_reward

                        break

                returns_per_episode[task_idx, episode_idx] = running_reward
            total_steps[task_idx] = step

            # with open(
            #     f"radars_{self.env_args['dutycycle']}_{self.eval_env.observation_type.radar_frequency}_any_new_auto_60_frametime_new.npy",
            #     "wb",
            # ) as f:
            #     np.save(f, np.array(total_affected_radars_data))

        # with open(f"ego_positions.npy", "wb") as f:
        #     np.save(f, ego_positions)
        if radar_csv_file is not None:
            radar_csv_file.close()
            print(f"Saved radar interference log to {radar_csv_path}")
        print(f"Total merges: {merges}")
        print(f"Total crashes: {crashes}")
        print(f"Total episodes: {task_idx}")

        print(f"Ego speed: {speed/total_steps.sum()}")
        print(f"Road speed: {road_speed/total_steps.sum()}")
        print(f"Average Reward: {total_reward/task_idx}")
        print(f"Average time to merge {sum(ttm_values)/task_idx:.3f}")

        print(f"Took {time.time() - start}")
        return returns_per_episode, success_rate, observations, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular("rl_loss/" + k, v)
        ## gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular("rl_loss/" + k, v)
        logger.dump_tabular()

    def log(self):
        # --- log training  ---
        ## set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)

        # --- evaluation ----

        if self.env_type in ["pomdp", "credit", "atari"]:
            returns_eval, success_rate_eval, _, total_steps_eval = self.evaluate(
                self.eval_tasks
            )
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    _,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(success_rate_eval)
            )

            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )
                logger.record_tabular(
                    "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                )

        else:
            raise ValueError

        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular(
            "z/fps",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, iter, perf):
        save_path = os.path.join(
            logger.get_dir(), "save", f"agent_{iter}_perf{perf:.3f}.pt"
        )
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print("load successfully from", ckpt_path)

        # action = ptu.FloatTensor([self.train_env.action_space.sample()])
        # obs = ptu.FloatTensor([self.train_env.observation_space.sample()])
        # reward = ptu.FloatTensor([0])
        # export model to onnx

    def enjoy(self, chkpt_path, render, num_runs):
        self.load_model(chkpt_path)

        self.evaluate(num_runs * [None], deterministic=True, render=render, log=True)
        # self.evaluate_parallel(num_runs * [None], deterministic=True, render=render, log=True)
