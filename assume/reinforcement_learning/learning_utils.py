# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable
from datetime import timedelta
import pandas as pd
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch as th

# check which ones are needed later
import logging
from typing import Optional, Dict, Type, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    ExponentialLR,
    LinearLR
)


# TD3 and PPO
class ObsActRew(TypedDict):
    observation: list[th.Tensor]
    action: list[th.Tensor]
    reward: list[th.Tensor]


# TD3 and PPO
observation_dict = dict[list[datetime], ObsActRew]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, action noise scale ...)
Schedule = Callable[[float], float]



# TD3
# Ornstein-Uhlenbeck Noise
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    A class that implements Ornstein-Uhlenbeck noise.
    """

    def __init__(self, action_dimension, mu=0, sigma=0.5, theta=0.15, dt=1e-2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.noise_prev = np.zeros(self.action_dimension)
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else np.zeros(self.action_dimension)
        )

    def noise(self):
        noise = (
            self.noise_prev
            + self.theta * (self.mu - self.noise_prev) * self.dt
            + self.sigma
            * np.sqrt(self.dt)
            * np.random.normal(size=self.action_dimension)
        )
        self.noise_prev = noise

        return noise


# TD3
class NormalActionNoise:
    """
    A gaussian action noise
    """

    def __init__(self, action_dimension, mu=0.0, sigma=0.1, scale=1.0, dt=0.9998):
        self.act_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.dt = dt

    def noise(self):
        noise = (
            self.dt
            * self.scale
            * np.random.normal(self.mu, self.sigma, self.act_dimension)
        )
        return noise

    def update_noise_decay(self, updated_decay: float):
        self.dt = updated_decay


# TD3
def polyak_update(params, target_params, tau: float):
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    Args:
        params: parameters to use to update the target params
        target_params: parameters to update
        tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def linear_schedule_func(
    start: float, end: float = 0.1, end_fraction: float = 1
) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = 1 - ``end_fraction``.

    Args:
        start: value to start with if ``progress_remaining`` = 1
        end: value to end with if ``progress_remaining`` = 0
        end_fraction: fraction of ``progress_remaining``
            where end is reached e.g 0.1 then end is reached after 10%
            of the complete training process.

    Returns:
        Linear schedule function.

    Note:
        Adapted from SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L100

    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_schedule(val: float) -> Schedule:
    """
    Create a function that returns a constant. It is useful for learning rate schedule (to avoid code duplication)

    Args:
        val: constant value
    Returns:
        Constant schedule function.

    Note:
        From SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L124

    """

    def func(_):
        return val

    return func


def collect_obs_for_central_critic(
    states: th.Tensor, i: int, obs_dim: int, unique_obs_dim: int, batch_size: int, public_info = True
) -> th.Tensor:
    """
    This function samels the observations from allagents for the central critic. 
    In detail it takes all actions and concates all unique_obs of the agents and one time the similar observations. 

    Args:
        actions (th.Tensor): The actions
        n_agents (int): Number of agents
        n_actions (int): Number of actions

    Returns:
        th.Tensor: The sampled actions
    """
    # Sample actions for the central critic

    # this takes the unique observations from all other agents assuming that
    # the unique observations are at the end of the observation vector
    
    temp = th.cat(
        (
            states[:, :i, obs_dim - unique_obs_dim :].reshape(
                batch_size, -1
            ),
            states[
                :, i + 1 :, obs_dim - unique_obs_dim :
            ].reshape(batch_size, -1),
        ),
        axis=1,
    )
    # the final all_states vector now contains the current agent's observation
    # and the unique observations from all other agents
    all_states = th.cat(
        (states[:, i, :].reshape(batch_size, -1), temp), axis=1
    ).view(batch_size, -1)
    
    #print(f"all_states_shape: {all_states.shape}")
    
    all_states_private = states[:, i, :].reshape(batch_size, -1)

    #print(f"all_states_private: {all_states_private.shape}")

    private_start_idx = obs_dim - unique_obs_dim
    private_obs = states[:, i, private_start_idx:]
    #print(f"private_obs shape: {private_obs.shape}")  # Debug info
    return all_states_private



def create_lr_scheduler(
        optimizer: Optimizer, 
        scheduler_type: str,
        scheduler_kwargs: Optional[dict] = None
        ):
    """
    Creates a learning rate scheduler for the optimization algorithm
    Args:
        optimizer
        scheduler_type:
            - "none"
            - "step"
            - "exp"
            - "linear
        scheduler_kwargs: 
            - step: {"step_size": int, "gamma": float}
            - exp: {"gamma": float}
            - linear: {"start_factor": float, "end_factor": float, "total_iters": int}
    """
    scheduler_kwargs = scheduler_kwargs or {}
    
    SCHEDULER_CONFIG = {
        "step": {
            "class": StepLR,
            "required_kwargs": {"step_size", "gamma"}
        },
        "exp": {
            "class": ExponentialLR,
            "required_kwargs": {"gamma"}
        },
        "linear": {
            "class": LinearLR,
            "required_kwargs": {"start_factor", "end_factor", "total_iters"}
        }
    }

    if scheduler_type.lower() == "none":
        return None
    
    if scheduler_type not in SCHEDULER_CONFIG:
        raise ValueError(f"unsuported scheduler type {scheduler_type}")
    
    scheduler_conf = SCHEDULER_CONFIG[scheduler_type]
    scheduler_class = scheduler_conf["class"]
    required_kwargs = scheduler_conf["required_kwargs"]
    missing_kwargs = required_kwargs - set(scheduler_kwargs.keys())

    if missing_kwargs:
        raise ValueError(f"missing kwargs: {missing_kwargs}")

    scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    return scheduler
    
