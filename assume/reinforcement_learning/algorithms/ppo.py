# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import torch as th
th.autograd.set_detect_anomaly(True)
from torch.nn import functional as F
from torch.optim import Adam

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.neural_network_architecture import CriticPPO
from assume.reinforcement_learning.learning_utils import collect_obs_for_central_critic

import wandb 
logger = logging.getLogger(__name__)


class PPO(RLAlgorithm):
    """
    Proximal Policy Optimization (PPO) is a robust and efficient policy gradient method for reinforcement learning.
    It strikes a balance between trust-region methods and simpler approaches by using clipped objective functions.
    PPO avoids large updates to the policy by restricting changes to stay within a specified range, which helps stabilize training.
    The key improvements include the introduction of a surrogate objective that limits policy updates and ensures efficient learning,
    as well as the use of multiple epochs of stochastic gradient descent on batches of data.

    Open AI Spinning guide: https://spinningup.openai.com/en/latest/algorithms/ppo.html#

    Original paper: https://arxiv.org/pdf/1802.09477.pdf
    """

    # Change order and mandatory parameters in the superclass, removed and newly added parameters
    def __init__(
        self,
        learning_role,
        learning_rate: float,
        gamma: float,  # Discount factor for future rewards
        gradient_steps: int,  # Number of steps for updating the policy
        clip_ratio: float,  # Clipping parameter for policy updates
        vf_coef: float,  # Value function coefficient in the loss function
        entropy_coef: float,  # Entropy coefficient for exploration
        max_grad_norm: float,  # Gradient clipping value
        gae_lambda: float,  # GAE lambda for advantage estimation
        actor_architecture: str,
        value_clip_ratio: float = 0.2,
        share_critic = False,
        use_base_bid = False,
        learn_std = True,
        public_info = False,
        individual_values = False
    ):
        super().__init__(
            learning_role=learning_role,
            learning_rate=learning_rate,
            gamma=gamma,
            actor_architecture=actor_architecture,
        )
        self.gradient_steps = gradient_steps
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef #policy clip ratio
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.n_updates = 0  # Number of updates performed
        self.batch_size = learning_role.batch_size 
        self.value_clip_ratio = value_clip_ratio #value function clip ratio
        self.value_stats = {} # stats dictionary for value normalization
        self.keep_value_stats = True
        self.share_critic = share_critic
        self.use_base_bid = use_base_bid
        self.learn_std = learn_std
        self.public_info = public_info
        self.wandb_step = 0
        self.individual_values = individual_values

        #check configuration
        print("-----Config Check:-----")
        print(f"self.value_clip_ratio: {self.value_clip_ratio}\nshare_critic: {self.share_critic}\nself.use_base_bid: {self.use_base_bid}\nself.learn_std: {self.learn_std}\nself.public_info: {self.public_info} \nself.individual_values: {self.individual_values}")        
        # write error if different actor_architecture than dist is used
        if actor_architecture != "dist":
            raise ValueError(
                "PPO only supports the 'dist' actor architecture. Please define 'dist' as actor architecture in config."
            )

    # Unchanged method from MATD3
    def save_params(self, directory):
        """
        This method saves the parameters of both the actor and critic networks associated with the learning role. It organizes the
        saved parameters into separate directories for critics and actors within the specified base directory.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

 
    # Removed critic_target in comparison to MATD3
    # Decentralized
    # def save_critic_params(self, directory):
    #     """
    #     Save the parameters of critic networks.

    #     This method saves the parameters of the critic networks, including the critic's state_dict and the critic's optimizer state_dict. 
    #     It organizes the saved parameters into a directory structure specific to the critic associated with each learning strategy.

    #     Args:
    #         directory (str): The base directory for saving the parameters.
    #     """
    #     os.makedirs(directory, exist_ok=True)
    #     for u_id in self.learning_role.rl_strats.keys():
    #         obj = {
    #             "critic": self.learning_role.rl_strats[u_id].critic.state_dict(),
    #             "critic_optimizer": self.learning_role.rl_strats[u_id].critic.optimizer.state_dict(),
    #         }
    #         path = f"{directory}/critic_{u_id}.pt"
            # th.save(obj, path)


    # Centralized
    def save_critic_params(self, directory):
        """
        Save the parameters of critic networks.

        This method saves the parameters of the critic networks, including the critic's state_dict, critic_target's state_dict. It organizes the saved parameters into a directory structure specific to the critic
        associated with each learning   strategy.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        os.makedirs(directory, exist_ok=True)
        if self.share_critic:
            obj = {
            "critic": self.shared_critic.state_dict(),
            "critic_optimizer": self.shared_critic.optimizer.state_dict(),
            }
            path = f"{directory}/shared_critic.pt"
            th.save(obj, path)
        else:
            for u_id in self.learning_role.rl_strats.keys():
                obj = {
                    "critic": self.learning_role.critics[u_id].state_dict(),
                    # "critic_target": self.learning_role.target_critics[u_id].state_dict(),
                    "critic_optimizer": self.learning_role.critics[
                        u_id
                    ].optimizer.state_dict(),
                }
                path = f"{directory}/critic_{u_id}.pt"
                th.save(obj, path)

    # Removed actor_target in comparison to MATD3
    def save_actor_params(self, directory):
        """
        Save the parameters of actor networks.

        This method saves the parameters of the actor networks, including the actor's state_dict, actor_target's state_dict, and
        the actor's optimizer state_dict. It organizes the saved parameters into a directory structure specific to the actor
        associated with each learning strategy.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        os.makedirs(directory, exist_ok=True)
        for u_id in self.learning_role.rl_strats.keys():
            obj = {
                "actor": self.learning_role.rl_strats[u_id].actor.state_dict(),
                # "actor_target": self.learning_role.rl_strats[
                #     u_id
                # ].actor_target.state_dict(),
                "actor_optimizer": self.learning_role.rl_strats[
                    u_id
                ].actor.optimizer.state_dict(),
            }
            path = f"{directory}/actor_{u_id}.pt"
            th.save(obj, path)

    # Unchanged method from MATD3
    def load_params(self, directory: str) -> None:
        """
        Load the parameters of both actor and critic networks.

        This method loads the parameters of both the actor and critic networks associated with the learning role from the specified
        directory. It uses the `load_critic_params` and `load_actor_params` methods to load the respective parameters.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        self.load_critic_params(directory)
        self.load_actor_params(directory)



    # Centralized
    def load_critic_params(self, directory: str) -> None:
        """
        Load the parameters of critic networks from a specified directory.

        This method loads the parameters of critic networks, including the critic's state_dict, critic_target's state_dict, and
        the critic's optimizer state_dict, from the specified directory. It iterates through the learning strategies associated
        with the learning role, loads the respective parameters, and updates the critic and target critic networks accordingly.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        logger.info("Loading critic parameters...")

        if not os.path.exists(directory):
            logger.warning(
                "Specified directory for loading the critics does not exist! Starting with randomly initialized values!"
            )
            return
        if self.share_critic:
            try:
                critic_params = self.load_obj(f"{directory}/critics/shared_critic.pt")
                self.shared_critic.load_state_dict(critic_params["critic"])
                self.shared_critic.optimizer.load_state_dict(critic_params["critic_optimizer"])
            except Exception:
                logger.warning("No critic values loaded")
        else:
            for u_id in self.learning_role.rl_strats.keys():
                try:
                    critic_params = self.load_obj(
                        directory=f"{directory}/critics/critic_{str(u_id)}.pt"
                    )
                    self.learning_role.critics[u_id].load_state_dict(
                        critic_params["critic"]
                    )
                    self.learning_role.critics[u_id].optimizer.load_state_dict(
                        critic_params["critic_optimizer"]
                    )
                except Exception:
                    logger.warning(f"No critic values loaded for agent {u_id}")

    # Removed actor_target in comparison to MATD3
    def load_actor_params(self, directory: str) -> None:
        """
        Load the parameters of actor networks from a specified directory.

        This method loads the parameters of actor networks, including the actor's state_dict, actor_target's state_dict, and
        the actor's optimizer state_dict, from the specified directory. It iterates through the learning strategies associated
        with the learning role, loads the respective parameters, and updates the actor and target actor networks accordingly.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        logger.info("Loading actor parameters...")
        if not os.path.exists(directory):
            logger.warning(
                "Specified directory for loading the actors does not exist! Starting with randomly initialized values!"
            )
            return

        for u_id in self.learning_role.rl_strats.keys():
            try:
                actor_params = self.load_obj(
                    directory=f"{directory}/actors/actor_{str(u_id)}.pt"
                )
                self.learning_role.rl_strats[u_id].actor.load_state_dict(
                    actor_params["actor"]
                )
                # self.learning_role.rl_strats[u_id].actor_target.load_state_dict(
                #     actor_params["actor_target"]
                # )
                self.learning_role.rl_strats[u_id].actor.optimizer.load_state_dict(
                    actor_params["actor_optimizer"]
                )
            except Exception:
                logger.warning(f"No actor values loaded for agent {u_id}")


    # Centralized
    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Create actor and critic networks for reinforcement learning.

        If `actors_and_critics` is None, this method creates new actor and critic networks.
        If `actors_and_critics` is provided, it assigns existing networks to the respective attributes.

        Args:
            actors_and_critics (dict): The actor and critic networks to be assigned.

        """
        if actors_and_critics is None:
            self.create_actors()
            if self.share_critic:
                    self.create_shared_critic()
                    print("created shared critic")
            else:
                self.create_critics()

        else:
            if self.share_critic:
                self.shared_critic = actors_and_critics["shared_critic"]
            else: 
                self.learning_role.critics = actors_and_critics["critics"]

            # self.learning_role.target_critics = actors_and_critics["target_critics"]
            for u_id, unit_strategy in self.learning_role.rl_strats.items():
                unit_strategy.actor = actors_and_critics["actors"][u_id]
                # unit_strategy.actor_target = actors_and_critics["actor_targets"][u_id]
                
            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

            #carry over value_stats from previous episode
            #
            if self.keep_value_stats:
                self.value_stats = actors_and_critics.get("value_stats", {})

        # fill value_stats in first episode or for new agents
        for u_id in self.learning_role.rl_strats.keys():
            if u_id not in self.value_stats:
                self.value_stats[u_id] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'updated': False
                }
        
        print(self.value_stats)

    # Removed actor_target in comparison to MATD3
    def create_actors(self) -> None:
        """
        Create actor networks for reinforcement learning for each unit strategy.

        This method initializes actor networks and their corresponding target networks for each unit strategy.
        The actors are designed to map observations to action probabilities in a reinforcement learning setting.

        The created actor networks are associated with each unit strategy and stored as attributes.

        Notes:
            The observation dimension need to be the same, due to the centralized criic that all actors share.
            If you have units with different observation dimensions. They need to have different critics and hence learning roles.

        """

        obs_dim_list = []
        act_dim_list = []

        actor_optimizers = []

        for _, unit_strategy in self.learning_role.rl_strats.items():
            unit_strategy.actor = self.actor_architecture_class(
                obs_dim=unit_strategy.obs_dim,
                act_dim=unit_strategy.act_dim,
                float_type=self.float_type,
                learn_std = self.learn_std,
                unique_obs_dim=unit_strategy.unique_obs_dim,
                num_timeseries_obs_dim=unit_strategy.num_timeseries_obs_dim,
            ).to(self.device)

            unit_strategy.actor.optimizer = Adam(
                unit_strategy.actor.parameters(), lr=self.learning_role.calc_lr_from_progress(1), weight_decay=1e-4
            )

            actor_optimizers.append(unit_strategy.actor.optimizer)   

            obs_dim_list.append(unit_strategy.obs_dim)
            act_dim_list.append(unit_strategy.act_dim)


        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                "All observation dimensions must be the same for all RL agents"
            )
        else:
            self.obs_dim = obs_dim_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError("All action dimensions must be the same for all RL agents")
        else:
            self.act_dim = act_dim_list[0]

    # Removed target_critics in comparison to MATD3
    # Changed initialization of CriticPPO compared to MATD3
    # Decentralized
    # def create_critics(self) -> None:
    #     """
    #     Create decentralized critic networks for reinforcement learning.

    #     This method initializes a separate critic network for each agent in the reinforcement learning setup.
    #     Each critic learns to predict the value function based on the individual agent's observation.

    #     Notes:
    #         Each agent has its own critic, so the critic is no longer shared among all agents.
    #     """

    #     unique_obs_dim_list = []
    #     n_agents = len(self.learning_role.rl_strats)

    #     for _, unit_strategy in self.learning_role.rl_strats.items():
    #         unit_strategy.critic = CriticPPO(
    #             obs_dim=unit_strategy.obs_dim,
    #             float_type=self.float_type,
    #             n_agents=n_agents,
    #             act_dim=unit_strategy.act_dim
    #         ).to(self.device)

    #         unit_strategy.critic.optimizer = Adam(
    #             unit_strategy.critic.parameters(), lr=self.learning_rate
    #         )

    #         unique_obs_dim_list.append(unit_strategy.unique_obs_dim)

    #     # Check if all unique_obs_dim are the same and raise an error if not
    #     # If they are all the same, set the unique_obs_dim attribute
    #     if len(set(unique_obs_dim_list)) > 1:
    #         raise ValueError(
    #             "All unique_obs_dim values must be the same for all RL agents"
    #         )
    #     else:
    #         self.unique_obs_dim = unique_obs_dim_list[0]


    def create_shared_critic(self) -> None:
        """
        Create a centralized critic network that will be used by all agents.
        I want to try how this works out, might delete later
        """
        # TODO: test and delete later
        n_agents = len(self.learning_role.rl_strats)
        strategy = next(iter(self.learning_role.rl_strats.values())) # get first strategy for obs/act_dim

        # create one shared critic for all agents
        self.shared_critic = CriticPPO(
            n_agents=n_agents,
            obs_dim=strategy.obs_dim,
            act_dim=strategy.act_dim,
            unique_obs_dim=strategy.unique_obs_dim,
            float_type=self.float_type,
            public_info=self.public_info,
        ).to(self.device)

        self.shared_critic.optimizer = Adam(
            self.shared_critic.parameters(),
            lr=self.learning_role.calc_lr_from_progress(1),
            weight_decay=1e-4
)

        # create single optimizer for shared critic

        self.unique_obs_dim = strategy.unique_obs_dim


    # Centralized
    def create_critics(self) -> None:
        """
        Create decentralized critic networks for reinforcement learning.

        This method initializes a separate critic network for each agent in the reinforcement learning setup.
        Each critic learns to predict the value function based on the individual agent's observation.

        Notes:
            Each agent has its own critic, so the critic is no longer shared among all agents.
        """

        n_agents = len(self.learning_role.rl_strats)
        strategy: LearningStrategy
        unique_obs_dim_list = []
        critic_optimizers = []

        for u_id, strategy in self.learning_role.rl_strats.items():
            self.learning_role.critics[u_id] = CriticPPO(
                n_agents=n_agents,
                obs_dim=strategy.obs_dim,
                act_dim=strategy.act_dim,
                unique_obs_dim=strategy.unique_obs_dim,
                float_type=self.float_type,
                public_info=self.public_info,
            )

            self.learning_role.critics[u_id].optimizer = Adam(
                self.learning_role.critics[u_id].parameters(), lr=self.learning_role.calc_lr_from_progress(1), weight_decay=1e-4
            )

            self.learning_role.critics[u_id] = self.learning_role.critics[u_id].to(
                self.device
            )
            critic_optimizers.append(self.learning_role.critics[u_id].optimizer)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
        


        # check if all unique_obs_dim are the same and raise an error if not
        # if they are all the same, set the unique_obs_dim attribute
        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                "All unique_obs_dim values must be the same for all RL agents"
            )
        else:
            self.unique_obs_dim = unique_obs_dim_list[0]


    # Centralized
    def extract_policy(self) -> dict:
        """
        Extract actor and critic networks.

        This method extracts the actor and critic networks associated with each learning strategy and organizes them into a
        dictionary structure. The extracted networks include actors, and critics. The resulting
        dictionary is typically used for saving and sharing these networks.

        Returns:
            dict: The extracted actor and critic networks.
        """
        actors = {}

        for u_id, unit_strategy in self.learning_role.rl_strats.items():
            actors[u_id] = unit_strategy.actor

        actors_and_critics = {
            "actors": actors,
            
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
            "value_stats": self.value_stats
        }
        if self.share_critic:
            actors_and_critics["shared_critic"] = self.shared_critic
            print("extracting shared critics")
        else:
            actors_and_critics["critics"] = self.learning_role.critics
            print("extracting individual critics")
        print(self.value_stats)
        return actors_and_critics
    
    def set_wandb_step(self, step: int) -> None:
            """Set the wandb step counter to continue from a loaded state."""
            self.wandb_step = step
    
    def get_values(self, all_states, all_actions, u_id=None, i=None, individual_values=False):
        """
        Gets values for a unit based on the observation using PPO. 
        Denormalizes critic outputs using running statistics. Induces critic to learn normalized value predictions

        Args:
            rl_strategy (RLStrategy): The strategy containing relevant information.
            next_observation (torch.Tensor): The observation.

        Returns:
            torch.Tensor: The value of the observation.
        """
        buffer_length = len(all_states) # get length of all states to pass it on as batch size, since the entire buffer is used for the PPO

        if individual_values:
            if self.public_info:
                select_states = collect_obs_for_central_critic(all_states, i, self.obs_dim, self.unique_obs_dim, buffer_length)
                select_actions = all_actions.view(buffer_length, -1).contiguous()
            else: 
                select_states  = all_states[:,i,:].reshape(buffer_length, -1) 
                select_actions = all_actions[:,i,:].view(buffer_length,-1)

            values = self.learning_role.critics[u_id](select_states, select_actions).squeeze()

        else: 
            i=0 # counter iterating over all agents for dynamic buffer slice
            n_agents = len(self.learning_role.rl_strats)
            values = th.zeros((buffer_length,n_agents), device=self.device)

            for i,u_id in enumerate(self.learning_role.rl_strats.keys()):

                # prepare states and actions depending on which information critic should see
                if self.public_info: 
                    select_states = collect_obs_for_central_critic(all_states, i, self.obs_dim, self.unique_obs_dim, buffer_length)
                    select_actions = all_actions.view(buffer_length, -1).contiguous() # critic sees all information
                else: 
                    select_states  = all_states[:,i,:].reshape(buffer_length, -1) 
                    select_actions = all_actions[:,i,:].view(buffer_length,-1) # critic does not see others private information

                # Pass the appropriate states and actions to shared or individual critics
                if self.share_critic:
                    values[:,i] = self.shared_critic(select_states, select_actions).squeeze()
                else:
                    #values[:,i] = self.denormalize_values(self.learning_role.critics[u_id](select_states, select_actions).squeeze(), u_id)
                    values[:,i] = self.learning_role.critics[u_id](select_states, select_actions).squeeze()
                
        return values
    

    def get_advantages(self, rewards, values):

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = th.zeros_like(rewards)
        returns = th.zeros_like(rewards)
        last_gae = 0

        # Iterate through the collected experiences in reverse order to calculate advantages and returns
        for t in reversed(range(len(rewards))):
            
            logger.debug(f"Reward: {t}")    
            next_value = values[t+1] if t < len(rewards) -1 else 0

            # Temporal difference delta Equation 12 from PPO paper
            delta = rewards[t] + self.gamma * next_value - values[t] # Use self.gamma for discount factor
            logger.debug(f"Delta: {delta}")

            # GAE advantage Equation 11 from PPO paper
            advantages[t] = delta + self.gamma * self.gae_lambda * last_gae # Use self.gae_lambda for advantage estimation
            last_gae = advantages[t]
            logger.debug(f"Last_advantage: {last_gae}")
            returns[t] = advantages[t] + values[t]

        #Normalize advantages in accordance with spinning up and mappo version of PPO
        mean_advantages = th.nanmean(advantages)
        std_advantages = th.std(advantages)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        #TODO: Should we detach here? I though because of normalisation not being included in backward
        # but unsure if this is correct
        return advantages, returns


    def update_policy(self):
        """
        Perform policy updates using PPO with the clipped objective.
        """

        logger.debug("Updating Policy")
        # We will iterate for multiple epochs to update both the policy (actor) and value (critic) networks
        # The number of epochs controls how many times we update using the same collected data (from the buffer).

        learning_rate = self.learning_role.calc_lr_from_progress(
            self.learning_role.get_progress_remaining()
        )
        # pass new learning rate to all actors and shared or individual critics TODO: make more concise!
        if not self.share_critic:
            for u_id, unit_strategy in self.learning_role.rl_strats.items():
                self.update_learning_rate(
                    [
                        self.learning_role.critics[u_id].optimizer,
                        self.learning_role.rl_strats[u_id].actor.optimizer,
                    ],
                    learning_rate=learning_rate,
                )
        else: 
            self.update_learning_rate([self.shared_critic.optimizer], learning_rate)
            for u_id, unit_strategy in self.learning_role.rl_strats.items():
                self.update_learning_rate(
                    [self.learning_role.rl_strats[u_id].actor.optimizer],
                    learning_rate=learning_rate,
                )
            
        # TODO: maybe update entropy coefficient ?!   
        # Retrieve experiences from the buffer
        # The collected experiences (observations, actions, rewards, log_probs) are stored in the buffer.
        transitions = self.learning_role.buffer.get()
        
        with th.no_grad():
            # Pass the current states through the critic network to get value estimates.
            full_values = self.get_values(transitions.observations, transitions.actions)
            #print(f"shape full_values : {full_values.shape}")
            # Compute advantages using Generalized Advantage Estimation (GAE)
            full_advantages, full_returns = self.get_advantages(transitions.rewards, full_values)
            
            #update running statistics for value normalization
            if not self.share_critic:
                self.update_value_normalizer(full_returns)

            value_loss = None
        
        states = transitions.observations
        actions = transitions.actions
        log_probs = transitions.log_probs
        advantages = full_advantages
        

        for counter in range(self.gradient_steps):
            self.n_updates += 1
            
            # always use updated values --> check later if best
            # Iterate through over each agent's strategy
            # Each agent has its own actor. Critic (value network) is centralized.
            for i, u_id in enumerate(self.learning_role.rl_strats.keys()):
                #calculate critic loss over own values or all
                if self.individual_values: 
                    values = self.get_values(states, actions, u_id, i, self.individual_values)
                    returns = full_returns[:,i]
                    old_values = full_values[:,i]
                else: 
                    values = self.get_values(states, actions)
                    returns = full_returns
                    old_values = full_values
                
                #print(f"values shape {values.shape}, \nreturn shape {returns.shape}, \nold_values shape {old_values.shape}")
                #print(f"values.shape {values.shape}")

                # Centralized
                if not self.share_critic:
                    critic = self.learning_role.critics[u_id]
                else: 
                    critic = self.shared_critic
                
                # Decentralized
                actor = self.learning_role.rl_strats[u_id].actor
                
                # Evaluate the new log-probabilities and entropy under the current policy
                state_i = states[:, i, :]      # Shape: (batch, obs_dim)
                actions_i = actions[:, i, :]   # Shape: (batch, action_dim)
                log_probs_i = log_probs[:, i]  # Shape: (batch)
                advantages_i = advantages[:, i] # Shape: (batch)
                returns_i = returns if self.individual_values else returns[:,i]
                #print(state_i[0, -1])
                base_bid = None
                if self.use_base_bid:
                    base_bid = state_i[0, -1].detach()
                action_distribution = actor(state_i, base_bid)[1]
                # if counter == 0:
                #     print(f" u_id: {u_id}, std: {action_distribution.stddev[0]}")
                new_log_probs = action_distribution.log_prob(actions_i).sum(-1)
                
                entropy = action_distribution.entropy().sum(-1)

                # Compute the ratio of new policy to old policy
                ratio = (new_log_probs - log_probs_i).exp()
                logger.debug(f"Ratio: {ratio}")

                # Surrogate loss calculation
                surrogate1 = ratio * advantages_i
                surrogate2 = (th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_i)  # Use self.clip_ratio
                logger.debug(f"surrogate1: {surrogate1}")
                logger.debug(f"surrogate2: {surrogate2}")

                # Final policy loss (clipped surrogate loss) equation 7 from PPO paper
                policy_loss = -th.min(surrogate1, surrogate2).mean()
                logger.debug(f"policy_loss: {policy_loss}")

                # Value loss (mean squared error between the predicted values and returns)
                if (not self.share_critic) or (self.share_critic == True and i == 0):
                    value_loss = F.mse_loss(returns.squeeze(), values.squeeze())

                    if self.value_clip_ratio is not None:
                        values_clipped = old_values + th.clamp(values - old_values, -self.value_clip_ratio, self.value_clip_ratio)
                        clipped_value_loss = F.mse_loss(returns.squeeze(), values_clipped.squeeze())

                        # use maximum between unclipped and clipped as actual value loss
                        value_loss = th.max(clipped_value_loss, value_loss)
                    
                    critic.optimizer.zero_grad()
                    value_loss.backward()
                    th.nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                    critic.optimizer.step()
                    logger.debug(f"value loss: {value_loss}")

                # Total loss: policy loss + value loss - entropy bonus
                # euqation 9 from PPO paper multiplied with -1 to enable minimizing
                # total_loss = (
                #      policy_loss # added minus in policy loss calculation already
                #     + self.vf_coef * value_loss
                #     #- self.entropy_coef * entropy.mean()
                # )  # Use self.vf_coef and self.entropy_coef
                actor_loss = policy_loss #- self.entropy_coef * entropy.mean()
                #logger.debug(f"total loss: {total_loss}")

                # Zero the gradients and perform backpropagation for both actor and critic
                actor.optimizer.zero_grad()
                actor_loss.backward()
                # Clip gradients to prevent gradient explosion
                th.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)  # Use self.max_grad_norm
                # Perform optimization step
                actor.optimizer.step()

                # wandb logging for comparisons
                if self.learning_role.learning_mode and wandb.run is not None:
                    try:
                        with th.no_grad():
                            # Store per-agent metrics
                            self.agent_metrics = getattr(self, 'agent_metrics', {})
                            if i == 0:  # Initialize/reset aggregate metrics
                                self.agent_metrics = {
                                    'policy_loss': [],
                                    'value_loss': [],
                                    'entropy': [],
                                    'mean_ratio': [],
                                    'mean_action': [],
                                    'action_std': [],
                                    'returns': [],
                                    'values': [],
                                    'advantages': [],
                                    'rewards': []
                                }
                            
                            # Collect metrics for current agent
                            agent_rewards = transitions.rewards[:, i]
                            self.agent_metrics['policy_loss'].append(policy_loss.item())
                            self.agent_metrics['value_loss'].append(value_loss.item() if value_loss else 0)
                            self.agent_metrics['entropy'].append(entropy.mean().item())
                            self.agent_metrics['mean_ratio'].append(ratio.mean().item())
                            self.agent_metrics['mean_action'].append(actions_i.mean().item())
                            self.agent_metrics['action_std'].append(actions_i.std().item())
                            self.agent_metrics['returns'].append(returns_i.mean().item())
                            self.agent_metrics['values'].append(values.mean().item())
                            self.agent_metrics['advantages'].append(advantages_i.mean().item())
                            self.agent_metrics['rewards'].append(agent_rewards.mean().item())

                            # Log individual agent metrics
                            agent_specific = {
                                f"agents/{u_id}/policy_loss": policy_loss.item(),
                                f"agents/{u_id}/value_loss": value_loss.item() if value_loss else 0,
                                f"agents/{u_id}/entropy": entropy.mean().item(),
                                f"agents/{u_id}/mean_ratio": ratio.mean().item(),
                                f"agents/{u_id}/mean_action": actions_i.mean().item(),
                                f"agents/{u_id}/action_std": actions_i.std().item(),
                                f"agents/{u_id}/mean_returns": returns_i.mean().item(),
                                f"agents/{u_id}/mean_values": values.mean().item(),
                                f"agents/{u_id}/mean_advantages": advantages_i.mean().item(),
                                f"agents/{u_id}/mean_reward": agent_rewards.mean().item(),
                                f"agents/{u_id}/cumulative_reward": agent_rewards.sum().item(),
                            }

                            # If this is the last agent, calculate and log aggregate metrics
                            if i == len(self.learning_role.rl_strats) - 1:
                                aggregate_metrics = {
                                    "training/mean_policy_loss": th.tensor(self.agent_metrics['policy_loss']).mean().item(),
                                    "training/mean_value_loss": th.tensor(self.agent_metrics['value_loss']).mean().item(),
                                    "training/mean_entropy": th.tensor(self.agent_metrics['entropy']).mean().item(),
                                    "training/mean_ratio": th.tensor(self.agent_metrics['mean_ratio']).mean().item(),
                                    "training/learning_rate": learning_rate,
                                }

                                # Core Training Progress for all agents
                                training_metrics = {
                                    "training/unclipped_value_loss": clipped_value_loss.item() if value_loss else 0,
                                }

                                # Episode Performance
                                rewards_tensor = transitions.rewards.clone().detach()
                                episode_metrics = {
                                    "episode/cumulative_reward": rewards_tensor.sum().item(),
                                    "episode/mean_reward": rewards_tensor.mean().item(),
                                    "episode/mean_reward_per_agent": th.tensor(self.agent_metrics['rewards']).mean().item(),
                                    "episode/episode_num": self.learning_role.episodes_done,
                                }

                                # values_flat = values.squeeze()
                                # td_errors = values_flat - returns[:,i]

                                # # Calculate correlation
                                # v_mean = values_flat.mean()
                                # v_std = values_flat.std()
                                # r_mean = returns.mean()
                                # r_std = returns.std()
                                # v_centered = values_flat - v_mean
                                # r_centered = returns - r_mean
                                # correlation = (v_centered * r_centered).mean() / (v_std * r_std + 1e-8)

                                # value_metrics = {
                                #     "value/mean_td_error": td_errors.mean().item(),
                                #     "value/explained_variance": 1 - (td_errors.var() / (returns.var() + 1e-8)).item(),
                                #     "value/predicted_mean": values_flat.mean().item(),
                                #     "value/actual_returns_mean": returns.mean().item(),
                                #     "value/max_difference": td_errors.abs().max().item(),
                                #     "value/correlation": correlation.item(),
                                # }

                                # Policy Updates
                                policy_metrics = {
                                    "policy/mean_action_all_agents": th.tensor(self.agent_metrics['mean_action']).mean().item(),
                                    "policy/action_std_all_agents": th.tensor(self.agent_metrics['action_std']).mean().item(),
                                }

                                # Training Stability
                                stability_metrics = {
                                    "stability/advantage_mean": th.tensor(self.agent_metrics['advantages']).mean().item(),
                                    "stability/advantage_std": advantages.std().item(),
                                }

                                # Combine all metrics
                                metrics = {}
                                metrics.update(agent_specific)
                                metrics.update(aggregate_metrics)
                                metrics.update(training_metrics)
                                metrics.update(episode_metrics)
                                #metrics.update(value_metrics)
                                metrics.update(policy_metrics)
                                metrics.update(stability_metrics)

                                # Log everything in a single call at the end of processing all agents
                                self.wandb_step = self.wandb_step + 1
                                self.learning_role.wandb_step = self.wandb_step
                                wandb.log(metrics, step=self.wandb_step)
                    except Exception as e:
                        logger.error(f"Failed to log metrics to WandB: {e}", exc_info=True)
                        print(f"Warning: Failed to log metrics: {str(e)}")
                                
                
            
    
    def update_value_normalizer(self, targets):
        """
        Update running statistics for value normalization per agent using an exponential moving average.
        Args:
            targets: Tensor of shape (buffer_size, n_agents) containing returns
        """
        for i, u_id in enumerate(self.learning_role.rl_strats.keys()):
            if not self.value_stats[u_id]['updated']:
                self.value_stats[u_id]['mean'] = targets[:, i].mean().item()
                self.value_stats[u_id]['std'] = targets[:, i].std().item()
                self.value_stats[u_id]['updated'] = True
            else:
                # Exponential moving average update for each agent
                self.value_stats[u_id]['mean'] = 0.98 * self.value_stats[u_id]['mean'] + 0.02 * targets[:, i].mean().item()
                self.value_stats[u_id]['std'] = 0.98 * self.value_stats[u_id]['std'] + 0.02 * targets[:, i].std().item()


    def denormalize_values(self, normalized_values, u_id):
        """
        Denormalize critic outputs using the running mean and standard deviation of returns for a specific agent
        Args:
            normalized_values: Tensor of normalized values from critic
            u_id: Agent identifier for accessing correct statistics
        Returns:
            Tensor of denormalized values
        """
        return self.value_stats[u_id]['mean'] + normalized_values * self.value_stats[u_id]['std']
    
def get_actions(rl_strategy, next_observation):
    """
    Gets actions for a unit based on the observation using PPO.

    Args:
        rl_strategy (RLStrategy): The strategy containing relevant information.
        next_observation (torch.Tensor): The observation.

    Returns:
        torch.Tensor: The sampled actions.
        torch.Tensor: The log probability of the sampled actions.
    """
    logger.debug("ppo.py: Get_actions method")

    actor = rl_strategy.actor
    device = rl_strategy.device
    learning_mode = rl_strategy.learning_mode
    perform_evaluation = rl_strategy.perform_evaluation
    base_bid = None
    # Pass observation through the actor network to get action logits (mean of action distribution)
    if False:
        base_bid = next_observation[-1]

    action_logits, action_distribution = actor(next_observation, base_bid)

    action_logits = action_logits.detach()
    logger.debug(f"Action logits: {action_logits}")

    logger.debug(f"Action distribution: {action_distribution}")

    if learning_mode and not perform_evaluation:

        # Sample an action from the distribution
        sampled_action = action_distribution.sample().to(device)

    else:
        # If not in learning mode or during evaluation, use the mean of the action distribution
        sampled_action = action_logits.detach()

    logger.debug(f"Sampled action: {sampled_action}")

    # Get the log probability of the sampled actions (for later PPO loss calculation)
    # Sum the log probabilities across all action dimensions TODO: Why sum?
    log_prob_action = action_distribution.log_prob(sampled_action).sum(dim=-1)

    # Detach the log probability tensor to stop gradient tracking (since we only need the value for later)
    log_prob_action = log_prob_action.detach()

    logger.debug(f"Detached log probability of the sampled action: {log_prob_action}")


    return sampled_action, log_prob_action

