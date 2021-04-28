import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch

import torch.nn.functional as F
import torch.optim as optim

from model_maddpg_1BN import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_FREQ = 1        # Update the model parameters every 1 in UPDATE_FREQ cycles
UPDATE_NUM = 1         # Number of model parameter updates per cycle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('\n model is model_maddpg_1BN')

class multi_Agent:
    """Mult-agent ddpg class of objects that interacts and learns
       from a multi-agent environment
    """
    def __init__(self, state_size, action_size, random_seed):
        """Initializes agents for mutli-agent environment
           Params
           ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # Use this time step to keep track of when to learn, initialize to zero
        self.time_step = 0
        # Instantiate the two ddpg agents to interact with the environment
        self.agent1 = Agent(self.state_size, self.action_size, random_seed)
        self.agent2 = Agent(self.state_size, self.action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory for both agents."""
        # Save experience / rewards
        # Breaking out each agent's state, action, next_state for
        # easier application to actor and critic models
        self.memory.add(states[0], states[1], actions[0], actions[1], rewards,
                        next_states[0], next_states[1], dones)

        # No learning until we have enough samples available in memory
        if len(self.memory) > BATCH_SIZE:

            # Use the time_step to keep track of when the next learning cycle takes place
            self.time_step += 1
            # Roll over the time step at the desired frequency for learning
            self.time_step %= UPDATE_FREQ

            # Learn every 1 in UPDATE_FREQ cycles
            if self.time_step == 0:

                # Learn UPDATE_NUM times each cycle
                ii = 0
                while ii < UPDATE_NUM:
                    experiences = self.memory.sample()
                    self.agent1.learn(experiences, GAMMA, agent_id=0)

                    experiences = self.memory.sample()
                    self.agent2.learn(experiences, GAMMA, agent_id=1)

                    ii += 1

    def act(self, state, add_noise):
        """Returns multi-agent actions for given state as per current policy."""
        return [self.agent1.act(state[0], add_noise),
                self.agent2.act(state[1], add_noise)]

    def reset(self):
        """Resets both agents"""
        self.agent1.reset()
        self.agent2.reset()

class Agent():
    """Indirectly interacts with and learns from the environment through
        the multi_Agent object.
    """

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        #random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)

        # Initialize the actor_target with actor_local's parameters
        # Adapted from:
        # https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/DDPG_agent.py
        self.actor_target.load_state_dict(self.actor_local.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)

        # Initialize the critic_target with critic_local's parameters
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        """Agent resets go here"""
        self.noise.reset()

    def learn(self, experiences, gamma, agent_id):
        """Two agent version: Update policy and value parameters using
            given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params:
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            agent_id (int) identifies which agent this is
        """
        s_a1, s_a2, act_a1, act_a2, rewards, ns_a1, ns_a2, dones = experiences
        # Need to convert to tensor to use it in torch.cat
        agent_id = torch.tensor([agent_id]).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # combine next states from both agents to use with critic models
        next_states = torch.cat((ns_a1, ns_a2), dim=1)
        # Grap next actions using each agent's next states individually
        a1_actions_next = self.actor_target(ns_a1)
        a2_actions_next = self.actor_target(ns_a2)
        # Combine next actions from both agent's next_states
        actions_next = torch.cat((a1_actions_next, a2_actions_next), dim=1)
        # Apply next_states and next_actions from both agents to critic target
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        # Need to select the rewards and dones associated with the current agent
        rewards = rewards.index_select(1, agent_id)
        dones = dones.index_select(1, agent_id)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Combine agent1 and agent2 states and actions, respectively
        states = torch.cat((s_a1, s_a2), dim=1)
        actions = torch.cat((act_a1, act_a2), dim=1)
        Q_expected = self.critic_local(states, actions)
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Combine predicted actions from each agents states
        actions_pred_a1 = self.actor_local(s_a1)
        actions_pred_a2 = self.actor_local(s_a2)
        actions_pred = torch.cat((actions_pred_a1, actions_pred_a2), dim=1)
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        np.random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample.
           Using np.random.randn() which is normal distribution with mean 0 and variance 1
           instead of random.random() which is uniform distribution between 0 and 1
           random.randn() is used in:
           https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/random_process.py
           and yields better results
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples for both agents."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int) for consistent random selections across runs
        """
        random.seed(seed)
        # self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_a1",
                                    "state_a2", "action_a1", "action_a2",
                                    "reward", "next_state_a1", "next_state_a2",
                                    "done"])

    def add(self, state_a1, state_a2, action_a1, action_a2, reward,
            next_state_a1, next_state_a2, done):
        """Add a new experience to memory."""
        e = self.experience(state_a1, state_a2, action_a1, action_a2, reward,
                            next_state_a1, next_state_a2, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states_a1 = torch.from_numpy(np.vstack([e.state_a1 for \
                    e in experiences if e is not None])).float().to(device)
        states_a2 = torch.from_numpy(np.vstack([e.state_a2 for \
                    e in experiences if e is not None])).float().to(device)
        actions_a1 = torch.from_numpy(np.vstack([e.action_a1 for \
                     e in experiences if e is not None])).float().to(device)
        actions_a2 = torch.from_numpy(np.vstack([e.action_a2 for \
                     e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for \
                  e in experiences if e is not None])).float().to(device)
        next_states_a1 = torch.from_numpy(np.vstack([e.next_state_a1 for \
                         e in experiences if e is not None])).float().to(device)
        next_states_a2 = torch.from_numpy(np.vstack([e.next_state_a2 for \
                         e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for \
                e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_a1, states_a2, actions_a1, actions_a2, rewards,
                next_states_a1, next_states_a2, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
