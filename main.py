import math
import random
from collections import namedtuple, deque
from itertools import count

import gym
import matplotlib.pyplot as plt
import torch

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 30
TAU = 0.005
LR = 1e-4

env = gym.make("CartPole-v1")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clean(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class AgentNN(torch.nn.Module):  # DQL module
    def __init__(self, n_observations, n_actions):
        super(AgentNN, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(n_observations + n_actions, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        :param observation: a ndarray representing the current state.
        :param action: an int.
        :return: expected return.
        """
        inin = torch.cat((observation, action), dim=1)

        return self.module(inin)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def peek_action(steps_done: int, state: torch.Tensor) -> torch.Tensor:
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # eps_threshold = min(1.0, EPS_END + (EPS_START - EPS_END) / pow(math.log(steps_done+2), 3))
    eps_threshold = min(1.0, (EPS_START - EPS_END) / pow(math.log(steps_done+2), 3))
    if sample > eps_threshold:
        with torch.no_grad():
            left = torch.Tensor([[1, 0]])
            right = torch.Tensor([[0, 1]])

            out = policy_net.forward(
                torch.cat((state, state), dim=0),
                torch.cat((left, right), dim=0)
            ).argmax()

            # if out is 0 then them going left is better and if out is 1 then going right is better
            return_out = torch.zeros((1, 2), dtype=torch.float32)
            return_out[0][out] = 1

            return return_out
    else:
        return_out = torch.zeros((1, 2), dtype=torch.float32)
        return_out[0][env.action_space.sample()] = 1

        return return_out


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)

    state_action_values = policy_net(state_batch, action_batch)

    go_to_the_left_actions = torch.cat((torch.ones(non_final_next_states.shape[0], 1), torch.zeros(non_final_next_states.shape[0], 1)), dim=1)
    go_to_the_right_actions = torch.cat((torch.zeros(non_final_next_states.shape[0], 1), torch.ones(non_final_next_states.shape[0], 1)), dim=1)

    next_state_values = torch.zeros(BATCH_SIZE, 1)
    with torch.no_grad():
        value_for_left = target_net(non_final_next_states, go_to_the_left_actions)
        value_for_right = target_net(non_final_next_states, go_to_the_right_actions)

        my_max = torch.max(value_for_left, value_for_right)
        next_state_values[non_final_mask] = my_max

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = AgentNN(n_observations, n_actions)
target_net = AgentNN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 600

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    step_done = 0

    for t in count():
        action = peek_action(step_done, state)
        step_done += 1
        observation, reward, terminated, truncated, _ = env.step((action == 1).squeeze(0).nonzero().item())
        reward = torch.tensor([reward])
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if len(memory) > 9000:
            memory.clean()

        if len(episode_durations) > 1000:
            episode_durations.clear()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

torch.save({
    "my_agent": policy_net.state_dict()
}, "my_agent_file.pt")

env = gym.make("CartPole-v1", render_mode="human")

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    step_done = 0

    for t in count():
        action = peek_action(step_done, state)
        step_done += 1
        observation, reward, terminated, truncated, _ = env.step((action == 1).squeeze(0).nonzero().item())

        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Move to the next state
        state = next_state

        if done:
            break
