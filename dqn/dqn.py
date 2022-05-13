import sys
import torch
from torch import nn
from math import prod
from collections import deque
from tqdm import tqdm
import gym
from torch.optim import Adam
import numpy as np
import gin
from einops.layers.torch import Rearrange
from atari_wrappers import AtariWrapper


def make_choice(env, eps, net, obs, device):
    if torch.rand(()) < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            expected_reward = net(torch.tensor(obs, device=device).unsqueeze(0))
            return torch.argmax(expected_reward).cpu().numpy()


def run_eval_episode(count, env, net, eps, device):
    assert count > 0
    env.reset()
    total_starting_q = 0
    total_reward = 0
    total_episode_len = 0
    for i_episode in range(count):
        obs = env.reset()
        with torch.no_grad():
            total_starting_q += (
                net(torch.tensor(obs, device=device).unsqueeze(0)).max().cpu().item()
            )
            while True:
                if not len(env.observation_space.shape) > 1:
                    env.render()
                obs, reward, done, _ = env.step(make_choice(env, eps, net, obs, device))
                total_reward += reward
                total_episode_len += 1
                if done:
                    break
    env.close()
    prefix = "" if count == 1 else "avg "
    print(f"eval {prefix}episode len: {total_episode_len/count}")
    print(f"eval {prefix}total reward: {total_reward/count}")
    print(f"eval {prefix}starting q: {total_starting_q/count}")


class PixelByteToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.to(torch.float) / 255.0


def atari_model(obs_n_channels, action_size):
    return nn.Sequential(
        Rearrange("n h w c -> n c h w"),
        PixelByteToFloat(),
        nn.Conv2d(obs_n_channels, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, action_size),
    )


class CastToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float)


def mlp_model(obs_size, action_size, hidden_size):
    return nn.Sequential(
        CastToFloat(),
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size),
    )


def get_linear_fn(start: float, end: float, end_fraction: float):
    def func(progress: float) -> float:
        if progress > end_fraction:
            return end
        else:
            return start + progress * (end - start) / end_fraction

    return func


@gin.configurable
def train_dqn(
    env_id,
    gamma,
    steps,
    start_eps,
    end_eps,
    exploration_frac,
    lr,
    train_freq,
    batch_size,
    buffer_size,
    multi_step_n,
    eval_freq,
    eval_count=1,
    start_training_step=0,
):

    # init DEVICE / env
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    eps_sched = get_linear_fn(start_eps, end_eps, exploration_frac)

    # init replay memory D to capacity N
    buffer = deque([], maxlen=buffer_size)

    # init action value function Q with random weights
    if len(env.observation_space.shape) > 1:
        eval_env = gym.make(env_id, render_mode="human")
        env = AtariWrapper(env)
        eval_env = AtariWrapper(eval_env, clip_reward=False)
        net = atari_model(
            obs_n_channels=env.observation_space.shape[-1],
            action_size=env.action_space.n,
        )
    else:
        net = mlp_model(
            obs_size=prod(env.observation_space.shape),
            action_size=env.action_space.n,
            hidden_size=64,
        )

    optim = Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # for episode = 1, M do
    # init preprocessed sequence s_1
    obs = env.reset()

    # for t = 1, T do
    for step in tqdm(range(steps), disable=False):
        # with probability eps select random action a_t
        # otherwise select a_t that maximizes Q function
        eps = eps_sched(step)
        choice = make_choice(env, eps, net, obs, DEVICE)

        # execute action a_t in emulator and observe reward r_t and image x_(t+1)
        new_obs, reward, done, _ = env.step(choice)

        # store transition in buffer
        buffer.append((obs, choice, reward, done))

        # set s_(t+1) = s_t
        obs = new_obs

        if done:
            obs = env.reset()

        if step >= start_training_step and (step + 1) % train_freq == 0:
            # sample random minibatch of (phi_j, a_j, r_j, phi_(j+1)) from D
            idxs = torch.randperm(len(buffer) - multi_step_n)[:batch_size]

            obs_batch = []
            choices = []
            rewards = []
            next_obs = []
            dones = []

            for idx in idxs:
                obs_b, choice_b, _, _ = buffer[idx]
                obs_batch.append(obs_b)
                choices.append(choice_b)
                total_reward_update = 0
                done_any = False
                for i in range(multi_step_n):
                    _, _, reward_b, done_b = buffer[idx + i]
                    total_reward_update += gamma**i + reward_b
                    if done_b:
                        done_any = True
                        break
                rewards.append(total_reward_update)
                next_obs.append(buffer[idx + multi_step_n][0])
                dones.append(done_any)

            obs_batch = torch.tensor(np.array(obs_batch), device=DEVICE)
            choices = torch.tensor(np.array(choices), device=DEVICE)
            rewards = torch.tensor(np.array(rewards), device=DEVICE, dtype=torch.float)
            next_obs = torch.tensor(np.array(next_obs), device=DEVICE)
            dones = torch.tensor(np.array(dones), device=DEVICE)

            # set targets y_j as described in paper
            with torch.no_grad():
                next_obs_actions = net(next_obs).argmax(dim=-1)
                targets = rewards + dones.logical_not() * (
                    gamma**multi_step_n
                    * net(next_obs)[torch.arange(idxs.size(0)), next_obs_actions]
                )

            # perform a gradient descent step on (y_j - Q(phi_j, a_j; theta))^2
            # according to equation 3 in paper
            actual = net(obs_batch)[torch.arange(idxs.size(0)), choices]

            loss = loss_fn(targets, actual)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if (step + 1) % eval_freq == 0:
            run_eval_episode(
                count=eval_count, env=eval_env, net=net, eps=0.05, device=DEVICE
            )

    # eval in openAI gym
    run_eval_episode(1, eval_env, net, eps=0, device=DEVICE)


if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])
    train_dqn()
