import torch
from torch import nn
from torch.distributions.categorical import Categorical
from math import prod
import gin
import gym
from gym.spaces import Discrete, Box
import numpy as np
from torch.optim import Adam


def mlp(obs_dim, hidden_size, n_acts):
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_acts),
    )


def train(env_name, hidden_size=32, lr=1e-2, epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    # make policy network
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    logits_net = mlp(obs_dim=obs_dim, hidden_size=hidden_size, n_acts=n_acts)

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs = env.reset()
        done = False
        ep_rews = []

        finished_rendering_this_epoch = False

        while True:
            if render and (not finished_rendering_this_epoch):
                env.render()

            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += [ep_ret] * ep_len

                obs, ep_rews, done = env.reset(), [], False
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # main training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )


if __name__ == "__main__":
    train("CartPole-v1")
