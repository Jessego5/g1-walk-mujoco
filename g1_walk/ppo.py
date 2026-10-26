"""PPO from scratch, replacing rsl_rl's OnPolicyRunner.

Re-implements rsl_rl 5.5.1's PPO semantics (same networks, normalization, GAE, clipped losses,
adaptive learning rate, timeout bootstrapping) so a run here is directly comparable with one from
train.py. Hyperparameters come from config.train_cfg().

One iteration:
  rollout: 24 steps x N robots, sample actions from a Gaussian, store everything
  -> GAE advantages -> 5 epochs x 4 minibatches of clipped PPO updates
  -> update observation normalizers -> log -> (save)
"""
import collections
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def mlp(n_in, hidden, n_out):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std of observations: (x - mean) / (std + eps). Updated explicitly, never in forward."""

    def __init__(self, dim, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + self.eps)

    @torch.no_grad()
    def update(self, x):
        # merge batch moments into the running ones (parallel variance formula)
        n = x.shape[0]
        self.count += n
        rate = n / self.count.item()
        mean_x, var_x = x.mean(0), x.var(0, unbiased=False)
        delta = mean_x - self.mean
        self.mean += rate * delta
        self.var += rate * (var_x - self.var + delta * (mean_x - self.mean))


class Actor(nn.Module):
    """Normalizer + MLP -> action mean. Kept separate so it can be exported alone as policy.pt."""

    def __init__(self, n_obs, hidden, n_act):
        super().__init__()
        self.norm = RunningNorm(n_obs)
        self.net = mlp(n_obs, hidden, n_act)

    def forward(self, obs):
        return self.net(self.norm(obs))


class ActorCritic(nn.Module):
    def __init__(self, n_actor_obs, n_critic_obs, n_act, actor_hidden, critic_hidden, init_std):
        super().__init__()
        self.actor = Actor(n_actor_obs, actor_hidden, n_act)
        self.std = nn.Parameter(init_std * torch.ones(n_act))  # state-independent, learned directly (not log-std)
        self.critic_norm = RunningNorm(n_critic_obs)
        self.critic = mlp(n_critic_obs, critic_hidden, 1)

    @staticmethod
    def split_obs(obs):
        """TensorDict from G1Env -> (actor obs, critic obs). The critic also sees the privileged base velocity."""
        return obs["policy"], torch.cat([obs["policy"], obs["privileged"]], 1)

    def dist(self, actor_obs):
        return torch.distributions.Normal(self.actor(actor_obs), self.std.clamp(min=1e-6))

    def value(self, critic_obs):
        return self.critic(self.critic_norm(critic_obs)).squeeze(-1)

    def act_inference(self, actor_obs):
        return self.actor(actor_obs)

    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.std]


def gaussian_kl(mu_old, std_old, mu_new, std_new):
    """KL(old || new) for diagonal Gaussians, summed over action dims."""
    return torch.sum(torch.log(std_new / std_old)
                     + (std_old**2 + (mu_old - mu_new) ** 2) / (2.0 * std_new**2) - 0.5, -1)


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """GAE over a (T, N) rollout. A done (fall or timeout) cuts both the bootstrap and the recursion;
    timeouts were already credited with gamma * V(s_t) in their reward during the rollout."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    adv = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + not_done * gamma * next_values - values[t]
        adv = delta + not_done * gamma * lam * adv
        advantages[t] = adv
    return advantages, advantages + values


class RolloutBuffer:
    """Preallocated (T, N, ...) tensors. Steps are copied in, so the update can backprop through them."""

    def __init__(self, T, N, n_actor_obs, n_critic_obs, n_act):
        z = lambda *shape: torch.zeros(T, N, *shape)
        self.actor_obs, self.critic_obs = z(n_actor_obs), z(n_critic_obs)
        self.actions, self.mu, self.sigma = z(n_act), z(n_act), z(n_act)
        self.log_probs, self.values, self.rewards, self.dones = z(), z(), z(), z()


class PPOTrainer:
    def __init__(self, env, cfg, log_dir=None):
        self.env, self.cfg = env, cfg
        a = cfg["algorithm"]
        self.gamma, self.lam, self.clip = a["gamma"], a["lam"], a["clip_param"]
        self.epochs, self.n_mb = a["num_learning_epochs"], a["num_mini_batches"]
        self.vf_coef, self.ent_coef, self.max_grad = a["value_loss_coef"], a["entropy_coef"], a["max_grad_norm"]
        self.desired_kl, self.lr = a["desired_kl"], a["learning_rate"]
        self.T = cfg["num_steps_per_env"]

        actor_obs, critic_obs = ActorCritic.split_obs(env.get_observations())
        self.dims = (actor_obs.shape[1], critic_obs.shape[1], env.num_actions)
        self.model = ActorCritic(*self.dims, cfg["actor"]["hidden_dims"], cfg["critic"]["hidden_dims"],
                                 cfg["actor"]["distribution_cfg"]["init_std"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buf = RolloutBuffer(self.T, env.num_envs, *self.dims)
        self.it = 0
        self.debug_checked = False

        self.log_dir = log_dir
        self.writer = None
        self.rew_hist, self.len_hist = collections.deque(maxlen=100), collections.deque(maxlen=100)
        self.cur_rew = torch.zeros(env.num_envs)
        self.cur_len = torch.zeros(env.num_envs)

    # ----------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, obs):
        b, ep_infos = self.buf, []
        for t in range(self.T):
            actor_obs, critic_obs = ActorCritic.split_obs(obs)
            dist = self.model.dist(actor_obs)
            actions = dist.sample()
            values = self.model.value(critic_obs)
            b.actor_obs[t].copy_(actor_obs)
            b.critic_obs[t].copy_(critic_obs)
            b.actions[t].copy_(actions)
            b.mu[t].copy_(dist.mean)
            b.sigma[t].copy_(dist.stddev)
            b.log_probs[t].copy_(dist.log_prob(actions).sum(-1))
            b.values[t].copy_(values)

            obs, rewards, dones, extras = self.env.step(actions)
            dones = dones.float()
            # The env already reset timed-out robots, so V(s_t) stands in for the value of the lost next state.
            b.rewards[t].copy_(rewards + self.gamma * values * extras["time_outs"].float())
            b.dones[t].copy_(dones)

            self.cur_rew += rewards
            self.cur_len += 1
            ids = dones.nonzero().squeeze(-1)
            self.rew_hist.extend(self.cur_rew[ids].tolist())
            self.len_hist.extend(self.cur_len[ids].tolist())
            self.cur_rew[ids] = 0.0
            self.cur_len[ids] = 0.0
            if "episode" in extras:
                ep_infos.append(extras["episode"])

        last_values = self.model.value(ActorCritic.split_obs(obs)[1])
        adv, returns = compute_gae(b.rewards, b.values, b.dones, last_values, self.gamma, self.lam)
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalized over the whole rollout
        self.returns = returns
        return obs, ep_infos

    # ------------------------------------------------------------------ update
    def update(self):
        b, m = self.buf, self.model
        flat = lambda x: x.flatten(0, 1)
        actor_obs, critic_obs, actions = flat(b.actor_obs), flat(b.critic_obs), flat(b.actions)
        old_mu, old_sigma, old_logp = flat(b.mu), flat(b.sigma), flat(b.log_probs)
        old_values, returns, advantages = flat(b.values), flat(self.returns), flat(self.advantages)

        n = actor_obs.shape[0]
        mb_size = n // self.n_mb
        perm = torch.randperm(self.n_mb * mb_size)  # one shuffle reused every epoch, as in rsl_rl
        stats = collections.defaultdict(float)
        for _ in range(self.epochs):
            for i in range(self.n_mb):
                idx = perm[i * mb_size:(i + 1) * mb_size]
                dist = m.dist(actor_obs[idx])
                logp = dist.log_prob(actions[idx]).sum(-1)
                values = m.value(critic_obs[idx])
                entropy = dist.entropy().sum(-1).mean()

                # adaptive learning rate: keep the policy change per minibatch near desired_kl
                with torch.no_grad():
                    kl = gaussian_kl(old_mu[idx], old_sigma[idx], dist.mean, dist.stddev).mean()
                    if kl > 2.0 * self.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.lr

                ratio = torch.exp(logp - old_logp[idx])
                if not self.debug_checked:  # nothing has changed yet, so the new policy must equal the old one
                    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), ratio.abs().max()
                    self.debug_checked = True
                adv = advantages[idx]
                surrogate = torch.max(-adv * ratio, -adv * ratio.clamp(1 - self.clip, 1 + self.clip)).mean()

                v_old = old_values[idx]
                v_clipped = v_old + (values - v_old).clamp(-self.clip, self.clip)
                value_loss = torch.max((values - returns[idx]) ** 2, (v_clipped - returns[idx]) ** 2).mean()

                loss = surrogate + self.vf_coef * value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.actor_parameters(), self.max_grad)
                nn.utils.clip_grad_norm_(m.critic.parameters(), self.max_grad)
                self.optimizer.step()

                stats["surrogate"] += surrogate.item()
                stats["value"] += value_loss.item()
                stats["entropy"] += entropy.item()

        # Normalizers move only after the update, so the stored log-probs stay valid during it.
        m.actor.norm.update(actor_obs)
        m.critic_norm.update(critic_obs)
        return {k: v / (self.epochs * self.n_mb) for k, v in stats.items()}

    # -------------------------------------------------------------------- loop
    def learn(self, num_iterations, init_at_random_ep_len=True):
        if init_at_random_ep_len:  # spread episode ends out so resets don't happen in lockstep
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        if self.log_dir and self.writer is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)
        obs = self.env.get_observations()
        start, end = self.it, self.it + num_iterations
        for it in range(start, end):
            self.it = it
            t0 = time.time()
            obs, ep_infos = self.rollout(obs)
            t1 = time.time()
            losses = self.update()
            t2 = time.time()
            self.log(it, losses, ep_infos, t1 - t0, t2 - t1)
            if self.writer and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
        if self.writer:
            self.save(os.path.join(self.log_dir, f"model_{self.it}.pt"))
            self.writer.flush()

    def log(self, it, losses, ep_infos, collect_s, learn_s):
        fps = self.T * self.env.num_envs / (collect_s + learn_s)
        mean_rew = sum(self.rew_hist) / len(self.rew_hist) if self.rew_hist else 0.0
        mean_len = sum(self.len_hist) / len(self.len_hist) if self.len_hist else 0.0
        if self.writer:
            w = self.writer
            for k in ep_infos[0] if ep_infos else []:
                w.add_scalar(f"Episode/{k}", sum(e[k] for e in ep_infos) / len(ep_infos), it)
            for k, v in losses.items():
                w.add_scalar(f"Loss/{k}", v, it)
            w.add_scalar("Loss/learning_rate", self.lr, it)
            w.add_scalar("Policy/mean_std", self.model.std.mean().item(), it)
            w.add_scalar("Perf/total_fps", fps, it)
            w.add_scalar("Perf/collection_time", collect_s, it)
            w.add_scalar("Perf/learning_time", learn_s, it)
            if self.rew_hist:
                w.add_scalar("Train/mean_reward", mean_rew, it)
                w.add_scalar("Train/mean_episode_length", mean_len, it)
        print(f"it {it:5d} | reward {mean_rew:7.2f} | ep len {mean_len:6.1f} | value {losses['value']:.4f} "
              f"| surr {losses['surrogate']:+.4f} | lr {self.lr:.1e} | std {self.model.std.mean().item():.2f} "
              f"| {fps:,.0f} steps/s")

    # --------------------------------------------------------------- save/load
    def save(self, path):
        torch.save({"custom_ppo": True, "dims": self.dims, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "iter": self.it, "lr": self.lr}, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=False)
        if not ckpt.get("custom_ppo"):
            raise ValueError(f"{path} is not a train_ppo.py checkpoint (rsl_rl checkpoints load with train.py)")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr, self.it = ckpt["lr"], ckpt["iter"] + 1

    def export_policy(self, log_dir, filename="policy.pt"):
        """TorchScript actor: raw (N, 47) observation -> action mean, normalizer included."""
        os.makedirs(log_dir, exist_ok=True)
        torch.jit.script(self.model.actor.eval()).save(os.path.join(log_dir, filename))
        self.model.actor.train()


def load_actor(path, cfg):
    """Rebuild the deterministic actor from a train_ppo.py checkpoint (used by play.py)."""
    ckpt = torch.load(path, weights_only=False)
    model = ActorCritic(*ckpt["dims"], cfg["actor"]["hidden_dims"], cfg["critic"]["hidden_dims"],
                        cfg["actor"]["distribution_cfg"]["init_std"])
    model.load_state_dict(ckpt["model"])
    return model.actor.eval()
