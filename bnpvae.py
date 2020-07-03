import pdb
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import torch.nn.functional as F

from torch.distributions import constraints


def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


class DPMM(nn.Module):
    def __init__(self, latent_dim=2, num_T=2, num_obs=1000, init_prior=None):
        super(DPMM, self).__init__()
        self.T = num_T
        self.latent_dim = latent_dim
        self.alpha = 1.
        self.num_obs = num_obs

        self.alpha_q = nn.Parameter(dist.Uniform(0, 2).sample([self.T - 1]))
        self.pi_c = nn.Parameter(init_prior["pi_c"])
        self.mu_c = nn.Parameter(init_prior["mu_c"])
        self.sd_q1 = nn.Parameter(init_prior["sd_q1"])
        self.sd_q2 = nn.Parameter(init_prior["sd_q2"])
        self.component_sd = None
        self.component_weights = None
        self.component_loc = None

    def model(self, data, batch_idx):
        with pyro.plate("beta_plate", self.T - 1):
            beta = pyro.sample("beta", dist.Beta(1, self.alpha))

        with pyro.plate("mu_plate", self.T):
            mu_sd = pyro.sample("musd", dist.InverseGamma(torch.ones_like(self.sd_q1),
                                                          torch.ones_like(self.sd_q2)).to_event(1))
            mu_c = pyro.sample("mu", dist.Normal(self.mu_c,
                                                 mu_sd*torch.ones_like(self.mu_c)).to_event(1))

        with pyro.plate("data", size=self.num_obs, subsample=batch_idx):
            ys = pyro.sample(f"cat", dist.Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
            pyro.sample(f"obs", dist.Normal(mu_c[ys], mu_sd[ys]).to_event(1), obs=data[batch_idx])

    def guide(self, data, batch_idx):

        alpha_q = pyro.param('alpha_q', self.alpha_q, constraint=constraints.positive)
        phi = pyro.param('pi_q', self.pi_c, constraint=constraints.simplex)
        mu_q = pyro.param("mu_q", self.mu_c)
        sd_q1 = pyro.param('sd_q1', self.sd_q1)
        sd_q2 = pyro.param('sd_q2', self.sd_q2)

        with pyro.plate("beta_plate", self.T - 1):
            f_beta = pyro.sample("beta", dist.Beta(torch.ones(self.T - 1), alpha_q))

        with pyro.plate("mu_plate", self.T):
            mu_sd = pyro.sample("musd", dist.InverseGamma(sd_q1, sd_q2).to_event(1))
            mu_c = pyro.sample("mu", dist.Normal(mu_q, mu_sd).to_event(1))

        with pyro.plate("data", size=self.num_obs, subsample=batch_idx):
            f_cat = pyro.sample(f"cat", dist.Categorical(phi[batch_idx]))

    def update_component_stats(self, posterior_samples):
        self.update_component_loc(torch.mean(posterior_samples["mu"], dim=0))
        self.update_component_sd(torch.mean(posterior_samples["musd"], dim=0))
        self.update_component_weights(torch.mean(posterior_samples["beta"], dim=0))
        # sort_val, sort_idx = torch.sort(self.component_weights, descending=True)
        # self.mu_c.data = self.mu_c.data[sort_idx]
        # self.sd_q1.data = self.sd_q1.data[sort_idx]
        # self.sd_q2.data = self.sd_q2.data[sort_idx]
        # self.pi_c.data = self.pi_c[:, sort_idx]
        # self.component_loc = self.component_loc[sort_idx]
        # self.component_sd = self.component_sd[sort_idx]
        # self.component_weights = self.component_weights[sort_idx]

    def update_component_sd(self, data):
        self.component_sd = data.detach()

    def update_component_weights(self, data):
        self.component_weights = mix_weights(data.detach())

    def update_component_loc(self, data):
        self.component_loc = data.detach()

    def check_dist(self, comp_idx):
        pass

    def likelihood(self, x_samples):
        assert x_samples.size(1) == self.latent_dim
        new_data = x_samples.repeat(1, self.T).unsqueeze(2)
        component_lik = dist.Normal(self.component_loc, self.component_sd).to_event(1).log_prob(new_data).exp()
        mixture_lik = (self.component_weights * component_lik).sum(-1)
        return mixture_lik
