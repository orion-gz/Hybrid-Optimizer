import torch
import torch.nn as nn
import random


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer.

    Wraps a base optimizer and performs a two-step update: first perturb the
    weights toward the sharpest direction, then compute the true gradient and
    apply the base optimizer update. Call first_step and second_step explicitly
    in your training loop instead of step().
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights in the sharpness direction and save original weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = (group["rho"] / (grad_norm + 1e-12)).to(group["params"][0].device)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original weights and apply the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM doesn't work like vanilla optimizers. "
            "Please call first_step and second_step in your training loop."
        )

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norm_val = ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                norms.append(norm_val.to(shared_device))
        if len(norms) == 0:
            return torch.tensor(0.0).to(shared_device)
        return torch.norm(torch.stack(norms), p=2)


class ESAM(torch.optim.Optimizer):
    """Efficient Sharpness-Aware Minimization optimizer.

    An efficient variant of SAM that randomly selects a subset of parameters
    for perturbation (controlled by beta) and runs the second step only on
    the top-gamma loss-increasing samples.
    """

    def __init__(self, params, base_optimizer, rho=0.05, beta=0.5, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        assert 0 < beta <= 1.0, f"Invalid beta: {beta}"
        defaults = dict(rho=rho, beta=beta, adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Randomly perturb a beta-fraction of parameters and save originals."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            beta = group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                # stochastic weight perturbation: select with probability beta
                if random.random() < beta:
                    e_w = ((torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale / beta)
                else:
                    e_w = torch.zeros_like(p.grad)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original weights and apply the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"].clone()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            torch.norm((torch.abs(p) if group["adaptive"] else 1.0) * p.grad, p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0).to(shared_device)
        return torch.norm(torch.stack(norms), p=2)